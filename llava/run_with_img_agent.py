from __future__ import annotations  # Enable newer type annotation syntax for Python 3.8+
import argparse
import torch
import os
import sys

# Add parent directory to path for importing modules
sys.path.insert(0, '../')

# Import llama_adapter_v21 modules
import llama
import cv2

import requests
from PIL import Image
from io import BytesIO
import re

from torchvision.transforms import RandomResizedCrop, RandomRotation, RandomAffine, ColorJitter 
from scipy.stats import entropy
import statistics

import torch.nn as nn
import logging
logging.basicConfig(level='ERROR')
import numpy as np
from pathlib import Path
import torch
import zlib
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from eval import *
from metric_util import get_text_metric, get_img_metric, save_output, convert, get_meta_metrics,get_img_metric_agent
import os
from typing import Optional
import os
from metrics_plugin import register_metric
from agent_mia.agent.base_agent import (
    BaseAgent,
    build_strategy_reference,
    parse_metric_definitions,
    evaluate_metrics,
    save_strategy_bank,
    load_strategy_bank,
)
# Custom prompt templates for metric generation and judgment.
from agent_mia.agent.prompt import prompt_new, judge_prompt_template, optimization_prompt_template
import json
from datetime import datetime
import re

# Global model context
model = None
preprocess = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "agent_round_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dedicated strategy bank path for image experiments
IMG_STRATEGY_BANK_PATH = os.path.join(CURRENT_DIR, "agent_mia", "agent", "strategy_bank_img.json")











def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--llama_path",
                        default=None,
                        help=(
                            "Path to LLaMA model weights directory. "
                            "If not provided, a cache directory under ~/.cache/vl-mia/llama_model_weights "
                            "will be used."
                        ))
    parser.add_argument("--num_gen_token", type=int, default=32)
    parser.add_argument("--gpu_id",type=int,default=0)
    parser.add_argument("--dataset", type=str, default='img_Flickr')
    parser.add_argument("--local_dataset_path", type=str, default=None,
                        help="path to local dataset directory. If specified, will load from local path instead of HuggingFace.")
    parser.add_argument("--model_type", type=str, default="LORA-BIAS-7B-v21",
                        help="Model type: BIAS-7B, LORA-BIAS-7B, LORA-BIAS-7B-v21")
    parser.add_argument("--output_dir", type=str, default="image_MIA")
    parser.add_argument("--severity", type=int, default=6)
    parser.add_argument("--metric_json", type=str, default=None)
    parser.add_argument(
        "--instruction_prompt",
        type=str,
        default="Describe this image concisely."
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--enable-optimization", action="store_true", default=True, 
                        help="Whether to enable the post-exploration optimization phase (default: enabled).")
    parser.add_argument("--disable-optimization", action="store_false", dest="enable_optimization",
                        help="Disable the optimization phase.")
    parser.add_argument("--top-k-optimize", type=int, default=5,
                        help="Top-k strategies to optimize during the optimization phase (default: 5).")
    parser.add_argument("--enable-reverse", action="store_true", default=True,
                        help="Whether to enable reverse-metric improvement (default: enabled).")
    parser.add_argument("--disable-reverse", action="store_false", dest="enable_reverse",
                        help="Disable reverse-metric improvement.")
    parser.add_argument("--reverse-strategy", type=str, default="immediate", choices=["skip", "immediate", "defer"],
                        help=(
                            "Reverse-metric handling strategy: "
                            "skip=only update code without rerunning, "
                            "immediate=rerun immediately, "
                            "defer=defer unified processing to the final round (default: defer)."
                        ))
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def load_image(image_file):
    if isinstance(image_file, Image.Image):  
        return image_file.convert("RGB")  
    
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    
    
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def generate_text(model, preprocess, img, text, gpu_id, num_gen_token):
    device = 'cuda:{}'.format(gpu_id)
    prompt = llama.format_prompt(text)
    
    if isinstance(img, Image.Image):
        img_tensor = preprocess(img).unsqueeze(0).to(device)
    else:
        img_pil = Image.open(img).convert('RGB')
        img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
    
    output_text = model.generate(img_tensor, [prompt], max_gen_len=num_gen_token, temperature=0, device=device)[0]
    
    return output_text

def evaluate_data(model, preprocess, test_data, text, gpu_id, num_gen_token):
    """
    Batch-evaluate all samples in the dataset and run the main member-inference attack pipeline.

    This is the entry point for the VL-MIA attack on images. For each sample in the test set,
    it performs full analysis including text generation and multi-dimensional metric computation.

    Args:
        model: Loaded multimodal model (llama_adapter_v21).
        preprocess: Image preprocessor.
        test_data: Test dataset containing images and metadata.
        text: Instruction prompt.
        gpu_id: GPU device index.
        num_gen_token: Maximum number of tokens to generate for text.

    Returns:
        all_output: List of inference results for all samples.
    """
    print(f"all data size: {len(test_data)}")
    all_output = []
    test_data = test_data

    for ex in tqdm(test_data): 
        description = generate_text(model, preprocess, ex['image'], text, gpu_id, num_gen_token)
        # description = ''
        new_ex = inference(model, preprocess, ex['image'], text, description, ex, gpu_id)

        all_output.append(new_ex)

    return all_output

def load_latest_eval_guidance(output_dir: str) -> Optional[str]:
    """
    Load the latest metrics_eval_round_*.json from output_dir and build a readable
    guidance text (including useful_insights, next_round_strategy, and summary).
    """
    import os, json

    files = [
        f for f in os.listdir(output_dir)
        if f.startswith("metrics_eval_round_") and f.endswith(".json")
    ]
    if not files:
        return None

    files.sort()
    latest = files[-1]
    path = os.path.join(output_dir, latest)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        return None

    ui = data.get("useful_insights", {})
    nrs = data.get("next_round_strategy", {})
    summary = data.get("summary", "")

    guidance_text = "[Previous Round Evaluation Guidance]\n"

    if ui:
        guidance_text += "\n[Effective Observations]\n"
        for k, v in ui.items():
            guidance_text += f"- {k}: {v}\n"

    if nrs:
        guidance_text += "\n[Next-Round Strategy Suggestions]\n"
        for k, v in nrs.items():
            guidance_text += f"- {k}: {v}\n"

    if summary:
        guidance_text += f"\n[Summary]\n{summary}\n"

    return guidance_text.strip()

def load_all_used_metric_names(output_dir: str) -> list:
    """
    Load all metric names generated in historical rounds from output_dir.
    Scan all metrics_round_*.json files and collect the 'name' field from each metric.
    """
    import os, json

    if not os.path.exists(output_dir):
        return []

    used_names = []
    files = [
        f for f in os.listdir(output_dir)
        if f.startswith("metrics_round_") and f.endswith(".json")
    ]

    for filename in files:
        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            reply = data.get("reply", "")
            if reply:
                metric_defs = parse_metric_definitions(reply)
                for metric in metric_defs:
                    name = metric.get("name")
                    if name and name not in used_names:
                        used_names.append(name)
        except Exception:
            continue

    return sorted(used_names)

def consolidate_method_auc_reports(base_output_dir: str) -> str:
    """
    Consolidate auc.txt files from each method subdirectory (output by fig_fpr_tpr_img)
    into a single summary file.
    """
    os.makedirs(base_output_dir, exist_ok=True)
    summary_path = os.path.join(base_output_dir, "auc_summary.txt")
    lines = []

    if os.path.exists(base_output_dir):
        for item in sorted(os.listdir(base_output_dir)):
            method_dir = os.path.join(base_output_dir, item)
            if not os.path.isdir(method_dir):
                continue
            auc_file = os.path.join(method_dir, "auc.txt")
            if not os.path.exists(auc_file):
                continue

            with open(auc_file, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        lines.append(f"[{item}] {stripped}")

    if not lines:
        lines.append("No metrics were generated for this round.")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return summary_path

@torch.inference_mode()
def logits_forward(model, tokens, visual_query):
    _bsz, seqlen = tokens.shape

    h = model.llama.tok_embeddings(tokens)
    freqs_cis = model.llama.freqs_cis.to(h.device)
    freqs_cis = freqs_cis[:seqlen]
    mask = None
    mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
    mask = torch.triu(mask, diagonal=0 + 1).type_as(h)

    for layer in model.llama.layers[:-1 * model.query_layer]:
        h = layer(h, 0, freqs_cis, mask)

    adapter = model.adapter_query.weight.reshape(model.query_layer, model.query_len, -1).unsqueeze(1)
    adapter_index = 0
    for layer in model.llama.layers[-1 * model.query_layer:]:
        dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
        dynamic_adapter = dynamic_adapter + visual_query
        h = layer(h, 0, freqs_cis, mask, dynamic_adapter)
        adapter_index = adapter_index + 1

    h = model.llama.norm(h)
    output = model.llama.output(h)

    assert model.llama.vocab_size == 32000

    return output

def inference(model, preprocess, img_path, text, description, ex, gpu_id):
    """
    Core per-sample analysis function for the Vision-Language Membership Inference Attack (VL-MIA).

    This function runs the model on a single image and its text description under multiple viewpoints
    and perturbations, computing probabilistic statistics for different input regions
    (image, instruction, description) to detect overfitting behavior. Training samples
    are often more sensitive to small perturbations and have more concentrated probability
    distributions, which can be exploited to distinguish members from non-members.

    Args:
        model: Loaded multimodal model (llama_adapter_v21).
        preprocess: Image preprocessor.
        img_path: Image path or PIL.Image object.
        text: Instruction prompt, e.g. "Describe this image concisely."
        description: Caption text generated by the model.
        ex: Sample dictionary (containing image and metadata) that will be updated.
        gpu_id: GPU index.

    Returns:
        ex: Updated sample dictionary. A new field ex["pred"] is added, which stores
            metric values for each target region ('inst_desp', 'inst', 'desp').
    """

    # Target regions to be evaluated:
    # 'inst_desp'  -> instruction + description as a whole
    # 'inst'       -> instruction only
    # 'desp'       -> description only
    goal_parts = ['inst_desp', 'inst', 'desp']
    all_pred = {}  # Store computed metrics for each target region

    # ========== Step 1. Load image ==========
    if isinstance(img_path, Image.Image):
        image = img_path.convert('RGB')  # Already a PIL.Image object
    else:
        image = Image.open(img_path).convert('RGB')  # Load from file path

    # ========== Step 2. Define and generate multiple augmented image variants ==========
    # These transformations simulate small perturbations (crop, rotation, affine, color jitter)
    # to test how sensitive the model output is to perturbations (signs of overfitting/memorization).

    # Randomly crop to a fixed size
    transform1 = RandomResizedCrop(size=(256, 256))
    aug1 = transform1(image)

    # Random rotation up to 45 degrees
    transform2 = RandomRotation(degrees=45)
    aug2 = transform2(image)

    # Random affine transformation (rotation, translation, scaling)
    transform3 = RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.75, 1.25))
    aug3 = transform3(image)

    # Random color jitter (brightness, contrast, saturation, hue)
    transform4 = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    aug4 = transform4(image)

    # ========== Step 3. Run multiple inferences for each target region ==========
    # Loop over regions ('img', 'inst_desp', 'inst', 'desp')
    for part in goal_parts:
        pred = {}  # Store statistics for the current region

        # (1) Metrics for the original image
        metrics = mod_infer(model, preprocess, image, text, description, gpu_id, part)

        # (2~5) Metrics for the four augmented images
        metrics1 = mod_infer(model, preprocess, aug1, text, description, gpu_id, part)
        metrics2 = mod_infer(model, preprocess, aug2, text, description, gpu_id, part)
        metrics3 = mod_infer(model, preprocess, aug3, text, description, gpu_id, part)
        metrics4 = mod_infer(model, preprocess, aug4, text, description, gpu_id, part)

        # ========== Step 4. Extract various metrics ==========
        # Read probabilities, entropies, Rényi entropies, etc. from mod_infer().
        # These values reflect the model's confidence and certainty about the inputs.

        # Log-probability distributions for the four augmented images
        aug1_prob = metrics1['log_probs']
        aug2_prob = metrics2['log_probs']
        aug3_prob = metrics3['log_probs']
        aug4_prob = metrics4['log_probs']

        # Metrics for the original image
        ppl = metrics["ppl"]                        # Perplexity
        all_prob = metrics["all_prob"]              # Probability matrix over all tokens
        p1_likelihood = metrics["loss"]             # Negative log-likelihood (loss)
        entropies = metrics["entropies"]            # Shannon entropy
        mod_entropy = metrics["modified_entropies"] # Adjusted entropy (e.g., denoised or regularized)
        max_p = metrics["max_prob"]                 # Maximum token probability
        org_prob = metrics["probabilities"]         # Original probability distribution
        log_probs = metrics["log_probs"]            # Log-probability distribution
        gap_p = metrics["gap_prob"]                 # Probability gap between top-2 tokens
        renyi_05 = metrics["renyi_05"]              # Rényi entropy (α=0.5)
        renyi_2 = metrics["renyi_2"]                # Rényi entropy (α=2)
        mod_renyi_05 = metrics["mod_renyi_05"]      # Adjusted Rényi entropy (α=0.5)
        mod_renyi_2 = metrics["mod_renyi_2"]        # Adjusted Rényi entropy (α=2)
        #########
        loss_var = metrics["loss_var"]              # Loss variance

        # ========== Step 5. Compute aggregated image-level metrics ==========
        # get_img_metric() would fuse the original and augmented results to extract
        # multi-perspective features (robustness, stability, confidence change),
        # acting as a "signal aggregator" for membership inference.
        # pred = get_img_metric(
        #     ppl, all_prob, p1_likelihood,
        #     entropies, mod_entropy, max_p,
        #     org_prob, gap_p,
        #     renyi_05, renyi_2,
        #     log_probs,
        #     aug1_prob, aug2_prob, aug3_prob, aug4_prob,
        #     mod_renyi_05, mod_renyi_2,
        # 
        # )

        base_keys = {
            "ppl",
            "all_prob",
            "loss",
            "entropies",
            "modified_entropies",
            "max_prob",
            "probabilities",
            "log_probs",
            "gap_prob",
            "renyi_05",
            "renyi_2",
            "mod_renyi_05",
            "mod_renyi_2",
            "loss_var",
        }

        # Store results for the current target region
        
        custom = {k: v for k, v in metrics.items() if k not in base_keys}
        pred = get_img_metric_agent(custom)
        all_pred[part] = pred

    # ========== Step 6. Aggregate and free GPU memory ==========
    ex["pred"] = all_pred  # Attach metrics for all regions to the sample dict
    torch.cuda.empty_cache()  # Clear GPU cache to avoid OOM

    # Return the sample with attached prediction metrics
    return ex

def mod_infer(model, preprocess, img, instruction, description, gpu_id, goal):
    device='cuda:{}'.format(gpu_id)

    if isinstance(img, Image.Image):
        img_tensor = preprocess(img).unsqueeze(0).to(device)
    else:
        img_pil = Image.open(img).convert('RGB')
        img_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    with torch.cuda.amp.autocast():
        visual_query = model.forward_visual(img_tensor)

    prompt = llama.format_prompt(instruction) + description

    prompt_t = model.tokenizer.encode(prompt, bos=True, eos=False)

    tokens = torch.tensor(prompt_t).to(device).long().unsqueeze(0) 

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            logits = logits_forward(model, tokens, visual_query)

    input_ids = tokens[0]
    
    descp_encoding = model.tokenizer.encode(description, bos=False, eos=False)

    goal_slice_dict = {
        'inst_desp' : slice(0, None),
        'inst' : slice(0,-len(descp_encoding)),
        'desp' : slice(-len(descp_encoding),None)
        } 

    target_slice = goal_slice_dict[goal]

    logits_slice = logits[0,target_slice,:]

    input_ids = input_ids[target_slice]

    probabilities = torch.nn.functional.softmax(logits_slice, dim=-1)
    log_probabilities = torch.nn.functional.log_softmax(logits_slice, dim=-1)
    
    return get_meta_metrics(input_ids, probabilities, log_probabilities)

def extract_json_content(text: str) -> str:
    """
    Extract raw JSON text from a possible ```json ... ``` block returned by an LLM.
    If no JSON object with curly braces is found, return the text as-is.
    """
    if not isinstance(text, str):
        return text

    fence_pattern = r"```(?:json)?(.*?)```"
    match = re.search(fence_pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)
    return text

def load_agent_metrics(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    reply_str = data["reply"]
    reply_str = extract_json_content(reply_str)
    reply_json = json.loads(reply_str)

    for m in reply_json.get("metrics", []):
        register_metric(m["name"], m["code"])

def parse_auc_file(auc_file_path: str) -> list[dict]:
    """
    Parse an auc.txt file and extract metric name and accuracy for each metric.

    Returns:
        A list of metric info dicts, each with: name, accuracy, auc, tpr_at_5_fpr, full_name.
    """
    metrics_info = []
    
    if not os.path.exists(auc_file_path):
        print(f"[Warning] auc.txt not found: {auc_file_path}")
        return metrics_info
    
    with open(auc_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse either:
            #   [method] metric_name   AUC 0.xxxx, Accuracy 0.xxxx, TPR@5%FPR of 0.xxxx
            # or:
            #   metric_name   AUC 0.xxxx, Accuracy 0.xxxx, TPR@5%FPR of 0.xxxx
            # Note: there may be spaces in "TPR@5% FPR"
            match = re.match(r'(\[[^\]]+\]\s+)?(.+?)\s+AUC\s+([\d.]+),\s+Accuracy\s+([\d.]+),\s+TPR@5%[^0-9]*([\d.]+)', line)
            if match:
                method_prefix = match.group(1)  # May be None
                metric_name_full = match.group(2).strip()
                auc = float(match.group(3))
                accuracy = float(match.group(4))
                tpr_at_5_fpr = float(match.group(5))
                
                # Extract clean metric name (remove method prefix if any)
                metric_name = metric_name_full
                method = None
                if method_prefix:
                    method = method_prefix.strip('[] ').strip()
                    # If the metric name itself also starts with the method prefix, strip it
                    if metric_name.startswith(f"[{method}]"):
                        metric_name = metric_name[len(f"[{method}]"):].strip()
                
                metrics_info.append({
                    "name": metric_name,           # Clean metric name
                    "full_name": metric_name_full, # Full name (may include method)
                    "method": method,
                    "accuracy": accuracy,
                    "auc": auc,
                    "tpr_at_5_fpr": tpr_at_5_fpr
                })
    
    return metrics_info

def generate_reversed_metric(agent: BaseAgent, metric_name: str, original_code: str, accuracy: float, auc: float) -> Optional[dict]:
    """
    Use judge_prompt_template to generate reversed metric code.

    Args:
        agent: BaseAgent instance.
        metric_name: Original metric name.
        original_code: Original metric code.
        accuracy: Original metric accuracy.
        auc: Original metric AUC.

    Returns:
        If successful, a dict containing the reversed metric definition; otherwise None.
    """
    # Build judge prompt
    judge_prompt = (
        judge_prompt_template
        + "\n\n==== Metric to be evaluated ====\n"
        + f"Metric Name: {metric_name}\n"
        + f"Current Accuracy: {accuracy:.4f}\n"
        + f"Current AUC: {auc:.4f}\n"
        + f"\nOriginal code:\n```python\n{original_code}\n```\n"
        + "Please decide whether this metric should be reversed. "
          "If Accuracy <= 0.52, generate a reversed version of the code."
    )
    
    try:
        reply = agent.ask(judge_prompt)
        cleaned_reply = extract_json_content(reply)
        reply_json = json.loads(cleaned_reply)
        
        metrics = reply_json.get("metrics", [])
        if metrics and len(metrics) > 0:
            reversed_metric = metrics[0]
            # Ensure reversed metric name includes original name and is marked as reversed
            reversed_name = f"{metric_name}_reversed"
            reversed_metric["name"] = reversed_name
            reversed_metric["original_name"] = metric_name
            return reversed_metric
        else:
            print(f"[Warning] Failed to parse reversed metric from LLM reply: {metric_name}")
            return None
    except Exception as e:
        print(f"[Error] Failed to generate reversed metric ({metric_name}): {e}")
        return None

def ensure_complete_definition(metric_def: dict) -> dict:
    """
    Ensure that a metric definition contains all required fields by filling missing ones.

    Args:
        metric_def: Metric definition dictionary.

    Returns:
        Completed metric definition dictionary.
    """
    if not isinstance(metric_def, dict):
        return metric_def
    
    # Define required fields and their default values
    required_fields = {
        "name": "",
        "formula": "",
        "description": "",
        "code": "",
        "expected_behavior": ""
    }
    
    # Ensure all required fields exist
    complete_def = metric_def.copy()
    for field, default_value in required_fields.items():
        if field not in complete_def or complete_def[field] is None:
            complete_def[field] = default_value
    
    return complete_def


def ensure_complete_definitions(metric_definitions: list[dict]) -> list[dict]:
    """
    Ensure that all metric definitions in a list contain all required fields.

    Args:
        metric_definitions: List of metric definition dicts.

    Returns:
        List of completed metric definitions.
    """
    if not isinstance(metric_definitions, list):
        return metric_definitions
    
    return [ensure_complete_definition(md) for md in metric_definitions]


def try_reverse_weak_metrics(agent: BaseAgent, 
                              args, 
                              auc_path: str, 
                              metric_definitions: list[dict],
                              round_idx: int):
    """
    Simplified reverse-metric handling:
    1. Find metrics with accuracy ≤ 0.501.
    2. Generate reversed versions (keeping the same metric name).
    3. Overwrite the original metric code.
    4. Do not rerun experiments or compare reversed vs original metrics here.

    Returns:
        (updated_metric_definitions, reversed_metrics_only, was_updated)
        - updated_metric_definitions: Full list of metric definitions (for evaluation).
        - reversed_metrics_only: List containing only reversed metrics (for saving/reruns).
        - was_updated: Whether any metric was updated.
    """

    print(f"\n[Round {round_idx}] Running simplified reverse-metric check...")

    # Parse auc.txt
    metrics_info = parse_auc_file(auc_path)
    if not metrics_info:
        print("[Info] auc.txt contains no valid metrics.")
        return metric_definitions, [], False

    # Aggregate by metric name and keep the worst accuracy per metric
    metrics_by_name = {}
    for m in metrics_info:
        name = m["name"]
        if name not in metrics_by_name or m["accuracy"] < metrics_by_name[name]["accuracy"]:
            metrics_by_name[name] = m

    # Collect weak metrics
    weak_metrics = [m for m in metrics_by_name.values() if m["accuracy"] <= 0.501]
    if not weak_metrics:
        print("[Info] No metrics need to be reversed.")
        return metric_definitions, [], False

    print(f"[Info] {len(weak_metrics)} metrics need to be reversed.")

    updated = False
    reversed_metrics_only = []  # Only metrics that need to be reversed

    for m in weak_metrics:
        name = m["name"]
        print(f"  - Processing weak metric: {name}")

        # Find corresponding metric definition
        original = None
        for md in metric_definitions:
            if md.get("name") == name:
                original = md
                break
        if not original:
            print(f"[Warning] Metric definition missing, skipping: {name}")
            continue

        original_code = original.get("code", "")
        if not original_code:
            print("[Warning] Metric has no 'code' field, skipping.")
            continue

        # Ask LLM to generate reversed metric (same name, code only changes)
        judge_prompt = (
            judge_prompt_template
            + "\n\n===== Weak Metric Info =====\n"
            + f"Metric Name: {name}\n"
            + f"Accuracy: {m['accuracy']:.4f}\n"
            + f"AUC: {m['auc']:.4f}\n"
            + "\nOriginal Code:\n```python\n"
            + original_code
            + "\n```\n"
            + "\n[CRITICAL: Must reverse the metric logic]\n"
            + "Reverse the return value by negating it. Requirements:\n"
            + "1. Change `return value` to `return -value`\n"
            + "2. Change `return expression` to `return -(expression)`\n"
            + "3. All return statements must be negated\n"
            + "4. Keep function structure, variables, and logic unchanged\n"
            + "5. Keep the metric name unchanged (no '_reversed' suffix)\n"
            + "\nExample:\n"
            + "Original: `return float(total_dispersion / count)`\n"
            + "Reversed: `return -float(total_dispersion / count)`\n"
            + "\nEnsure the returned code is logically reversed (negated value)."
        )

        try:
            reply = agent.ask(judge_prompt)
            reversed_json = json.loads(extract_json_content(reply))
            reversed_metric = reversed_json["metrics"][0]

            # Only update the 'code' field while preserving name, formula, description, expected_behavior, etc.
            # Ensure the original definition is complete.
            original = ensure_complete_definition(original)
            original["code"] = reversed_metric.get("code", original.get("code", ""))
            # If the reversed metric includes new description or expected_behavior, they could be updated too,
            # but for consistency we only update code here.
            updated = True

            # Append updated metric into reversed_metrics_only with a deep copy
            import copy
            reversed_metric_copy = copy.deepcopy(original)
            # Ensure the copy is also complete
            reversed_metric_copy = ensure_complete_definition(reversed_metric_copy)
            reversed_metrics_only.append(reversed_metric_copy)

            print(f"    ✓ Original metric overridden: {name}")

        except Exception as e:
            print(f"[Error] Failed to generate reversed metric for {name}: {e}")

    return metric_definitions, reversed_metrics_only, updated



def run_img_mia_once(args, metric_json_path: Optional[str] = None, round_idx: Optional[int] = None) -> str:
    """
    Run a single round of image-based MIA experiment and return the path of the
    aggregated auc.txt file.
    """
    global model, preprocess
    if model is None:
        raise RuntimeError("Model has not been initialized; please call main() to load the model first.")

    if metric_json_path is None:
        metric_json_path = getattr(args, "metric_json", None)

    if metric_json_path is not None:
        print(f"[MIA-IMG] Loading agent metrics from: {metric_json_path}")
        load_agent_metrics(metric_json_path)
    else:
        print("[MIA-IMG] No metric_json provided, using only built-in metrics.")

    num_gen_token = getattr(args, "num_gen_token", 32)
    base_output_dir = os.path.join(
        args.output_dir,
        args.dataset,
        f"gen_{num_gen_token}_tokens"
    )
    if round_idx is not None:
        output_dir = os.path.join(base_output_dir, f"round_{round_idx}")
    else:
        output_dir = base_output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    from datasets import load_from_disk
    # Support specifying dataset path via --local_dataset_path or MIA_IMAGE_DATASET_DIR
    dataset_path = getattr(args, "local_dataset_path", None) or os.environ.get("MIA_IMAGE_DATASET_DIR")
    if dataset_path is None:
        raise RuntimeError(
            "Dataset path is not set. Please provide --local_dataset_path or set MIA_IMAGE_DATASET_DIR."
        )
    dataset = load_from_disk(dataset_path)
    data = convert_huggingface_data_to_list_dic(dataset)

    instruction = getattr(args, "instruction_prompt", "Describe this image concisely.")

    all_output = evaluate_data(
        model,
        preprocess,
        data,
        instruction,
        args.gpu_id,
        num_gen_token,
    )

    fig_fpr_tpr_img(all_output, output_dir)
    summary_path = consolidate_method_auc_reports(output_dir)
    print(f"[MIA-IMG] AUC summary written to: {summary_path}")
    return summary_path


def run_one_round(agent: BaseAgent, args, round_idx: int, used_metric_names: Optional[list] = None):
    """
    Run one round of: metric generation -> MIA experiment -> result evaluation.
    """
    if used_metric_names is None:
        used_metric_names = []

    # Select n good and n bad strategies from the strategy bank (default: 3 each)
    n_good = 3  # number of good strategies
    n_bad = 3   # number of bad strategies
    strategy_reference = build_strategy_reference(
        n_good=n_good,
        n_bad=n_bad,
        path=IMG_STRATEGY_BANK_PATH,
    )
    eval_guidance = load_latest_eval_guidance(OUTPUT_DIR)

    used_metrics_block = ""
    if used_metric_names:
        used_metrics_text = "\n".join([f"- {name}" for name in used_metric_names])
        used_metrics_block = (
            "\n\n==============================\n"
            + "[List of Used Metrics (Please Avoid Duplication)]\n"
            + "The following metrics have been generated in previous rounds of this experiment, please do not generate the same metrics again:\n\n"
            + used_metrics_text
            + "\n\nPlease ensure that the metric names and calculation logic you generate are significantly different from the above metrics."
        )

    guidance_block = ""
    if eval_guidance:
        guidance_block = (
            "\n\n==============================\n"
            + eval_guidance
        )

    if eval_guidance:
        generation_prompt = (
            prompt_new
            + used_metrics_block
            + guidance_block
            + "\n\n==============================\n"
              "[Strategy Bank Reference (including strong and weak strategies to illustrate what works and what does not)]\n"
            + strategy_reference
        )
    else:
        generation_prompt = (
            prompt_new
            + used_metrics_block
            + "\n\n==============================\n"
              "[Strategy Bank Reference (including strong and weak strategies to illustrate what works and what does not)]\n"
            + strategy_reference
        )

    reply = agent.ask(generation_prompt)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_json_path = os.path.join(
        OUTPUT_DIR, f"metrics_round_{round_idx}_{timestamp}.json"
    )
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"prompt": generation_prompt, "reply": reply},
            f,
            ensure_ascii=False,
            indent=2,
        )

    metric_definitions = parse_metric_definitions(reply)
    if not metric_definitions:
        print("[Warning] No metric definitions were successfully parsed in this round; skipping.")
        return []

    # Ensure all metric definitions contain required fields
    metric_definitions = ensure_complete_definitions(metric_definitions)

    new_metric_names = []
    for metric in metric_definitions:
        name = metric.get("name")
        if name and name not in used_metric_names:
            new_metric_names.append(name)
            used_metric_names.append(name)

    print(f"[Round {round_idx}] New metrics generated in this round: {new_metric_names}")

    auc_path = run_img_mia_once(
        args,
        metric_json_path=metrics_json_path,
        round_idx=round_idx,
    )

    # Check and attempt to reverse metrics with accuracy <= 0.501
    enable_reverse = getattr(args, "enable_reverse", True)
    reverse_strategy = getattr(args, "reverse_strategy", "defer")
    
    if not enable_reverse:
        print(f"[Round {round_idx}] Reverse-metric improvement is disabled; skipping.")
        # Ensure metric definitions are complete
        complete_metric_definitions = ensure_complete_definitions(metric_definitions)
        # Evaluate metrics and persist them into the strategy bank
        eval_result = evaluate_metrics(
            agent=agent,
            metrics_text_path=auc_path,
            save_dir=OUTPUT_DIR,
            save_prefix=f"metrics_eval_round_{round_idx}",
            metric_definitions=complete_metric_definitions,
            strategy_store_path=IMG_STRATEGY_BANK_PATH,
            persist_strategies=True,
        )
        print(f"[Round {round_idx}] Structured evaluation result:")
        print(eval_result)
        return new_metric_names, metric_definitions, auc_path
    
    updated_metric_definitions, reversed_metrics_only, was_updated = try_reverse_weak_metrics(
        agent=agent,
        args=args,
        auc_path=auc_path,
        metric_definitions=metric_definitions,
        round_idx=round_idx,
    )
    
    # 评估指标并保存到策略库（使用更新后的指标定义，如果有反向更新）
    # 使用 updated_metric_definitions 而不是原始的 metric_definitions，以便保存更新后的指标代码
    final_metric_definitions = updated_metric_definitions if was_updated else metric_definitions
    # 确保指标定义完整
    complete_final_metric_definitions = ensure_complete_definitions(final_metric_definitions)
    
    eval_result = evaluate_metrics(
        agent=agent,
        metrics_text_path=auc_path,
        save_dir=OUTPUT_DIR,
        save_prefix=f"metrics_eval_round_{round_idx}",
        metric_definitions=complete_final_metric_definitions,
        strategy_store_path=IMG_STRATEGY_BANK_PATH,
        persist_strategies=True,
    )
    
    print(f"[Round {round_idx}] 评估结构化结果：")
    print(eval_result)
    
    if was_updated:
        if reverse_strategy == "skip":
            print(f"[Round {round_idx}] Reverse-metric improvement detected, strategy 'skip'; "
                  "updating code only without rerunning experiment.")
            # Save only reversed metrics, do not rerun
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            updated_metrics_json_path = os.path.join(
                OUTPUT_DIR, f"metrics_round_{round_idx}_updated_{timestamp}.json"
            )
            # Save only reversed metrics, not normal ones
            updated_reply = json.dumps({"metrics": reversed_metrics_only}, ensure_ascii=False, indent=2)
            with open(updated_metrics_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"prompt": "Updated metric definitions (reversed metrics only)", "reply": updated_reply},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            return new_metric_names, updated_metric_definitions, auc_path
        
        elif reverse_strategy == "immediate":
            print(f"\n[Round {round_idx}] Reverse-metric improvement detected, strategy 'immediate'; "
                  "rerunning experiment for reversed metrics...")
            
            # 保存更新后的指标定义（只包含需要反向处理的指标）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            updated_metrics_json_path = os.path.join(
                OUTPUT_DIR, f"metrics_round_{round_idx}_updated_{timestamp}.json"
            )
            
            # Save only reversed metrics, not normal ones
            updated_reply = json.dumps({"metrics": reversed_metrics_only}, ensure_ascii=False, indent=2)
            
            with open(updated_metrics_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"prompt": "Updated metric definitions (reversed metrics only)", "reply": updated_reply},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            

            # 使用更新后的指标定义重新运行实验（只运行反向指标）
            final_auc_path = run_img_mia_once(
                args,
                metric_json_path=updated_metrics_json_path,
                round_idx=round_idx,
            )
            
            # Ensure reversed metric definitions are complete
            complete_reversed_metrics = ensure_complete_definitions(reversed_metrics_only)
            # Evaluate based on updated results (reversed metrics only)
            final_eval_result = evaluate_metrics(
                agent=agent,
                metrics_text_path=final_auc_path,
                save_dir=OUTPUT_DIR,
                save_prefix=f"metrics_eval_round_{round_idx}_final",
                metric_definitions=complete_reversed_metrics,
                strategy_store_path=IMG_STRATEGY_BANK_PATH,
                persist_strategies=True,
            )
            
            print(f"[Round {round_idx}] Final evaluation result (reversed metrics only):")
            print(final_eval_result)
            return new_metric_names, updated_metric_definitions, final_auc_path
        
        elif reverse_strategy == "defer":
            print(f"[Round {round_idx}] Reverse-metric improvement detected, strategy 'defer'; "
                  "deferring unified processing to the final stage.")
            # Save updated metric definitions (reversed metrics only) without rerunning now
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            updated_metrics_json_path = os.path.join(
                OUTPUT_DIR, f"metrics_round_{round_idx}_updated_{timestamp}.json"
            )
            # Save only reversed metrics, not normal ones
            updated_reply = json.dumps({"metrics": reversed_metrics_only}, ensure_ascii=False, indent=2)
            with open(updated_metrics_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"prompt": "Updated metric definitions (reversed metrics only, deferred)", "reply": updated_reply},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            return new_metric_names, updated_metric_definitions, auc_path
    
    return new_metric_names, metric_definitions, auc_path


def main():
    """Entry point for running the image-based VL-MIA agent pipeline."""
    global model, preprocess

    args = parse_args()

    # ------------------------------------------------------------------
    # 1) Load llama_adapter_v21 model (with optional auto-download)
    # ------------------------------------------------------------------
    logging.info("======= Initializing llama_adapter_v21 =======")

    # If the user does not provide --llama_path, fall back to a cache directory
    # under the user's home. This makes the script portable for open-source use.
    user_provided_llama_dir = getattr(args, "llama_path", None)
    if user_provided_llama_dir:
        llama_dir = user_provided_llama_dir
    else:
        default_llama_root = os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "vl-mia",
            "llama_model_weights",
        )
        os.makedirs(default_llama_root, exist_ok=True)
        llama_dir = default_llama_root
        print(f"[Init] No --llama_path provided. Using default cache directory: {llama_dir}")

    # If the user explicitly set a path and it does not exist, fail fast with a clear message.
    if user_provided_llama_dir and not os.path.exists(llama_dir):
        raise FileNotFoundError(
            f"LLaMA model directory not found: {llama_dir}\n"
            f"Please specify the correct path with --llama_path"
        )

    device = f"cuda:{args.gpu_id}"

    # Choose model type; keep original default for backward compatibility.
    model_type = getattr(args, "model_type", "LORA-BIAS-7B-v21")

    # Use a configurable download root for pre-trained checkpoints.
    # If weights are missing, `llama.load` is expected to download them here.
    download_root = os.getenv(
        "VL_MIA_CKPT_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "vl-mia", "ckpts"),
    )
    os.makedirs(download_root, exist_ok=True)

    print(f"[Init] Using LLaMA model path: {llama_dir}")
    print(f"[Init] Checkpoints will be stored in: {download_root}")

    model, preprocess = llama.load(
        model_type,
        llama_dir,
        llama_type="7B",
        device=device,
        download_root=download_root,
    )
    model.eval()

    print("[Init] Model loaded successfully.")

    # ------------------------------------------------------------------
    # 2) Initialize agent and strategy bank
    # ------------------------------------------------------------------
    agent = BaseAgent()
    n_rounds = max(1, getattr(args, "rounds", 10))

    # Ensure strategy bank exists for image experiments (do not clear if present)
    print(f"[Init] Checking image strategy bank: {IMG_STRATEGY_BANK_PATH}")
    os.makedirs(os.path.dirname(IMG_STRATEGY_BANK_PATH), exist_ok=True)

    if not os.path.exists(IMG_STRATEGY_BANK_PATH):
        print("[Init] Strategy bank not found, creating an empty one.")
        save_strategy_bank([], IMG_STRATEGY_BANK_PATH)
    else:
        print("[Init] Strategy bank already exists; keeping existing entries.")

    print("[Init] Loading historically used metric names...")
    used_metric_names = load_all_used_metric_names(OUTPUT_DIR)
    if used_metric_names:
        print(f"[Init] Found {len(used_metric_names)} historical metrics: {used_metric_names}")
    else:
        print("[Init] No historical metrics found; starting from the first round.")

    # ------------------------------------------------------------------
    # 3) Main exploration loop with (optional) deferred reverse-metric handling
    # ------------------------------------------------------------------
    deferred_reverse_metrics = []  # Store tuples of (round_idx, reversed_metrics, auc_path)
    reverse_strategy = getattr(args, "reverse_strategy", "defer")

    for r in range(1, n_rounds + 1):
        print(f"\n{'=' * 60}")
        print(f"[Round {r}] Starting. Currently used metric count: {len(used_metric_names)}")
        print(f"{'=' * 60}\n")

        new_names, metric_definitions, auc_path = run_one_round(agent, args, r, used_metric_names)
        if new_names:
            print(f"[Round {r}] New metrics added to used list: {new_names}")
            print(f"[Round {r}] Total used metrics after this round: {len(used_metric_names)}")
        
        # For deferred strategy, collect reverse metrics for later unified processing
        if reverse_strategy == "defer" and metric_definitions:
            import glob
            updated_files = glob.glob(
                os.path.join(OUTPUT_DIR, f"metrics_round_{r}_updated_*.json")
            )
            if updated_files:
                # Load reverse metrics from the most recent updated file
                try:
                    latest_file = max(updated_files, key=os.path.getmtime)
                    with open(latest_file, "r", encoding="utf-8") as f:
                        file_data = json.load(f)
                    reply_str = file_data.get("reply", "")
                    reply_str = extract_json_content(reply_str)
                    reversed_metrics = json.loads(reply_str).get("metrics", [])
                    if reversed_metrics:
                        deferred_reverse_metrics.append((r, reversed_metrics, auc_path))
                except Exception as e:
                    print(f"[Warning] Failed to load reverse-metric file for round {r}: {e}")
                    # Fallback to using the full metric_definitions for this round
                    deferred_reverse_metrics.append((r, metric_definitions, auc_path))
    
    # If using deferred reverse-metric strategy, process all at the end
    if reverse_strategy == "defer" and deferred_reverse_metrics:
        print(f"\n{'=' * 60}")
        print(f"[Deferred Reverse] Processing reverse-metric improvements from {len(deferred_reverse_metrics)} rounds")
        print(f"{'=' * 60}\n")
        
        all_deferred_metrics = []
        for round_idx, reversed_metrics, _ in deferred_reverse_metrics:
            all_deferred_metrics.extend(reversed_metrics)
            print(f"[Deferred Reverse] Round {round_idx}: {len(reversed_metrics)} reverse metrics")
        
        if all_deferred_metrics:
            # Save all deferred reverse metrics
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            deferred_metrics_json_path = os.path.join(
                OUTPUT_DIR, f"metrics_deferred_reverse_{timestamp}.json"
            )
            deferred_reply = json.dumps({"metrics": all_deferred_metrics}, ensure_ascii=False, indent=2)
            with open(deferred_metrics_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"prompt": "All deferred reverse metrics", "reply": deferred_reply},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            
            # Run a unified experiment for all reverse metrics
            final_round_idx = n_rounds + 1  # Use a special round index
            final_auc_path = run_img_mia_once(
                args,
                metric_json_path=deferred_metrics_json_path,
                round_idx=final_round_idx,
            )
            
            # Ensure reverse metric definitions are complete
            complete_deferred_metrics = ensure_complete_definitions(all_deferred_metrics)
            # Evaluate all reverse metrics
            final_eval_result = evaluate_metrics(
                agent=agent,
                metrics_text_path=final_auc_path,
                save_dir=OUTPUT_DIR,
                save_prefix="metrics_eval_deferred_reverse",
                metric_definitions=complete_deferred_metrics,
                strategy_store_path=IMG_STRATEGY_BANK_PATH,
                persist_strategies=True,
            )
            
            print(f"[Deferred Reverse] Finished unified processing for {len(all_deferred_metrics)} reverse metrics")
            print(f"[Deferred Reverse] Evaluation result:")
            print(final_eval_result)

    # ------------------------------------------------------------------
    # 4) Clean up model resources
    # ------------------------------------------------------------------
    print("[Cleanup] Releasing model resources...")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()