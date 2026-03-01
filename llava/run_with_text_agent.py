import argparse
import torch
import math
import os
import random
import glob
import logging
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
# from fastchat import model as fmodel  # Not needed for llama_adapter_v21
import llama
import cv2
from PIL import Image
# Global model context for the text-based VL-MIA pipeline
model = None
preprocess = None
##################################
from agent_mia.agent.base_agent import BaseAgent
import requests
from io import BytesIO
import re
import logging
logging.basicConfig(level='ERROR')
from pathlib import Path
from tqdm import tqdm
from eval import *
import pdb
from datasets import load_dataset
import sys
sys.path.insert(0,'../')
from metric_util import get_text_metric, get_img_metric, save_output, get_meta_metrics, get_text_metric_agent
import os
from agent_mia.agent.base_agent import (
    BaseAgent,
    build_strategy_reference,
    parse_metric_definitions,
    evaluate_metrics,
    save_strategy_bank,
 )
 # Custom prompt module
from agent_mia.agent.prompt import prompt_new,judge_prompt_template  # Metric generation prompt template
import json
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "agent_round_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define separate strategy bank path for text experiments
TEXT_STRATEGY_BANK_PATH = os.path.join(CURRENT_DIR, "agent_mia", "agent", "strategy_bank_text.json")













def shuffle_sentence(sentence):
    words = sentence.split()
    random.shuffle(words)    
    shuffled_sentence = ' '.join(words)
    return shuffled_sentence

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--llama_path",
                        default=None,
                        help=(
                            "Path to LLaMA model weights directory. "
                            "If not provided, a cache directory under ~/.cache/vl-mia/llama_model_weights "
                            "will be used."
                        ))
    parser.add_argument("--gpu_id",
                        type=int,
                        default=0,
                        help="specify the gpu to load the model.")
    parser.add_argument('--output_dir', type=str, default="outputs")
    parser.add_argument('--dataset', type=str, default="llava_v15_gpt_text")
    parser.add_argument("--text_len", type=int, default=32)
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="path to local dataset directory. If specified, will load from this path instead of default VL-MIA-text_{text_len}. If not specified, uses CURRENT_DIR/VL-MIA-text_{text_len}.")
    parser.add_argument("--model_type", type=str, default="LORA-BIAS-7B-v21",
                        help="Model type: BIAS-7B, LORA-BIAS-7B, LORA-BIAS-7B-v21")
    parser.add_argument("--metric_json", type=str, default=None)
    parser.add_argument(
        "--n_metrics",
        type=int,
        default=5,
        help="Number of new metrics to ask the agent to generate per round (default: 10).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of exploration-evaluation rounds to run (default: 10).",
    )
    args = parser.parse_args()
    return args


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
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


def evaluate_data(model, test_data, col_name, gpu_id):
    print(f"all data size: {len(test_data)}")
    all_output = []
    test_data = test_data
    for ex in tqdm(test_data): 
        text = ex[col_name]
        new_ex = inference(model, text, ex, gpu_id)
        all_output.append(new_ex)
    return all_output



########################################################

import os
from typing import Optional, List, Dict

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
        guidance_text += "\n[Key Observations]\n"
        for k, v in ui.items():
            guidance_text += f"- {k}: {v}\n"

    if nrs:
        guidance_text += "\n[Next Round Strategy Suggestions]\n"
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
            # Ignore files that fail to parse
            continue
    
    return sorted(used_names)  # Sort for easier display


def get_completed_rounds(output_dir: str) -> set:
    """
    Check which rounds have been completed.

    Completed rounds are inferred by scanning metrics_eval_round_*.json files.
    If there exists a metrics_eval_round_{round}_final*.json file, that round
    is considered fully completed (including reverse-metric processing).
    Otherwise, if there exists a metrics_eval_round_{round}_*.json file, the
    round is also considered completed.

    Returns:
        A set of completed round indices.
    """
    import os, re
    
    if not os.path.exists(output_dir):
        return set()
    
    completed_rounds = set()
    
    files = [
        f for f in os.listdir(output_dir)
        if f.startswith("metrics_eval_round_") and f.endswith(".json")
    ]
    
    for filename in files:
        # Filename format examples:
        #   metrics_eval_round_1_20260115_150801.json
        #   metrics_eval_round_1_final_20260115_151201.json
        # Only need to extract the round number from the prefix
        match = re.match(r"metrics_eval_round_(\d+)", filename)
        if match:
            round_num = int(match.group(1))
            completed_rounds.add(round_num)
    
    return completed_rounds



############################################



def load_conversation_template(template_name):
    conv_template = fmodel.get_conversation_template(template_name)
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()
    return conv_template

def inference(model, text, ex, gpu_id):
    pred = {}
    
    metrics = mod_infer(model, text, gpu_id)
    metrics_lower = mod_infer(model, text.lower(), gpu_id)

    ppl = metrics["ppl"]
    all_prob = metrics["all_prob"]
    p1_likelihood = metrics["loss"]
    entropies = metrics["entropies"]
    mod_entropy = metrics["modified_entropies"]
    max_p = metrics["max_prob"]
    org_prob = metrics["probabilities"]
    gap_p = metrics["gap_prob"]
    renyi_05 = metrics["renyi_05"]
    renyi_2 = metrics["renyi_2"]
    mod_renyi_05 = metrics["mod_renyi_05"]
    mod_renyi_2 = metrics["mod_renyi_2"]
    #########
    loss_var = metrics["loss_var"]              # loss variance
    ppl_lower = metrics_lower["ppl"]
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
###################################################
    custom = {k: v for k, v in metrics.items() if k not in base_keys}
    pred = get_text_metric_agent(custom)##########################
    # custom = {k: v for k, v in metrics.items() if k not in base_keys}
    ex["pred"] = pred
    return ex





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

def mod_infer(model, text, gpu_id):
    device='cuda:{}'.format(gpu_id)

    img = Image.new('RGB', (1024, 1024), color = 'black')
    img = preprocess(img).unsqueeze(0).to(device)

    with torch.cuda.amp.autocast():
        visual_query = model.forward_visual(img)

    prompt = llama.format_prompt("") + text

    prompt_t = model.tokenizer.encode(prompt, bos=True, eos=False)

    tokens = torch.tensor(prompt_t).to(device).long().unsqueeze(0) 

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            logits = logits_forward(model, tokens, visual_query)

    input_ids = tokens[0]

    descp_encoding = model.tokenizer.encode(text, bos=False, eos=False)

    target_slice = slice(-len(descp_encoding)-10,None)

    logits_slice = logits[0,target_slice,:]

    input_ids = input_ids[target_slice]

    probabilities = torch.nn.functional.softmax(logits_slice, dim=-1)
    log_probabilities = torch.nn.functional.log_softmax(logits_slice, dim=-1)
    
    return get_meta_metrics(input_ids, probabilities, log_probabilities)




##################
from metrics_plugin import register_metric, clear_metrics_registry
import json

def extract_json_content(text: str) -> str:
        """
        Extract pure JSON text from LLM response that may be wrapped in ```json ... ``` block.
        Return as-is if no curly braces found.
        """
        if not isinstance(text, str):
            return text

        # First remove wrapper ```json ``` or ``` ```
        fence_pattern = r"```(?:json)?(.*?)```"
        match = re.search(fence_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()

        # Then extract the first JSON object from the text
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            return brace_match.group(0)
        return text
def load_agent_metrics(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    reply_str = data["reply"]

    # Force processing with extract_json_content
    reply_str = extract_json_content(reply_str)

    # Then parse with json.loads
    reply_json = json.loads(reply_str)

    for m in reply_json["metrics"]:
        register_metric(m["name"], m["code"])

########################
# ========================================
#             Model Initialization
# ========================================

#####################################################################


# run_text_with_agent.py




###############################################

def parse_auc_file(auc_file_path: str) -> List[Dict]:
    """
    Parse auc.txt file and extract metric name and accuracy for each metric.
    
    Returns:
        List of metric info, each element contains: name, accuracy, auc, tpr_at_5_fpr, full_name
    """
    metrics_info = []
    
    if not os.path.exists(auc_file_path):
        print(f"[Warning] auc.txt file not found: {auc_file_path}")
        return metrics_info
    
    with open(auc_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse format: [method] metric_name   AUC 0.xxxx, Accuracy 0.xxxx, TPR@5%FPR of 0.xxxx
            # Or: metric_name   AUC 0.xxxx, Accuracy 0.xxxx, TPR@5%FPR of 0.xxxx
            # Note: There may be spaces in TPR@5% FPR
            match = re.match(r'(\[[^\]]+\]\s+)?(.+?)\s+AUC\s+([\d.]+),\s+Accuracy\s+([\d.]+),\s+TPR@5%[^0-9]*([\d.]+)', line)
            if match:
                method_prefix = match.group(1)  # May be None
                metric_name_full = match.group(2).strip()
                auc = float(match.group(3))
                accuracy = float(match.group(4))
                tpr_at_5_fpr = float(match.group(5))
                
                # Extract clean metric name (remove method prefix)
                metric_name = metric_name_full
                method = None
                if method_prefix:
                    method = method_prefix.strip('[] ').strip()
                    # Remove method prefix from metric name if present
                    if metric_name.startswith(f"[{method}]"):
                        metric_name = metric_name[len(f"[{method}]"):].strip()
                
                metrics_info.append({
                    "name": metric_name,  # Clean metric name
                    "full_name": metric_name_full,  # Full name (may include method)
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
        agent: BaseAgent instance
        metric_name: Original metric name
        original_code: Original metric code
        accuracy: Accuracy of the original metric
        auc: AUC of the original metric
    
    Returns:
        If successful, returns dict containing reversed metric definition; otherwise None
    """
    # Build judge prompt
    judge_prompt = (
        judge_prompt_template
        + "\n\n==== Metric to Evaluate ====\n"
        + f"Metric Name: {metric_name}\n"
        + f"Current Accuracy: {accuracy:.4f}\n"
        + f"Current AUC: {auc:.4f}\n"
        + f"\nOriginal Code:\n```python\n{original_code}\n```\n"
        + "\nPlease determine if reversal is needed. If Accuracy <= 0.52, generate the reversed version of the code."
    )
    
    try:
        reply = agent.ask(judge_prompt)
        cleaned_reply = extract_json_content(reply)
        reply_json = json.loads(cleaned_reply)
        
        metrics = reply_json.get("metrics", [])
        if metrics and len(metrics) > 0:
            reversed_metric = metrics[0]
            # Ensure reversed metric name contains original name and is marked as reversed
            reversed_name = f"{metric_name}_reversed"
            reversed_metric["name"] = reversed_name
            reversed_metric["original_name"] = metric_name
            return reversed_metric
        else:
            print(f"[Warning] Failed to parse reversed metric from LLM response: {metric_name}")
            return None
    except Exception as e:
        print(f"[Error] Failed to generate reversed metric ({metric_name}): {e}")
        return None


def ensure_complete_definition(metric_def: dict) -> dict:
    """
    Ensure metric definition contains all required fields, fill in missing fields.
    
    Args:
        metric_def: Metric definition dict
    
    Returns:
        Completed metric definition dict
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


def ensure_complete_definitions(metric_definitions: List[Dict]) -> List[Dict]:
    """
    Batch ensure all metric definitions contain all required fields.
    
    Args:
        metric_definitions: List of metric definitions
    
    Returns:
        List of completed metric definitions
    """
    if not isinstance(metric_definitions, list):
        return metric_definitions
    
    return [ensure_complete_definition(md) for md in metric_definitions]


def try_reverse_weak_metrics(agent: BaseAgent, 
                              args, 
                              auc_path: str, 
                              metric_definitions: List[Dict],
                              round_idx: int):
    """
    Simplified reversed metric processing:
    1. Find metrics with accuracy <= 0.501
    2. Generate reversed version (name unchanged)
    3. Overwrite original metric's code
    4. No secondary experiments, no reversed vs original comparison
    """

    print(f"\n[Round {round_idx}] Simplified reversed metric check...")

    # Parse auc.txt
    metrics_info = parse_auc_file(auc_path)
    if not metrics_info:
        print("[Info] auc.txt has no valid metrics")
        return metric_definitions, False

    # Aggregate by metric name and take worst accuracy
    metrics_by_name = {}
    for m in metrics_info:
        name = m["name"]
        if name not in metrics_by_name or m["accuracy"] < metrics_by_name[name]["accuracy"]:
            metrics_by_name[name] = m

    # Find weak metrics
    weak_metrics = [m for m in metrics_by_name.values() if m["accuracy"] <= 0.501 or m.get("auc", 0.5) < 0.3]
    if not weak_metrics:
        print("[Info] No metrics need reversal")
        return metric_definitions, False

    print(f"[Info] Need to process {len(weak_metrics)} metrics for reversal")

    updated = False

    for m in weak_metrics:
        name = m["name"]
        print(f"  - Processing weak metric: {name}")

        # Find corresponding definition
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
            print(f"[Warning] Metric has no code field, skipping")
            continue

        # Call LLM to generate reversed metric (name unchanged, only change code)
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

            # **Ensure original definition is complete, then only update code, keep all other fields (name, formula, description, expected_behavior, etc.)**
            original = ensure_complete_definition(original)
            # Only update code field
            original["code"] = reversed_metric.get("code", original.get("code", ""))
            updated = True

            print(f"    ✓ Overwritten original metric: {name}")

        except Exception as e:
            print(f"[Error] Reversed metric generation failed {name}: {e}")

    # Ensure all metric_definitions are complete
    metric_definitions = ensure_complete_definitions(metric_definitions)
    return metric_definitions, updated
############################################









def run_text_mia_once(args, metric_json_path: Optional[str] = None, round_idx: Optional[int] = None) -> str:
    """
    Run one round of text MIA experiment:
    - Optional: Load agent-generated metric plugins from metric_json_path for this round
    - Use currently registered metrics (built-in + plugins) to perform MIA on dataset
    - Write auc.txt and return its path
    """

    # 1. Do not parse_args() here, args are passed from outside

    # 2. Ensure global model is initialized
    global model, preprocess
    if model is None:
        raise RuntimeError("Model not yet initialized. Please initialize the model in main() first.")

    # 2.5. Clear metrics from previous round to prevent accumulation and increasing computation time
    if round_idx is not None:
        print(f"[Round {round_idx}] Clearing metrics registry from previous round...")
        clear_metrics_registry()

    # 3. Load metric plugins for this round (if metrics.json provided)
    #    Prefer function param metric_json_path, then try args.metric_json
    if metric_json_path is None:
        metric_json_path = getattr(args, "metric_json", None)

    if metric_json_path is not None:
        print(f"[MIA] Loading agent metrics from: {metric_json_path}")
        load_agent_metrics(metric_json_path)
    else:
        print("[MIA] No metric_json provided, using only built-in metrics.")
        
    # Conversation template already initialized in main()

    # 5. Construct output directory (optionally separate by round)
    text_len = args.text_len

    base_output_dir = args.output_dir
    # If you want to separate results by round:
    if round_idx is not None:
        output_dir = os.path.join(base_output_dir, f"round_{round_idx}", f"length_{text_len}")
    else:
        output_dir = os.path.join(base_output_dir, f"length_{text_len}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 6. Load dataset (from local path)
    from datasets import load_from_disk
    # If dataset_path is specified, use it; otherwise use default path
    if hasattr(args, 'dataset_path') and args.dataset_path:
        dataset_path = args.dataset_path
    else:
        dataset_path = os.path.join(CURRENT_DIR, f"VL-MIA-text_{text_len}")
    print(f"[MIA] Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    data = convert_huggingface_data_to_list_dic(dataset)

    logging.info("=======Initialization Finished=======")

    # 7. Run inference + compute all metrics (including plugin metrics)
    all_output = evaluate_data(
        model,
        data,
        "input",
        args.gpu_id,
    )

    # 8. Plot ROC and write auc.txt (fig_fpr_tpr should write auc.txt internally)
    fig_fpr_tpr(all_output, output_dir)

    # Assume auc.txt is under output_dir:
    auc_path = os.path.join(output_dir, "auc.txt")
    print(f"[MIA] AUC file written to: {auc_path}")
    return auc_path




def run_one_round(agent: BaseAgent, args, round_idx: int, used_metric_names: list = None):
    """
    Run one round of metric generation and evaluation.
    
    Args:
        agent: BaseAgent instance
        args: Arguments object
        round_idx: Current round index
        used_metric_names: List of used metric names, will be updated after each round
    """
    if used_metric_names is None:
        used_metric_names = []
    
    # 1. Build strategy reference text (select n good and n bad strategies from strategy bank)
    n_good = 3  # Number of good strategies
    n_bad = 3   # Number of bad strategies
    strategy_reference = build_strategy_reference(
        n_good=n_good,
        n_bad=n_bad,
        path=TEXT_STRATEGY_BANK_PATH,
    )
    
    eval_guidance = load_latest_eval_guidance(OUTPUT_DIR) 
    
    # 2. Build text block of used metrics list
    used_metrics_block = ""
    if used_metric_names:
        used_metrics_text = "\n".join([f"- {name}" for name in used_metric_names])
        used_metrics_block = (
            "\n\n==============================\n"
            + "[Previously Used Metrics List (Please avoid generating duplicates)]\n"
            + f"The following metrics have been generated in previous rounds of this experiment. Please do not generate identical or highly similar metrics:\n\n"
            + used_metrics_text
            + "\n\nPlease ensure that the metric names and computation logic you generate are significantly different from the above metrics."
        )
    
    guidance_block = ""
    if eval_guidance:
        guidance_block = (
            "\n\n==============================\n"
            + eval_guidance
        )
    # 2.1 Determine number of metrics to generate this round from args (default 10)
    n_metrics = getattr(args, "n_metrics", 5)
    
    # Use args.n_metrics if available, otherwise use default
    if hasattr(args, "n_metrics"):
        n_metrics = args.n_metrics

    # Replace {{N_METRICS}} placeholder in prompt with actual number
    base_generation_prompt = prompt_new.replace("{{N_METRICS}}", str(n_metrics))

    # 3. Generate new metrics (call LLM)
    if eval_guidance:
        generation_prompt = (
            base_generation_prompt
            + used_metrics_block  # Used metrics list (if any)
            + guidance_block  # Previous round evaluation conclusions (if any)
            + "\n\n==============================\n[Strategy Bank Reference (Contains successful and failed strategies to help understand what works and what doesn't)]\n"
            + strategy_reference
        )
    else:
        generation_prompt = (
            base_generation_prompt
            + used_metrics_block  # Used metrics list (if any)
            + "\n\n==============================\n[Strategy Bank Reference (Contains successful and failed strategies to help understand what works and what doesn't)]\n"
            + strategy_reference
        )
    reply = agent.ask(generation_prompt)

    # 3. Save this round's metrics.json (just the "prompt+reply" shell)
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

    # 4. Parse metric_definitions (for later evaluation, save to strategy bank)
    metric_definitions = parse_metric_definitions(reply)
    if not metric_definitions:
        print("[Warning] Failed to parse any metric definitions this round, skipping.")
        return []  # Return empty list to indicate no new metrics
    
    # Extract newly generated metric names this round
    new_metric_names = []
    for metric in metric_definitions:
        name = metric.get("name")
        if name and name not in used_metric_names:
            new_metric_names.append(name)
            used_metric_names.append(name)  # Update used list
    
    print(f"[Round {round_idx}] New metrics generated this round: {new_metric_names}")

    # 5. Run one round of MIA experiment (using run_text_mia_once)
    auc_path = run_text_mia_once(
        args,
        metric_json_path=metrics_json_path,  # This round's metrics.json
        round_idx=round_idx,                 # For distinguishing output directory
    )

    # Check and try to reverse process metrics with accuracy <= 0.502
    updated_metric_definitions, was_updated = try_reverse_weak_metrics(
        agent=agent,
        args=args,
        auc_path=auc_path,
        metric_definitions=metric_definitions,
        round_idx=round_idx,
    )
    
    # Evaluate metrics and save to strategy bank (using updated metric definitions if reversal occurred)
    # Use updated_metric_definitions (already ensured complete in try_reverse_weak_metrics) to save updated metric code
    # Ensure all metric definitions are complete (including name, formula, description, code, expected_behavior, etc.)
    final_metric_definitions = ensure_complete_definitions(updated_metric_definitions)
    
    eval_result = evaluate_metrics(
        agent=agent,
        metrics_text_path=auc_path,
        save_dir=OUTPUT_DIR,
        save_prefix=f"metrics_eval_round_{round_idx}",
        metric_definitions=final_metric_definitions,
        strategy_store_path=TEXT_STRATEGY_BANK_PATH,
        persist_strategies=True,
    )
    
    print(f"[Round {round_idx}] Evaluation structured results:")
    print(eval_result)
    
    if was_updated:
        print(f"\n[Round {round_idx}] Reversed metric improvement detected, re-running experiment with updated metric definitions...")
        
        # Save updated metric definitions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        updated_metrics_json_path = os.path.join(
            OUTPUT_DIR, f"metrics_round_{round_idx}_updated_{timestamp}.json"
        )
        
        # Build updated JSON reply format
        updated_reply = json.dumps({"metrics": updated_metric_definitions}, ensure_ascii=False, indent=2)
        
        with open(updated_metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {"prompt": "Updated metric definitions (including reversed metrics)", "reply": updated_reply},
                f,
                ensure_ascii=False,
            indent=2,
        )
        
        
        # Re-run experiment with updated metric definitions
        final_auc_path = run_text_mia_once(
            args,
            metric_json_path=updated_metrics_json_path,
            round_idx=round_idx,
        )
        
        # Ensure updated metric definitions are complete
        updated_metric_definitions = ensure_complete_definitions(updated_metric_definitions)
        # Re-evaluate with updated results
        final_eval_result = evaluate_metrics(
            agent=agent,
            metrics_text_path=final_auc_path,
            save_dir=OUTPUT_DIR,
            save_prefix=f"metrics_eval_round_{round_idx}_final",
            metric_definitions=updated_metric_definitions,
            strategy_store_path=TEXT_STRATEGY_BANK_PATH,
            persist_strategies=True,
        )
        
        print(f"[Round {round_idx}] Final evaluation results (including reversed metric improvement):")
        print(final_eval_result)

    # Return list of newly generated metric names for caller to update global list
    return new_metric_names





##############################################################################################


def main():
    """Entry point for running the text-based VL-MIA agent pipeline."""
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
    
    # Check which rounds have already been completed (for resumable runs)
    completed_rounds = get_completed_rounds(OUTPUT_DIR)
    if completed_rounds:
        print(f"[Init] Detected completed rounds: {sorted(completed_rounds)}")
        start_round = max(completed_rounds) + 1
        print(f"[Init] Resuming from round {start_round}")
    else:
        start_round = 1
        print("[Init] No completed rounds found; starting from round 1.")
    
    # If a strategy bank already exists and contains data, keep it instead of clearing.
    if os.path.exists(TEXT_STRATEGY_BANK_PATH):
        try:
            with open(TEXT_STRATEGY_BANK_PATH, "r", encoding="utf-8") as f:
                existing_strategies = json.load(f)
            if existing_strategies:
                print(f"[Init] Existing strategy bank detected; keeping historical strategies.")
            else:
                print("[Init] Strategy bank is empty; initializing a new one.")
                save_strategy_bank([], TEXT_STRATEGY_BANK_PATH)
        except Exception as e:
            print(f"[Init] Failed to read strategy bank; reinitializing: {e}")
            os.makedirs(os.path.dirname(TEXT_STRATEGY_BANK_PATH), exist_ok=True)
            save_strategy_bank([], TEXT_STRATEGY_BANK_PATH)
    else:
        print(f"[Init] Strategy bank not found; creating a new one at: {TEXT_STRATEGY_BANK_PATH}")
        os.makedirs(os.path.dirname(TEXT_STRATEGY_BANK_PATH), exist_ok=True)
        save_strategy_bank([], TEXT_STRATEGY_BANK_PATH)
    
    # Load all historically used metric names (from previous rounds)
    print("[Init] Loading historically used metric names...")
    used_metric_names = load_all_used_metric_names(OUTPUT_DIR)
    if used_metric_names:
        print(f"[Init] Found {len(used_metric_names)} historical metrics: {used_metric_names}")
    else:
        print("[Init] No historical metrics found; starting a fresh record from round 1.")
    
    # ------------------------------------------------------------------
    # 3) Run multiple rounds of exploration and evaluation
    # ------------------------------------------------------------------
    for r in range(start_round, n_rounds + 1):
        print(f"\n{'=' * 60}")
        print(f"[Round {r}] Starting. Currently used metric count: {len(used_metric_names)}")
        print(f"{'=' * 60}\n")
        
        # Run one round, passing the used-metric list
        new_names = run_one_round(agent, args, r, used_metric_names)
        
        # Synchronize used-metric list (run_one_round already updates it, this is just for logging)
        if new_names:
            print(f"[Round {r}] New metrics added to used list.")
            print(f"[Round {r}] Total used metrics after this round: {len(used_metric_names)}")
    
    # ------------------------------------------------------------------
    # 4) Clean up resources
    # ------------------------------------------------------------------
    print("[Cleanup] Releasing model resources...")
    del model, preprocess
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()


