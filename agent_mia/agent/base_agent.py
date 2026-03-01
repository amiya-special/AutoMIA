"""Core agent and metric-evaluation utilities for the VL-MIA project.

This module provides:
- A `BaseAgent` wrapper around an OpenAI-compatible chat completion API.
- Utilities for saving prompts/replies, parsing JSON from model output.
- A small framework for designing, evaluating, and persisting metric strategies.

Note:
    The path fix below (modifying ``sys.path``) must remain near the top of the file
    so that the module can be executed both as a script and as a package module.
"""

# =================== PATH FIX (must stay at the top) ===================
from __future__ import annotations  # Enable newer type annotation syntax for Python 3.8+
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))            # agent_mia/agent
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))   # VL-MIA-main
sys.path.insert(0, ROOT_DIR)
# =============================================================
# Correctly import variables from the prompt module
from agent_mia.agent.prompt import (
    base_prompt_template,
    feature_list,
    prompt_new,
    evaluator_prompt_template,
)
import logging
import json
import re
import numpy as np
from openai import OpenAI
from agent_mia.agent.config_loader import Config     # Absolute import
from datetime import datetime
from typing import Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "outputs")
STRATEGY_BANK_PATH = os.path.join(CURRENT_DIR, "strategy_bank.json")


class BaseAgent:
    ####################################
    def __init__(self, config_path="/root/autodl-tmp/VL-MIA-main/agent_mia/config.yaml"):  
        if config_path is None:
            config_path = os.path.join(ROOT_DIR, "agent_mia", "config.yaml")  

        # Load config
        self.config = Config(config_path)

        logging.basicConfig(level="INFO")

        api_type = self.config.get("api.type", "deepseek").lower()
        api_key = self.config.get("api.api_key")
        base_url = self.config.get("api.base_url")
        
        # Automatically set base_url based on api_type (if not specified)
        if not base_url:
            if api_type == "openrouter":
                base_url = "https://openrouter.ai/api/v1"
            elif api_type == "deepseek":
                base_url = "https://api.deepseek.com"
            elif api_type == "openai":
                base_url = "https://api.openai.com/v1"
            else:
                # Fallback: use base_url from config; if still empty, default to DeepSeek
                base_url = base_url or "https://api.deepseek.com"

        # Initialize OpenAI-compatible client
        client_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
        }
        
        # OpenRouter requires additional default headers
        if api_type == "openrouter":
            default_headers = {
                "HTTP-Referer": self.config.get("api.http_referer", "https://github.com/VL-MIA"),
                "X-Title": self.config.get("api.x_title", "VL-MIA Agent"),
            }
            client_kwargs["default_headers"] = default_headers
        
        self.client = OpenAI(**client_kwargs)
        self.api_type = api_type

        self.model_name = self.config.get("model.name")
        self.temperature = self.config.get("model.temperature")

        print(f"[Agent] Ready. API = {api_type}, Model = {self.model_name}, Base URL = {base_url}")

    def ask(self, prompt, max_retries=3, timeout=120.0):
        """
        Send a request to the API with a retry mechanism.

        Args:
            prompt: Input prompt text.
            max_retries: Maximum number of retries (default: 3).
            timeout: Request timeout in seconds (default: 120 seconds).
        """
        import time
        
        for attempt in range(max_retries):
            try:
                print(f"[Agent] Trying request {attempt + 1}/{max_retries} ...")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=32000,
                    timeout=timeout,  # Extended timeout
                )
                content = response.choices[0].message.content
                if content is None or content.strip() == "":
                    logging.warning(f"[Agent] API returned empty content; model may have failed or timed out")
                    print(f"[Warning] API returned empty content. Please check API configuration and network connectivity.")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5  # Increasing wait time: 5s, 10s, 15s
                        print(f"[Agent] Waiting {wait_time} seconds before retrying...")
                        time.sleep(wait_time)
                        continue
                    return ""
                print(f"[Agent] Request succeeded, response length: {len(content)} characters")
                return content
            except Exception as e:
                error_msg = str(e)
                is_timeout = "timeout" in error_msg.lower() or "timed out" in error_msg.lower()
                
                logging.error(f"[Agent] API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"[Error] Agent API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                # If this is a timeout error and we still have retries left, retry
                if is_timeout and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10  # Longer wait after timeouts: 10s, 20s, 30s
                    print(f"[Agent] Timeout detected, waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    continue
                elif attempt < max_retries - 1:
                    # Retry for other errors as well, with a shorter wait
                    wait_time = (attempt + 1) * 3
                    print(f"[Agent] Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Final attempt also failed
                    import traceback
                    traceback.print_exc()
                    print(f"[Error] All {max_retries} attempts failed, returning empty string")
                    return ""
        
        return ""  # All retries failed
    
    

def save_as_json(prompt, reply, file_path):
        """Save to JSON with proper formatting."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                {"prompt": prompt, "reply": reply},
                f, ensure_ascii=False, indent=2,
            )

def save_as_markdown(prompt, reply, file_path):
        """Save to Markdown for easy reading."""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# 🧠 Agent Reply\n\n")
            f.write("## 📝 Prompt\n")
            f.write("```\n" + prompt + "\n```\n\n")
            f.write("## 💬 Model Reply\n")
            f.write(reply.strip() + "\n")


def extract_json_content(text: str) -> str:
        """
        Extract raw JSON text from a possible ```json ... ``` block returned by an LLM.
        If no curly-brace JSON object is found, return the text as-is.
        """
        if not isinstance(text, str):
            return text

        # First strip surrounding ```json ``` or ``` ``` fences
        fence_pattern = r"```(?:json)?(.*?)```"
        match = re.search(fence_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()

        # Then extract the first JSON object from the text
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            return brace_match.group(0)
        return text


def load_strategy_bank(path: str = STRATEGY_BANK_PATH) -> list:
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            return data.get("strategies", [])
        except Exception:
            return []


###################### Dynamic metric design strategy #########################
def compute_metric_score(item,
                         w_auc: float = 0.6,
                         w_acc: float = 0.3,
                         w_tpr: float = 0.1) -> float:
    """
    Compute an overall score for a single ranking item.

    Args:
        item: A dict containing performance metrics; must include auc, accuracy, tpr_at_5_fpr.
        w_auc: Weight for AUC (default 0.5).
        w_acc: Weight for accuracy (default 0.3).
        w_tpr: Weight for TPR@5%FPR (default 0.2).

    Returns:
        Overall score in the range [0.0, 1.0].
    """
    auc = item.get("auc") or 0.0
    acc = item.get("accuracy") or 0.0
    tpr = item.get("tpr_at_5_fpr") or 0.0
    # Ensure the weights are normalized
    total_weight = w_auc + w_acc + w_tpr
    if total_weight > 0:
        w_auc, w_acc, w_tpr = w_auc / total_weight, w_acc / total_weight, w_tpr / total_weight
    return w_auc * auc + w_acc * acc + w_tpr * tpr

def assign_dynamic_categories(ranking: list[dict],
                              q_low: float = 0.3,
                              q_high: float = 0.7,
                              w_auc: float = 0.6,
                              w_acc: float = 0.3,
                              w_tpr: float = 0.1) -> list[dict]:
    """
    Compute scores for all metrics in this round, then dynamically assign categories strong / mid / weak.
    This is a dynamic evaluation scheme: thresholds are adapted based on the current round's score distribution.

    Args:
        ranking: List of metric ranking items, each containing performance metrics.
        q_low: Lower quantile threshold for weak (default 0.3, i.e., bottom 30%).
        q_high: Upper quantile threshold for strong (default 0.7, i.e., top 30%).
        w_auc: Weight for AUC (default 0.6).
        w_acc: Weight for accuracy (default 0.3).
        w_tpr: Weight for TPR@5%FPR (default 0.1).

    Returns:
        The ranking list with updated category and score fields.

    Category rules:
        - score <= q_low quantile: weak (metrics with relatively poor performance)
        - score >= q_high quantile: strong (metrics with excellent performance)
        - otherwise: mid (metrics with medium performance)
    """
    if not ranking:
        return ranking

    # Compute the overall score for each metric
    scores = [compute_metric_score(item, w_auc=w_auc, w_acc=w_acc, w_tpr=w_tpr) for item in ranking]
    
    # Dynamically compute thresholds based on the score distribution in this round
    if len(scores) > 1:
        low_th = float(np.quantile(scores, q_low))
        high_th = float(np.quantile(scores, q_high))
    else:
        # If there is only one metric, mark it as mid directly
        low_th = scores[0] - 0.1
        high_th = scores[0] + 0.1

    # Assign category for each metric
    for item, s in zip(ranking, scores):
        item["score"] = s  # Write score back into ranking, for later storage and use
        if s <= low_th:
            cat = "weak"
        elif s >= high_th:
            cat = "strong"
        else:
            cat = "mid"
        # Override or complement the LLM-provided category with dynamically computed category
        item["category"] = cat
        item["dynamic_score"] = s  # Store dynamic score
    
    return ranking



############################################################


def save_strategy_bank(strategies: list, path: str = STRATEGY_BANK_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"strategies": strategies}, f, ensure_ascii=False, indent=2)


def append_to_strategy_bank(new_entries: list, path: str = STRATEGY_BANK_PATH):
        if not new_entries:
            return
        strategies = load_strategy_bank(path)
        existing_index = {item.get("metric_name"): idx for idx, item in enumerate(strategies)}
        for entry in new_entries:
            name = entry.get("metric_name")
            if name in existing_index:
                strategies[existing_index[name]] = entry
            else:
                strategies.append(entry)
        save_strategy_bank(strategies, path)


def build_strategy_reference(max_items: int = 5, 
                              n_good: int = 3,
                              n_bad: int = 3,
                              path: str = STRATEGY_BANK_PATH) -> str:
        """
        Build a textual strategy reference from the strategy bank, supporting selection of n good and n bad strategies.

        Args:
            max_items: Maximum number of strategies to select (backward compatibility; ignored if n_good and n_bad are set).
            n_good: Number of good strategies (category strong) to include.
            n_bad: Number of bad strategies (category weak) to include.
            path: Path to the strategy bank file.

        Returns:
            A strategy reference text that presents both good and bad strategies.
        """
        strategies = load_strategy_bank(path)
        if not strategies:
            return "No historical strategies available. You are free to explore new metrics."
        
        # Group and sort strategies by category and score
        def get_score(item):
            """Compute a score for a strategy, preferring dynamic_score, otherwise falling back to AUC."""
            score = item.get("dynamic_score")
            if score is not None:
                return float(score)
            perf = item.get("performance", {})
            return perf.get("auc", 0.0)
        
        # Group by category
        strong_strategies = []
        mid_strategies = []
        weak_strategies = []
        
        for strat in strategies:
            category = strat.get("category", "mid")
            if category == "strong":
                strong_strategies.append(strat)
            elif category == "weak":
                weak_strategies.append(strat)
            else:
                mid_strategies.append(strat)
        
        # Sort within each group by score
        strong_strategies = sorted(strong_strategies, key=get_score, reverse=True)
        weak_strategies = sorted(weak_strategies, key=get_score, reverse=False)  # Bad ones: score from low to high
        mid_strategies = sorted(mid_strategies, key=get_score, reverse=True)
        
        # Decide how many strategies to select
        if n_good is not None and n_bad is not None:
            # New behavior: select n_good good strategies and n_bad bad strategies
            selected_good = strong_strategies[:n_good] if n_good > 0 else []
            selected_bad = weak_strategies[:n_bad] if n_bad > 0 else []
        else:
            # Backward compatibility: only select the top max_items strategies
            all_strategies = sorted(strategies, key=get_score, reverse=True)
            selected_good = all_strategies[:max_items]
            selected_bad = []
        
        lines = []
        
        # Show good strategies (category strong)
        if selected_good:
            lines.append("[Successful Strategy Reference (Metrics with good performance, can serve as design direction references)]")
            for strat in selected_good:
                perf = strat.get("performance", {})
                metric_name = strat.get('metric_name', 'Unknown Metric')
                category = strat.get("category", "unknown")
                
                # Extract code snippet from the metric definition
                definition = strat.get("definition", {})
                code = definition.get("code", "")
                
                # Build base information line
                base_info = (
                    f"- {metric_name} [Category: {category}]: "
                    f"AUC {perf.get('auc', 'N/A')}, "
                    f"Accuracy {perf.get('accuracy', 'N/A')}, "
                    f"TPR@5%FPR {perf.get('tpr_at_5_fpr', 'N/A')}. "
                    f"Analysis: {strat.get('analysis', 'None')}"
                )
                lines.append(base_info)
                
                # If code exists, append it to the output
                if code:
                    lines.append(f"  Code Implementation:\n```python\n{code}\n```")
                
                # Add separators between strategies
                if strat != selected_good[-1]:
                    lines.append("")
            
            lines.append("")  # Blank line between good and bad strategies

        # Show bad strategies (category weak)
        if selected_bad:
            lines.append("[Failed Strategy Reference (Metrics with poor performance, should avoid similar designs or analyze failure reasons)]")
            for strat in selected_bad:
                perf = strat.get("performance", {})
                metric_name = strat.get('metric_name', 'Unknown Metric')
                category = strat.get("category", "unknown")
                
                # Extract code snippet from the metric definition
                definition = strat.get("definition", {})
                code = definition.get("code", "")
                
                # Build base information line
                base_info = (
                    f"- {metric_name} [Category: {category}]: "
                    f"AUC {perf.get('auc', 'N/A')}, "
                    f"Accuracy {perf.get('accuracy', 'N/A')}, "
                    f"TPR@5%FPR {perf.get('tpr_at_5_fpr', 'N/A')}. "
                    f"Analysis: {strat.get('analysis', 'None')}"
                )
                lines.append(base_info)
                
                # If code exists, append it to the output
                if code:
                    lines.append(f"  Code Implementation:\n```python\n{code}\n```")
                
                # Add separators between strategies
                if strat != selected_bad[-1]:
                    lines.append("")
        
        if not lines:
            return "No strategies in the strategy bank match the criteria."
        
        return "\n".join(lines)

def parse_metric_definitions(source) -> list:
        """
        Support three kinds of inputs:
        - Raw string reply from an LLM
        - A JSON dict
        - A local file path
        """
        text = None
        payload = None

        if isinstance(source, dict):
            payload = source
        elif isinstance(source, str):
            if os.path.exists(source):
                try:
                    with open(source, "r", encoding="utf-8") as f:
                        text = f.read()
                except Exception:
                    return []
            else:
                text = source
        elif source is None:
            return []
        else:
            return []

        if payload is None:
            cleaned = extract_json_content(text or "")
            try:
                payload = json.loads(cleaned)
            except Exception:
                return []

        metrics = payload.get("metrics")
        if isinstance(metrics, list):
            return metrics

        reply_field = payload.get("reply")
        if isinstance(reply_field, str):
            cleaned = extract_json_content(reply_field)
            try:
                nested = json.loads(cleaned)
                metrics = nested.get("metrics")
                if isinstance(metrics, list):
                    return metrics
            except Exception:
                return []

        return []


def load_historical_metric_definitions(output_dir: Optional[str] = None) -> dict:
    """
    Load all metric definitions from historical metrics JSON files and build a name -> definition mapping.

    Args:
        output_dir: Directory path containing metrics_round_*.json files.

    Returns:
        A dict mapping metric name to its full definition (including name, formula, description, code, expected_behavior).
    """
    if not output_dir or not os.path.exists(output_dir):
        return {}
    
    historical_definitions = {}
    
    # Find all metrics_round_*.json files
    try:
        files = [
            f for f in os.listdir(output_dir)
            if f.startswith("metrics_round_") and f.endswith(".json")
        ]
        
        # Sort by filename so that newer files have priority
        files.sort(reverse=True)
        
        for filename in files:
            filepath = os.path.join(output_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Parse metric definitions from the reply field
                reply = data.get("reply", "")
                if reply:
                    metric_defs = parse_metric_definitions(reply)
                    for metric in metric_defs:
                        if not isinstance(metric, dict):
                            continue
                        metric_name = metric.get("name")
                        if metric_name and metric_name not in historical_definitions:
                            # Ensure the definition is complete
                            complete_def = ensure_complete_definition(metric)
                            historical_definitions[metric_name] = complete_def
                            
                            # Also use base metric name (without prefix) as key
                            base_name = extract_base_metric_name(metric_name)
                            if base_name != metric_name and base_name not in historical_definitions:
                                historical_definitions[base_name] = complete_def
            except Exception as e:
                logging.debug(f"[load_historical_metric_definitions] Skip file {filename}: {e}")
                continue
    except Exception as e:
        logging.warning(f"[load_historical_metric_definitions] Failed to load historical definitions: {e}")
    
    return historical_definitions


def ensure_complete_definition(metric_def: dict) -> dict:
    """
    Ensure that a metric definition contains all required fields, filling in any missing values.

    Args:
        metric_def: Metric definition dict.

    Returns:
        Completed metric definition dict.
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
    
    # Ensure all required fields are present
    complete_def = metric_def.copy()
    for field, default_value in required_fields.items():
        if field not in complete_def or complete_def[field] is None:
            complete_def[field] = default_value
    
    return complete_def


def fix_metric_name_format(name: str) -> str:
    """
    Fix formatting issues in metric names.

    Supported input formats:
    - "[prefix] metric_name" -> "[prefix] metric_name" (standard format)
    - "metric_name[prefix]" -> "[prefix] metric_name" (suffix format converted to standard)
    - "prefix] metric_name" -> "[prefix] metric_name" (missing leading '[')

    Args:
        name: Metric name that may be incorrectly formatted.

    Returns:
        A fixed metric name in standard format: "[prefix] metric_name".
    """
    if not isinstance(name, str):
        return name
    
    # If already in standard format (starts with '['), return as-is
    if name.startswith('['):
        return name
    
    # Handle "metric_name[prefix]" and convert to "[prefix] metric_name"
    suffix_pattern = r'^(.+?)\[([^\]]+)\]$'
    suffix_match = re.match(suffix_pattern, name)
    if suffix_match:
        metric = suffix_match.group(1).strip()
        prefix = suffix_match.group(2).strip()
        return f"[{prefix}] {metric}"
    
    # Fix format: match "xxx] metric_name" and add missing "["
    # Example: "inst_desp] probability_mass_moved" -> "[inst_desp] probability_mass_moved"
    fix_pattern = r'^([^\]]+)\]\s*(.+)$'
    match = re.match(fix_pattern, name)
    if match:
        prefix = match.group(1).strip()
        metric = match.group(2).strip()
        return f"[{prefix}] {metric}"
    
    return name


def extract_base_metric_name(name: str) -> str:
    """
    Extract the base metric name from a prefixed name.

    Supported input formats:
    - "[prefix] metric_name" -> "metric_name"
    - "metric_name[prefix]" -> "metric_name"
    - "metric_name" -> "metric_name" (no prefix)

    Examples:
    - "[inst_desp] probability_distribution_heaviness" -> "probability_distribution_heaviness"
    - "probability_distribution_heaviness[inst_desp]" -> "probability_distribution_heaviness"

    Args:
        name: Metric name that may contain a prefix.

    Returns:
        The base metric name with prefix removed.
    """
    if not isinstance(name, str):
        return name
    
    # First normalize the format (if incorrect) to "[prefix] metric_name"
    name = fix_metric_name_format(name)
    
    # Match "[xxx]" prefix format (standard format)
    prefix_pattern = r'^\[[^\]]+\]\s*(.+)$'
    match = re.match(prefix_pattern, name)
    if match:
        return match.group(1).strip()
    
    # If still not in standard format, try to handle "metric_name[prefix]" format
    suffix_pattern = r'^(.+?)\[([^\]]+)\]$'
    suffix_match = re.match(suffix_pattern, name)
    if suffix_match:
        return suffix_match.group(1).strip()
    
    return name


def persist_best_strategies(eval_result: dict,
                            metric_definitions: list,
                            store_path: str = STRATEGY_BANK_PATH,
                            save_all_strategies: bool = True,
                            output_dir: Optional[str] = None) -> list:
        """
        Persist strategies into the strategy bank.
        Now supports saving all strategies (good and bad) and tagging their categories.

        Args:
            eval_result: Evaluation result dict, containing ranking and summary.
            metric_definitions: List of metric definitions.
            store_path: Path to store the strategy bank.
            save_all_strategies: If True, save all strategies (strong/mid/weak);
                                 if False, only save best_metrics_to_save.
            output_dir: Optional output directory path, used to recover missing definitions from historical metrics JSON files.

        Returns:
            List of strategy entries that were saved.
        """
        if not isinstance(eval_result, dict):
            return []

        ranking = eval_result.get("ranking", [])
        if not ranking:
            return []
        
        # Ensure ranking already has dynamic categories; if not, assign them first
        if not any("dynamic_score" in item for item in ranking):
            ranking = assign_dynamic_categories(ranking)
            eval_result["ranking"] = ranking

        # Build ranking_map, supporting both raw and fixed names
        ranking_map = {}
        # Collect all prefixes used in ranking
        prefixes_used = set()
        for item in ranking:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if name:
                # Save both raw and fixed names for later lookup
                ranking_map[name] = item
                fixed_name = fix_metric_name_format(name)
                if fixed_name != name:
                    ranking_map[fixed_name] = item
                
                # Extract prefixes
                if name.startswith('['):
                    prefix_match = re.match(r'^\[([^\]]+)\]\s*(.+)$', name)
                    if prefix_match:
                        prefixes_used.add(prefix_match.group(1).strip())
                elif '[' in name and name.endswith(']'):
                    suffix_match = re.match(r'^(.+?)\[([^\]]+)\]$', name)
                    if suffix_match:
                        prefixes_used.add(suffix_match.group(2).strip())
        
        # Build definition_map, supporting both full and base names as keys
        definition_map = {}
        
        # First load definitions from historical files (if output_dir is provided)
        if output_dir:
            historical_definitions = load_historical_metric_definitions(output_dir)
            definition_map.update(historical_definitions)
        
        # Then add current-round definitions (override historical ones to ensure latest version)
        for item in metric_definitions:
            if not isinstance(item, dict):
                continue
            metric_name = item.get("name")
            if not metric_name:
                continue
            
            # Use the raw name as a key
            definition_map[metric_name] = item
            
            # Extract base name
            base_name = extract_base_metric_name(metric_name)
            
            # If the raw name is the same as the base name (no prefix), create mappings for all prefixes
            if base_name == metric_name:
                # Create mappings for all actually used prefixes
                all_prefixes = list(prefixes_used) + ["inst_dalle", "inst_desp_dalle", "img_dalle", "desp_dalle", "inst_desp", "inst", "img", "desp"]
                # Deduplicate
                all_prefixes = list(set(all_prefixes))
                
                for prefix in all_prefixes:
                    prefixed_name_std = f"[{prefix}] {base_name}"  # Standard format
                    prefixed_name_suffix = f"{base_name}[{prefix}]"  # Suffix format
                    
                    if prefixed_name_std not in definition_map:
                        definition_map[prefixed_name_std] = item
                    if prefixed_name_suffix not in definition_map:
                        definition_map[prefixed_name_suffix] = item
            else:
                # If the raw name has a prefix, also use base name as key and create mappings
                if base_name not in definition_map:
                    definition_map[base_name] = item
                
                # Extract original prefix
                if metric_name.startswith('['):
                    orig_prefix_match = re.match(r'^\[([^\]]+)\]\s*(.+)$', metric_name)
                    if orig_prefix_match:
                        orig_prefix = orig_prefix_match.group(1).strip()
                        # Create suffix-format mapping for the original prefix
                        suffix_format = f"{base_name}[{orig_prefix}]"
                        if suffix_format not in definition_map:
                            definition_map[suffix_format] = item

        entries = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary = eval_result.get("summary", {})

        if save_all_strategies:
            # Save all strategies, including strong, mid, weak
            for item in ranking:
                name = item.get("name")
                if not name:
                    continue
                
                # Fix malformed metric name (normalize to "[prefix] metric_name")
                name_fixed = fix_metric_name_format(name)
                    
                perf = item
                
                # Try to match definition: several formats in order of priority
                # 1. First try the fixed full name
                definition = definition_map.get(name_fixed, {})
                
                # 2. If not found, try the raw name
                if not definition:
                    definition = definition_map.get(name, {})
                
                # 3. If still not found, try the base name
                if not definition:
                    base_name = extract_base_metric_name(name)
                    definition = definition_map.get(base_name, {})
                
                # 4. If still not found, try suffix format (metric_name[prefix])
                if not definition:
                    suffix_pattern = r'^(.+?)\[([^\]]+)\]$'
                    suffix_match = re.match(suffix_pattern, name)
                    if suffix_match:
                        metric_base = suffix_match.group(1).strip()
                        prefix = suffix_match.group(2).strip()
                        std_format = f"[{prefix}] {metric_base}"
                        definition = definition_map.get(std_format, {})
                        if not definition:
                            definition = definition_map.get(metric_base, {})
                
                # If still not found, log a warning
                if not definition:
                    logging.warning(f"[persist_best_strategies] Cannot find metric definition: {name} (fixed: {name_fixed})")
                
                # Normalize storage format to suffix "metric_name[prefix]" to keep strategy bank consistent
                # If already in suffix format, keep as-is; otherwise convert from standard format
                if '[' in name and name.endswith(']') and not name.startswith('['):
                    # Already in suffix format "metric_name[prefix]"
                    final_name = name
                else:
                    # Convert from standard format "[prefix] metric_name" to suffix format
                    base_name = extract_base_metric_name(name_fixed)
                    # Extract prefix
                    if name_fixed.startswith('['):
                        prefix_match = re.match(r'^\[([^\]]+)\]\s*(.+)$', name_fixed)
                        if prefix_match:
                            prefix = prefix_match.group(1).strip()
                            final_name = f"{base_name}[{prefix}]"
                        else:
                            final_name = base_name
                    else:
                        final_name = base_name
                
                entry = {
                    "metric_name": final_name,
                    "performance": {
                        "auc": perf.get("auc"),
                        "accuracy": perf.get("accuracy"),
                        "tpr_at_5_fpr": perf.get("tpr_at_5_fpr"),
                    },
                    "analysis": perf.get("comment") or "No detailed analysis provided in the evaluation report.",
                    "definition": definition,
                    "saved_at": timestamp,
                    "overall_quality": summary.get("overall_quality"),
                    "category": perf.get("category", "mid"),  # Store dynamically assigned category
                    "dynamic_score": perf.get("score") or perf.get("dynamic_score"),  # Store dynamic score
                }
                entries.append(entry)
        else:
            # Save only best strategies (backward compatibility)
            best_names = summary.get("best_metrics_to_save") or []
            if not best_names:
                return []
            
            for name in best_names:
                # Fix malformed metric name
                name = fix_metric_name_format(name)
                
                perf = ranking_map.get(name, {})
                if not perf:
                    continue
                
                # Try to match definition: first by full name, then by base name
                definition = definition_map.get(name, {})
                if not definition:
                    base_name = extract_base_metric_name(name)
                    definition = definition_map.get(base_name, {})
                
                entry = {
                    "metric_name": name,
                    "performance": {
                        "auc": perf.get("auc"),
                        "accuracy": perf.get("accuracy"),
                        "tpr_at_5_fpr": perf.get("tpr_at_5_fpr"),
                    },
                    "analysis": perf.get("comment") or "No detailed analysis provided in the evaluation report.",
                    "definition": definition,
                    "saved_at": timestamp,
                    "overall_quality": summary.get("overall_quality"),
                    "category": perf.get("category", "strong"),  # Best strategies default to strong
                    "dynamic_score": perf.get("score") or perf.get("dynamic_score"),
                }
                entries.append(entry)

        append_to_strategy_bank(entries, store_path)
        return entries


def build_eval_guidance_text(eval_result: dict) -> str:
        """
        Extract useful_insights and next_round_strategy from evaluator's JSON result
        and generate natural-language guidance text for the next round of metric generation.
        """
        if not isinstance(eval_result, dict):
            return "No evaluation conclusion available. You are free to explore new metrics."

        ui = eval_result.get("useful_insights") or {}
        nrs = eval_result.get("next_round_strategy") or {}

        strong_families = ui.get("strong_metric_families") or []
        weak_families = ui.get("weak_metric_families") or []
        notes = ui.get("notes") or ""

        focus_metrics = nrs.get("focus_metrics") or ""
        new_ideas = nrs.get("new_ideas") or ""
        experiment_suggestions = nrs.get("experiment_suggestions") or ""

        lines = []
        lines.append("[Key Information from Previous Round Evaluation]")
        if strong_families:
            lines.append("1. Strong metric families recommended to retain and expand:")
            for fam in strong_families:
                lines.append(f"   - {fam}")
        if weak_families:
            lines.append("2. Weak metric families suggested to reduce or discard:")
            for fam in weak_families:
                lines.append(f"   - {fam}")
        if notes:
            lines.append(f"3. Evaluation summary notes: {notes}")

        lines.append("\n[Next Round Improvement Strategy Suggestions]")
        if focus_metrics:
            lines.append(f"- Focus areas for metrics: {focus_metrics}")
        if new_ideas:
            lines.append(f"- New ideas to try: {new_ideas}")
        if experiment_suggestions:
            lines.append(f"- Experiment setup suggestions: {experiment_suggestions}")

        return "\n".join(lines)


def load_last_eval_guidance(save_dir: str = OUTPUT_DIR) -> str:
        """
        Read the most recent evaluation guidance text generated as temporary memory.
        If it does not exist, return a default guidance message.
        """
        path = os.path.join(save_dir, "last_eval_guidance.txt")
        if not os.path.exists(path):
            return "No previous round evaluation conclusion available. You are free to design new metrics."
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return "Failed to read previous round evaluation conclusion. This section will be ignored temporarily."



def evaluate_metrics(agent: BaseAgent,
                     metrics_text_path: str,
                     save_dir: Optional[str] = None,
                     save_prefix: str = "eval",
                     metric_definitions: Optional[list] = None,
                     strategy_store_path: str = STRATEGY_BANK_PATH,
                     persist_strategies: bool = True):
        """
        Use an LLM to evaluate metric results (e.g., auc.txt):
        - Compare metric winners/losers and judge overall solution quality.
        - Analyze which metric families are useful.
        - Provide strategies for the next round of improvement.
        """
        # Read metric result text
        with open(metrics_text_path, "r", encoding="utf-8") as f:
            metrics_text = f.read()

        # Compose evaluation prompt
        eval_prompt = evaluator_prompt_template + "\n\n==== Experimental metric result text ====\n" + metrics_text

        # Call LLM for evaluation
        reply = agent.ask(eval_prompt)
        
        # Check whether reply is empty
        if not reply or reply.strip() == "":
            logging.warning("[evaluate_metrics] Agent returned empty reply, skipping evaluation and strategy persistence")
            print("[Warning] Agent returned empty reply; API call may have failed. Please check:")
            print("  1. Whether API config is correct (api_key and base_url in config.yaml)")
            print("  2. Whether network connectivity is normal")
            print("  3. Whether the model service is available")
            # Return an empty evaluation result to avoid downstream errors
            parsed = {
                "summary": {"overall_quality": "Error: Agent returned empty reply"},
                "ranking": [],
                "useful_insights": {},
                "next_round_strategy": {}
            }
        else:
            # Parse JSON; on failure, return the raw reply string
            cleaned_reply = extract_json_content(reply)
            try:
                parsed = json.loads(cleaned_reply)
            except Exception as e:
                logging.warning(f"[evaluate_metrics] Failed to parse JSON: {e}")
                parsed = {"raw_reply": reply}

        # Apply dynamic category assignment to ranking
        if isinstance(parsed, dict) and "ranking" in parsed:
            ranking = parsed.get("ranking", [])
            if isinstance(ranking, list) and ranking:
                parsed["ranking"] = assign_dynamic_categories(ranking)

        # Optional: save evaluation result + generate "temporary memory" guidance text
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = os.path.join(save_dir, f"{save_prefix}_{timestamp}.json")
            md_path = os.path.join(save_dir, f"{save_prefix}_{timestamp}.md")
            save_as_json(eval_prompt, reply, json_path)
            save_as_markdown(eval_prompt, reply, md_path)

            guidance_text = build_eval_guidance_text(parsed)
            guidance_path = os.path.join(save_dir, "last_eval_guidance.txt")
            with open(guidance_path, "w", encoding="utf-8") as gf:
                gf.write(guidance_text)

        if persist_strategies and parsed:
            persist_best_strategies(
                eval_result=parsed,
                metric_definitions=metric_definitions or [],
                store_path=strategy_store_path,
                save_all_strategies=True,  # Save all strategies (good and bad)
                output_dir=save_dir,  # Pass output_dir so that definitions can be recovered from historical files
            )

        return parsed



if __name__ == "__main__":
    agent = BaseAgent()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ########## Generate metrics (shared strategy-bank memory) ##########

    strategy_reference = build_strategy_reference()
    
    eval_guidance = load_last_eval_guidance(OUTPUT_DIR) 
    # By default, generate 5 metrics per round; adjust here or read from CLI if needed
    default_n_metrics = 5
    prompt_with_n = prompt_new.replace("{{N_METRICS}}", str(default_n_metrics))

    generation_prompt = (
        prompt_with_n
        + "\n\n==============================\n[Guidance from previous round evaluation]\n"
        + eval_guidance
        + "\n\n==============================\n[Historical high-quality metric strategy bank]\n"
        + strategy_reference
    )
    reply = agent.ask(generation_prompt)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(OUTPUT_DIR, f"reply_{timestamp}.json")
    json_path = os.path.join(OUTPUT_DIR, f"metrics.json")
    md_path = os.path.join(OUTPUT_DIR, f"reply_{timestamp}.md")
    save_as_json(generation_prompt, reply, json_path)
    save_as_markdown(generation_prompt, reply, md_path)
    # metric_definitions = parse_metric_definitions(reply)
    # metric_definitions = parse_metric_definitions(os.path.join(CURRENT_DIR, "outputs/metrics.json"))
    metric_definitions = parse_metric_definitions(os.path.join(CURRENT_DIR, "metrics.json"))
    ########## Evaluate metrics and update strategy bank ##########
    auc_path = os.path.join(os.path.dirname(__file__), "auc.txt")
    eval_result = evaluate_metrics(
        agent=agent,
        metrics_text_path=auc_path,
        save_dir=OUTPUT_DIR,
        save_prefix="metrics_eval",
        metric_definitions=metric_definitions,
        strategy_store_path=STRATEGY_BANK_PATH,
        persist_strategies=True,
    )

    print("Structured evaluation result:")
    print(eval_result)