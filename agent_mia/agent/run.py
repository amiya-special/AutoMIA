"""Command-line interface for running the Agent Mia metric pipeline.

This module exposes a light-weight interface (function + CLI) so that
other code can invoke the generation/evaluation pipeline, while also
serving as a quick manual test entry point.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path when running as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agent_mia.agent.base_agent import (
    BaseAgent,
    OUTPUT_DIR,
    STRATEGY_BANK_PATH,
    build_strategy_reference,
    evaluate_metrics,
    load_last_eval_guidance,
    parse_metric_definitions,
    save_as_json,
    save_as_markdown,
)
from agent_mia.agent.prompt import prompt_new


def create_agent(config_path: Optional[str] = None) -> BaseAgent:
    """Factory helper so callers can inject alternative config paths."""
    return BaseAgent(config_path=config_path)


def generate_metrics(agent: BaseAgent,
                     output_dir: str = OUTPUT_DIR) -> Tuple[str, List[Dict[str, Any]]]:
    """Ask the agent for new metric ideas and persist the raw responses."""
    strategy_reference = build_strategy_reference()
    eval_guidance = load_last_eval_guidance(output_dir)
    prompt = (
        prompt_new
        + "\n\n==============================\n【上一轮评估结论指导】\n"
        + eval_guidance
        + "\n\n==============================\n【历史优秀指标策略库】\n"
        + strategy_reference
    )
    reply = agent.ask(prompt)
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"metrics_{ts}.json")
    md_path = os.path.join(output_dir, f"reply_{ts}.md")
    save_as_json(prompt, reply, json_path)
    save_as_markdown(prompt, reply, md_path)
    metric_defs = parse_metric_definitions(reply)
    return json_path, metric_defs


def evaluate_and_update(agent: BaseAgent,
                        metrics_text_path: str,
                        metric_definitions: List[Dict[str, Any]],
                        output_dir: str = OUTPUT_DIR,
                        strategy_path: str = STRATEGY_BANK_PATH) -> Dict[str, Any]:
    """Run evaluator model and update the persistent strategy bank."""
    return evaluate_metrics(
        agent=agent,
        metrics_text_path=metrics_text_path,
        save_dir=output_dir,
        save_prefix="metrics_eval",
        metric_definitions=metric_definitions,
        strategy_store_path=strategy_path,
        persist_strategies=True,
    )


def run_pipeline(config_path: Optional[str] = None,
                 metrics_text_path: Optional[str] = None,
                 output_dir: str = OUTPUT_DIR,
                 strategy_path: str = STRATEGY_BANK_PATH) -> Dict[str, Any]:
    """High-level interface combining generation + evaluation steps."""
    if metrics_text_path is None:
        metrics_text_path = os.path.join(CURRENT_DIR, "auc.txt")

    agent = create_agent(config_path)
    metrics_file, metric_defs = generate_metrics(agent, output_dir)
    eval_result = evaluate_and_update(
        agent=agent,
        metrics_text_path=metrics_text_path,
        metric_definitions=metric_defs,
        output_dir=output_dir,
        strategy_path=strategy_path,
    )
    return {
        "metrics_file": metrics_file,
        "metric_definitions": metric_defs,
        "evaluation": eval_result,
        "strategy_bank_path": strategy_path,
        "guidance_file": os.path.join(output_dir, "last_eval_guidance.txt"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Agent Mia pipeline once.")
    parser.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="Path to YAML config. Defaults to BaseAgent fallback.",
    )
    parser.add_argument(
        "--metrics",
        dest="metrics_text_path",
        default=None,
        help="Path to metrics text (default: agent/auc.txt).",
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        default=OUTPUT_DIR,
        help="Directory for generated replies/evaluations.",
    )
    parser.add_argument(
        "--strategy",
        dest="strategy_path",
        default=STRATEGY_BANK_PATH,
        help="Path to strategy bank JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point to manually test the pipeline."""
    args = _parse_args()
    result = run_pipeline(
        config_path=args.config_path,
        metrics_text_path=args.metrics_text_path,
        output_dir=args.output_dir,
        strategy_path=args.strategy_path,
    )
    print("策略库：", result["strategy_bank_path"])
    print("最新评估结论：", result["evaluation"])


if __name__ == "__main__":
    main()