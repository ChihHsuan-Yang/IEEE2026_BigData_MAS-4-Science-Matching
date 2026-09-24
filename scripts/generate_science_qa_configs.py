#!/usr/bin/env python3
"""Generate generic science-QA configs for the four paper protocols."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


MODES = ("baseline_llm", "single_agent_hint_llm", "per_hint_llm", "broadcast_hint_llm")
MODEL_FAMILIES = {
    "gpt_oss_120b": {
        "actor_model": "openai/gpt-oss-120b",
        "evaluator_model": "openai/gpt-oss-120b",
        "base_model_family": "gpt_oss_120b",
        "max_tokens": 1024,
    },
    "gemma4_31b_eval_oss120b": {
        "actor_model": "google/gemma-4-31B-it",
        "evaluator_model": "openai/gpt-oss-120b",
        "base_model_family": "gemma4_31b_eval_oss120b",
        "max_tokens": 1024,
    },
}
FAILURE_HINT = (
    "The submitted answer was judged incorrect. Re-check the requested answer format, "
    "the option-letter mapping if choices are present, numeric units if relevant, and the "
    "scientific reasoning. Do not repeat a rejected answer unless the prior issue was only formatting."
)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )


def is_evaluator(agent: dict[str, Any]) -> bool:
    return str(agent.get("agent_type", "")).strip().lower() == "evaluator"


def set_models(config: dict[str, Any], family: dict[str, Any]) -> None:
    for agent in config.get("agents", []):
        llm = agent.get("llm") or {}
        llm["model"] = family["evaluator_model"] if is_evaluator(agent) else family["actor_model"]
        llm["max_tokens"] = int(family["max_tokens"])
        agent["llm"] = llm
        memory = agent.get("memory") or {}
        if memory.get("summary_model"):
            memory["summary_model"] = llm["model"]
        agent["memory"] = memory


def set_prompt_preset(config: dict[str, Any], mode: str) -> None:
    config.pop("prompts", None)
    config["prompt_preset"] = {"path": f"configs/paper/prompt_presets/science_qa/{mode}.yaml"}


def set_roles(config: dict[str, Any], mode: str) -> None:
    role_by_name = {
        "Solo": "LLM-only scientific reasoning solver",
        "Planner": "Planner who identifies the relevant scientific concepts, answer format, and constraints",
        "Executor": "Executor who solves the scientific problem and submits the final boxed answer",
        "Reviewer": "Reviewer who checks scientific reasoning and answer-format compliance before submission",
        "Evaluator": "Constrained scientific-answer evaluator",
        "Mira": "Evidence-first scientific peer; check equations, units, option wording, and requested answer format.",
        "Theo": "Skeptical scientific peer; look for conceptual traps, unit mistakes, and unsupported assumptions.",
        "Sora": "Synthesis-oriented scientific peer; consolidate the strongest reasoning into one final answer.",
        "Rowan": "Skeptical scientific peer; look for conceptual traps, unit mistakes, and unsupported assumptions.",
        "Talia": "Synthesis-oriented scientific peer; consolidate the strongest reasoning into one final answer.",
    }
    deliberator_fields = {
        "group_description": "scientific reasoning broadcast-deliberation group",
        "answer_format_guidance": (
            "Final answers in this run must be boxed. Use option letters when the task is multiple-choice "
            "(including multi-letter answers such as \\boxed{ABD}); otherwise use the requested numeric, symbolic, "
            "or short textual answer."
        ),
        "intent_language_guidance": "using concise scientific reasoning",
        "review_focus_guidance": (
            "Check whether the candidate follows the requested answer format, option mapping, units, and scientific reasoning."
        ),
        "review_feedback_guidance": (
            "a concise scientific review explaining why you accept the answer or what needs repair"
        ),
        "final_answer_guidance": (
            "your best boxed answer, e.g. \\boxed{A}, \\boxed{ABD}, \\boxed{50.7 atm}, or \\boxed{x=2}"
        ),
    }
    for agent in config.get("agents", []):
        name = str(agent.get("name", ""))
        if name in role_by_name:
            agent["role_description"] = role_by_name[name]
        if mode == "broadcast_hint_llm" and str(agent.get("agent_type", "")) == "deliberator":
            agent.update(deliberator_fields)


def set_environment(config: dict[str, Any], mode: str) -> None:
    env = config.get("environment") or {}
    rule = env.get("rule") or {}
    evaluator = rule.get("evaluator") or {}
    evaluator["type"] = "llm"
    evaluator["data_name"] = "science-qa"
    evaluator["generic_failure_hint"] = FAILURE_HINT
    rule["evaluator"] = evaluator
    if mode == "broadcast_hint_llm":
        rule["discussion_instruction"] = (
            "Contribute one focused message to the scientific reasoning discussion. "
            "If you believe the group is ready to consider a concrete final answer, include:\n"
            "Candidate Answer:\n"
            "\\boxed{<answer>}"
        )
    env["rule"] = rule
    config["environment"] = env


def generate_config(base_root: Path, output_root: Path, model_family: str, family: dict[str, Any], mode: str) -> None:
    config = load_yaml(
        base_root
        / "lab_bench"
        / "llm_strict"
        / str(family["base_model_family"])
        / mode
        / "config.yaml"
    )
    config["name"] = f"science-qa-{mode}"
    set_prompt_preset(config, mode)
    set_roles(config, mode)
    set_environment(config, mode)
    set_models(config, family)
    write_yaml(output_root / model_family / mode / "config.yaml", config)


def generate_benchmark(output_root: Path, model_family: str, mode: str) -> None:
    tag = f"science_qa_{model_family}_{mode}"
    benchmark = {
        "name": mode,
        "task_name": f"paper/science_qa/{model_family}/{mode}",
        "dataset_loader": "jsonl",
        "dataset_path": "data/science-benchmarks/jeebench/all.jsonl",
        "trace_tag": tag,
        "artifact_tag": tag,
        "output_path": f"results/science-benchmarks/{model_family}/{mode}/all",
    }
    write_yaml(output_root / model_family / mode / "benchmark.yaml", benchmark)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", default="configs/paper")
    parser.add_argument("--output-root", default="configs/paper/science_qa")
    parser.add_argument(
        "--model-families",
        nargs="+",
        default=list(MODEL_FAMILIES),
        choices=sorted(MODEL_FAMILIES),
        help="Science-QA model-family bundles to generate.",
    )
    args = parser.parse_args()

    base_root = Path(args.base_root)
    output_root = Path(args.output_root)
    for model_family in args.model_families:
        family = MODEL_FAMILIES[model_family]
        for mode in MODES:
            generate_config(base_root, output_root, model_family, family, mode)
            generate_benchmark(output_root, model_family, mode)


if __name__ == "__main__":
    main()
