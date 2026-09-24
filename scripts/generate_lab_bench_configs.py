#!/usr/bin/env python3
"""Generate LAB-Bench paper configs from the four Omni-MATH protocol configs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


MODES = ("baseline_llm", "single_agent_hint_llm", "per_hint_llm", "broadcast_hint_llm")
SLICES = {
    "llm_strict": "data/lab-bench-public/llm_strict/all.jsonl",
    "text_no_tool": "data/lab-bench-public/text_no_tool/all.jsonl",
}
MODEL_FAMILIES = {
    "gpt_oss_120b": {
        "actor_model": "openai/gpt-oss-120b",
        "evaluator_model": "openai/gpt-oss-120b",
    },
    "gemma4_31b_eval_oss120b": {
        "actor_model": "google/gemma-4-31B-it",
        "evaluator_model": "openai/gpt-oss-120b",
    },
    "gemma4_e4b_eval_oss120b": {
        "actor_model": "google/gemma-4-E4B-it",
        "evaluator_model": "openai/gpt-oss-120b",
    },
    "llama31_70b_eval_oss120b": {
        "actor_model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "evaluator_model": "openai/gpt-oss-120b",
    },
}
FAILURE_HINT = (
    "The selected option was judged incorrect. Re-check the question wording, "
    "the in-prompt evidence, and the option-letter mapping. Do not repeat a "
    "rejected letter unless the previous issue was only formatting."
)
LAB_BENCH_MAX_TOKENS = 1024


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )


def _agent_is_evaluator(agent: dict[str, Any]) -> bool:
    return str(agent.get("agent_type", "")).strip().lower() == "evaluator"


def _set_agent_models(config: dict[str, Any], *, actor_model: str, evaluator_model: str) -> None:
    for agent in config.get("agents", []):
        llm = agent.get("llm") or {}
        llm["model"] = evaluator_model if _agent_is_evaluator(agent) else actor_model
        llm["max_tokens"] = LAB_BENCH_MAX_TOKENS
        agent["llm"] = llm
        memory = agent.get("memory") or {}
        if memory.get("summary_model"):
            memory["summary_model"] = llm["model"]
        agent["memory"] = memory


def _apply_prompt_refs(config: dict[str, Any], mode: str) -> None:
    preset_path = f"configs/paper/prompt_presets/lab_bench_mc/{mode}.yaml"
    config.pop("prompts", None)
    config["prompt_preset"] = {"path": preset_path}

    for agent in config.get("agents", []):
        agent_type = str(agent.get("agent_type", "")).strip().lower()
        if mode == "baseline_llm":
            if agent_type == "solver":
                agent["prepend_prompt_template"] = {"prompt_ref": "solver_prepend_prompt"}
                agent["append_prompt_template"] = {"prompt_ref": "solver_append_prompt"}
            elif agent_type == "evaluator":
                agent["prepend_prompt_template"] = {"prompt_ref": "evaluator_prepend_prompt"}
                agent["append_prompt_template"] = {"prompt_ref": "evaluator_append_prompt"}
        elif mode == "single_agent_hint_llm":
            if agent_type == "solver":
                agent["first_attempt_prepend_prompt_template"] = {"prompt_ref": "first_attempt_prepend_prompt"}
                agent["first_attempt_append_prompt_template"] = {"prompt_ref": "first_attempt_append_prompt"}
                agent["prepend_prompt_template"] = {"prompt_ref": "retry_prepend_prompt"}
                agent["append_prompt_template"] = {"prompt_ref": "retry_append_prompt"}
            elif agent_type == "evaluator":
                agent["prepend_prompt_template"] = {"prompt_ref": "evaluator_prepend_prompt"}
                agent["append_prompt_template"] = {"prompt_ref": "evaluator_append_prompt"}
        elif mode == "per_hint_llm":
            if agent_type == "solver":
                agent["prepend_prompt_template"] = {"prompt_ref": "solver_prepend_prompt"}
                agent["append_prompt_template"] = {"prompt_ref": "solver_append_prompt"}
            elif agent_type == "executor":
                agent["prepend_prompt_template"] = {"prompt_ref": "executor_prepend_prompt"}
                agent["append_prompt_template"] = {"prompt_ref": "executor_append_prompt"}
            elif agent_type == "critic":
                agent["prepend_prompt_template"] = {"prompt_ref": "critic_prepend_prompt"}
                agent["append_prompt_template"] = {"prompt_ref": "critic_append_prompt"}
            elif agent_type == "evaluator":
                agent["prepend_prompt_template"] = {"prompt_ref": "evaluator_prepend_prompt"}
                agent["append_prompt_template"] = {"prompt_ref": "evaluator_append_prompt"}
        elif mode == "broadcast_hint_llm":
            if agent_type == "deliberator":
                agent["prepend_prompt_template"] = {"prompt_ref": "deliberator_prepend_prompt"}
                agent["append_prompt_template"] = {"prompt_ref": "deliberator_append_prompt"}
            elif agent_type == "evaluator":
                agent["prepend_prompt_template"] = {"prompt_ref": "evaluator_prepend_prompt"}
                agent["append_prompt_template"] = {"prompt_ref": "evaluator_append_prompt"}


def _set_lab_roles(config: dict[str, Any], mode: str) -> None:
    role_by_name = {
        "Solo": "Text-only LAB-Bench multiple-choice solver",
        "Planner": "Planner who identifies the relevant biology evidence and option constraints",
        "Executor": "Executor who selects one option letter from the plan and prompt evidence",
        "Reviewer": "Reviewer who checks whether the selected option letter is justified before submission",
        "Evaluator": "Constrained LAB-Bench multiple-choice evaluator",
        "Mira": "Evidence-first biology peer; check prompt details and option wording before proposing an answer.",
        "Theo": "Skeptical biology peer; look for misleading distractors, sequence/protocol traps, and unsupported assumptions.",
        "Sora": "Synthesis-oriented biology peer; consolidate the strongest evidence into one option-letter answer.",
        "Rowan": "Skeptical biology peer; look for misleading distractors, sequence/protocol traps, and unsupported assumptions.",
        "Talia": "Synthesis-oriented biology peer; consolidate the strongest evidence into one option-letter answer.",
    }
    deliberator_fields = {
        "group_description": "biology multiple-choice LAB-Bench deliberation group",
        "answer_format_guidance": (
            "LAB-Bench answers in this run must be exactly one listed multiple-choice option "
            "letter in boxed form, for example \\boxed{A}."
        ),
        "intent_language_guidance": "using concise evidence-based biology reasoning",
        "review_focus_guidance": (
            "Reason carefully about whether the selected letter follows from the prompt evidence "
            "and matches the option text."
        ),
        "review_feedback_guidance": (
            "a concise evidence-based review explaining why you accept the option letter or what is unsupported/wrong"
        ),
        "final_answer_guidance": (
            "your best boxed listed option-letter answer, for example \\boxed{A}"
        ),
    }
    for agent in config.get("agents", []):
        name = str(agent.get("name", ""))
        if name in role_by_name:
            agent["role_description"] = role_by_name[name]
        if mode == "broadcast_hint_llm" and str(agent.get("agent_type", "")) == "deliberator":
            agent.update(deliberator_fields)


def _set_lab_environment(config: dict[str, Any], mode: str) -> None:
    rule = (config.get("environment") or {}).get("rule") or {}
    evaluator = rule.get("evaluator") or {}
    evaluator["type"] = "llm"
    evaluator["data_name"] = "lab-bench"
    evaluator["generic_failure_hint"] = FAILURE_HINT
    rule["evaluator"] = evaluator
    if mode == "broadcast_hint_llm":
        rule["discussion_instruction"] = (
            "Contribute one focused message to the LAB-Bench multiple-choice discussion. "
            "Avoid repeating points already made. If you believe the group is ready to consider "
            "a concrete final answer, include a short section exactly in this form:\n"
            "Candidate Answer:\n"
            "\\boxed{<listed option letter>}"
        )
    config["environment"]["rule"] = rule


def _generate_config(base_root: Path, out_dir: Path, mode: str, actor_model: str, evaluator_model: str) -> None:
    config = _load_yaml(base_root / mode / "config.yaml")
    config["name"] = f"lab-bench-{mode}"
    _apply_prompt_refs(config, mode)
    _set_lab_roles(config, mode)
    _set_lab_environment(config, mode)
    _set_agent_models(config, actor_model=actor_model, evaluator_model=evaluator_model)
    _write_yaml(out_dir / "config.yaml", config)


def _generate_benchmark(out_dir: Path, *, slice_name: str, model_family: str, mode: str, dataset_path: str) -> None:
    tag = f"lab_bench_{slice_name}_{model_family}_{mode}"
    benchmark = {
        "name": mode,
        "task_name": f"paper/lab_bench/{slice_name}/{model_family}/{mode}",
        "dataset_loader": "jsonl",
        "dataset_path": dataset_path,
        "trace_tag": tag,
        "artifact_tag": tag,
        "output_path": f"results/lab-bench-public/{slice_name}/{model_family}/{mode}/all",
    }
    _write_yaml(out_dir / "benchmark.yaml", benchmark)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", default="configs/paper")
    parser.add_argument("--output-root", default="configs/paper/lab_bench")
    args = parser.parse_args()

    base_root = Path(args.base_root)
    output_root = Path(args.output_root)
    for slice_name, dataset_path in SLICES.items():
        for model_family, models in MODEL_FAMILIES.items():
            for mode in MODES:
                out_dir = output_root / slice_name / model_family / mode
                _generate_config(
                    base_root,
                    out_dir,
                    mode,
                    actor_model=models["actor_model"],
                    evaluator_model=models["evaluator_model"],
                )
                _generate_benchmark(
                    out_dir,
                    slice_name=slice_name,
                    model_family=model_family,
                    mode=mode,
                    dataset_path=dataset_path,
                )
                print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
