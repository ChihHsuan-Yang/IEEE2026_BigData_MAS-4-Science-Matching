"""Build a readable Hugging Face staging layout for MAS protocol traces.

The older HF staging layout mirrors raw trace trees. This script builds a
public-resource-oriented layout:

  registry/
  experiments/<benchmark_id>/<short_run_id>/
    tables...
    configs...
    by_tier/<tier>/<problem_id>/outcomes.csv
    by_tier/<tier>/<problem_id>/clear_traces/<protocol>.json

The browsing hierarchy intentionally keeps short, human-readable paths and
removes chunk/worker folders. Original raw paths are preserved only in trace
manifests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROTOCOLS = [
    ("baseline_llm", "Baseline", 0, "Direct one-shot solving with the actor model."),
    ("single_agent", "Single", 1, "Single-agent iterative solver/evaluator wrapper."),
    ("per", "PER", 2, "Planner-executor-reviewer collaboration protocol."),
    ("broadcast", "Broadcast", 3, "Broadcast-style multi-agent deliberation."),
]

GPT_RAW_PROTOCOL_DIRS = {
    "baseline_llm": "baseline_llm",
    "single_agent": "single_agent",
    "per": "per",
    "broadcast": "broadcast",
}

GEMMA_RAW_PROTOCOL_DIRS = {
    "baseline_gemma3_27b_eval_oss120b": "baseline_llm",
    "single_agent_hint_gemma3_27b_eval_oss120b": "single_agent",
    "per_hint_gemma3_27b_eval_oss120b": "per",
    "broadcast_hint_gemma3_27b_eval_oss120b": "broadcast",
}

PROTOCOL_TO_COLUMNS = {
    "baseline_llm": {
        "correct": "baseline_final_passed",
        "tokens": "baseline_total_tokens",
        "calls": "baseline_model_calls",
        "wall": "baseline_wall_time_seconds",
    },
    "single_agent": {
        "correct": "single_agent_final_passed",
        "tokens": "single_agent_total_tokens",
        "calls": "single_agent_model_calls",
        "wall": "single_agent_wall_time_seconds",
    },
    "per": {
        "correct": "per_final_passed",
        "tokens": "per_total_tokens",
        "calls": "per_model_calls",
        "wall": "per_wall_time_seconds",
    },
    "broadcast": {
        "correct": "broadcast_final_passed",
        "tokens": "broadcast_total_tokens",
        "calls": "broadcast_model_calls",
        "wall": "broadcast_wall_time_seconds",
    },
}

PROTOCOL_OUTCOME_FIELDS = [
    "experiment_id",
    "problem_id",
    "legacy_problem_id",
    "protocol_id",
    "raw_protocol_name",
    "actor_set_id",
    "evaluator_id",
    "temperature",
    "final_correct",
    "first_pass_correct",
    "final_answer",
    "answer_parse_status",
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "model_calls",
    "evaluator_calls",
    "wall_time_seconds",
    "trace_path",
    "error_status",
]

PROBLEM_INDEX_FIELDS = [
    "experiment_id",
    "problem_id",
    "legacy_problem_id",
    "source_problem_id",
    "benchmark_id",
    "subset",
    "tier",
    "difficulty_tier",
    "difficulty_score",
    "problem",
    "answer_available",
]

GPT_EXPERIMENT_ID = "omnimath2_gpt_oss_120b"
GPT_EXPERIMENT_PATH = Path("experiments/omnimath2/gpt_oss_120b")
GPT_RUN_DATE = "20260423"

GEMMA_EXPERIMENT_ID = "omnimath2_gemma3_27b"
GEMMA_EXPERIMENT_PATH = Path("experiments/omnimath2/gemma3_27b")
GEMMA_RUN_DATE = "20260506"

SENSITIVE_PATTERNS = [
    (re.compile(r"/Users/[^\s\"']+"), "<LOCAL_PATH>"),
    (re.compile(r"/home/[^\s\"']+"), "<REMOTE_HOME_PATH>"),
    (re.compile(r"/lus/[^\s\"']+"), "<CLUSTER_PATH>"),
    (re.compile(r"/grand/[^\s\"']+"), "<CLUSTER_PATH>"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+"), r"\1<REDACTED>"),
    (re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,\"']+"), r"\1<REDACTED>"),
    (re.compile(r"(?i)(authorization\s*[:=]\s*)[^\n,]+"), r"\1<REDACTED>"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def sanitize_str(value: str) -> str:
    out = value
    for pattern, replacement in SENSITIVE_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def sanitize_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return sanitize_str(obj)
    if isinstance(obj, list):
        return [sanitize_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): sanitize_obj(v) for k, v in obj.items()}
    return obj


def boolish(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return "1" if value else "0"
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return "1"
    if text in {"false", "0", "no"}:
        return "0"
    return str(value)


def floatish(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def tier_label(value: str | int | None) -> str:
    if value is None or value == "":
        return ""
    text = str(value)
    digits = re.sub(r"\D", "", text)
    if digits:
        return f"tier_{int(digits):02d}"
    return sanitize_path_token(text)


def sanitize_path_token(value: Any) -> str:
    text = str(value)
    text = text.replace(":", "__")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def omni_raw_key(problem: Any, source: Any, difficulty: Any) -> tuple[str, str, str]:
    difficulty_text = ""
    if difficulty not in (None, ""):
        try:
            difficulty_text = str(float(difficulty))
        except (TypeError, ValueError):
            difficulty_text = str(difficulty)
    return (normalize_text(problem), normalize_text(source), difficulty_text)


def difficulty_tier_from_score(value: Any) -> str:
    try:
        tier = max(1, min(10, int(round(float(value)))))
        return f"tier_{tier:02d}"
    except (TypeError, ValueError):
        return tier_label(value)


def official_omnimath2_records(
    repo_root: Path,
) -> tuple[dict[tuple[str, str, str], str], list[dict[str, Any]], dict[str, str]]:
    raw_path = repo_root / "data/omni-math-2-filtered/raw/Omni-Math-2.jsonl"
    lookup: dict[tuple[str, str, str], str] = {}
    ids_by_tier: dict[str, list[str]] = defaultdict(list)
    filtered_records: list[dict[str, Any]] = []
    for row in read_jsonl(raw_path):
        if row.get("tags"):
            continue
        problem_id = str(row.get("id"))
        tier = difficulty_tier_from_score(row.get("difficulty"))
        lookup[omni_raw_key(row.get("problem"), row.get("source"), row.get("difficulty"))] = problem_id
        ids_by_tier[tier].append(problem_id)
        filtered_records.append(
            {
                "id": problem_id,
                "problem": row.get("problem"),
                "solution": row.get("solution"),
                "answer": row.get("answer"),
                "tags": row.get("tags", []),
                "source": row.get("source"),
                "domain": row.get("domain", []),
                "difficulty": row.get("difficulty"),
                "difficulty_tier": tier,
            }
        )
    legacy_id_map: dict[str, str] = {}
    for tier, ids in ids_by_tier.items():
        compact_tier = tier.replace("_", "")
        for i, official_id in enumerate(ids, start=1):
            legacy_id_map[f"{compact_tier}:{i}"] = official_id
    return lookup, filtered_records, legacy_id_map


def public_problem_id(row: dict[str, Any], official_lookup: dict[tuple[str, str, str], str]) -> str:
    key = omni_raw_key(row.get("problem"), row.get("source"), row.get("difficulty"))
    if key in official_lookup:
        return official_lookup[key]
    return sanitize_path_token(row.get("problem_id") or row.get("id") or "unknown")


def problem_lookup(
    rows: list[dict[str, Any]],
    *,
    tier_field: str,
    problem_field: str = "problem",
    source_field: str = "source",
    problem_id_field: str = "problem_id",
    sort_field: str | None = None,
) -> dict[str, dict[tuple[str, str, str], list[str]]]:
    full: dict[tuple[str, str, str], list[tuple[int, str]]] = defaultdict(list)
    no_source: dict[tuple[str, str, str], list[tuple[int, str]]] = defaultdict(list)
    for i, row in enumerate(rows):
        tier = tier_label(row.get(tier_field))
        problem = normalize_text(row.get(problem_field))
        source = normalize_text(row.get(source_field))
        pid = str(row.get(problem_id_field, ""))
        if not pid:
            continue
        if sort_field and str(row.get(sort_field, "")).isdigit():
            order = int(str(row.get(sort_field)))
        else:
            order = i
        full[(tier, problem, source)].append((order, pid))
        no_source[(tier, problem, "")].append((order, pid))

    def sorted_ids(d: dict[tuple[str, str, str], list[tuple[int, str]]]) -> dict[tuple[str, str, str], list[str]]:
        return {k: [pid for _order, pid in sorted(v)] for k, v in d.items()}

    return {"full": sorted_ids(full), "no_source": sorted_ids(no_source)}


def resolve_problem_id(
    obj: dict[str, Any],
    tier: str,
    protocol_id: str,
    lookup: dict[str, dict[tuple[str, str, str], list[str]]],
    counters: dict[tuple[str, tuple[str, str, str]], int],
) -> str:
    tier_key = tier_label(tier)
    problem = normalize_text(obj.get("problem") or obj.get("question") or obj.get("input"))
    source = normalize_text(obj.get("source"))
    keys = [
        ("full", (tier_key, problem, source)),
        ("no_source", (tier_key, problem, "")),
    ]
    for lookup_name, key in keys:
        ids = lookup[lookup_name].get(key)
        if not ids:
            continue
        counter_key = (protocol_id, key)
        idx = counters[counter_key]
        counters[counter_key] += 1
        if idx < len(ids):
            return ids[idx]
        return ids[-1]
    return problem_id_from_trace(obj, tier_key.replace("_", ""))


def problem_id_from_trace(obj: dict[str, Any], tier: str) -> str:
    qid = obj.get("question_id")
    if qid is None:
        qid = obj.get("answer_number") or obj.get("source_id") or "unknown"
    return f"{tier}:{qid}"


def first_metric_file(folder: Path) -> Path | None:
    files = sorted(folder.glob("*.metrics.jsonl"))
    return files[0] if files else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metric_summary(metric: dict[str, Any] | None) -> dict[str, str]:
    if not metric:
        return {
            "final_correct": "",
            "first_pass_correct": "",
            "total_tokens": "",
            "prompt_tokens": "",
            "completion_tokens": "",
            "model_calls": "",
            "evaluator_calls": "",
            "wall_time_seconds": "",
        }
    tokens = metric.get("total_tokens") or {}
    agent_metrics = metric.get("agent_metrics") or {}
    evaluator_calls = ""
    if isinstance(agent_metrics, dict) and "Evaluator" in agent_metrics:
        evaluator_calls = str((agent_metrics["Evaluator"] or {}).get("invocation_count", ""))
    return {
        "final_correct": boolish(metric.get("correct")),
        "first_pass_correct": boolish(
            metric.get("first_system_try_correct", metric.get("first_round_correct"))
        ),
        "total_tokens": floatish(tokens.get("total_tokens")),
        "prompt_tokens": floatish(tokens.get("prompt_tokens")),
        "completion_tokens": floatish(tokens.get("completion_tokens")),
        "model_calls": floatish(metric.get("total_model_calls")),
        "evaluator_calls": evaluator_calls,
        "wall_time_seconds": floatish(metric.get("wall_time_seconds")),
    }


def trace_payload(
    *,
    experiment_id: str,
    problem_id: str,
    protocol_id: str,
    raw_protocol_name: str,
    actor_set_id: str,
    evaluator_id: str,
    obj: dict[str, Any],
    outcome: dict[str, Any],
    source_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    return sanitize_obj(
        {
            "schema_version": "mas-protocol-trace-v1",
            "experiment_id": experiment_id,
            "problem_id": problem_id,
            "protocol_id": protocol_id,
            "raw_protocol_name": raw_protocol_name,
            "actor_set_id": actor_set_id,
            "evaluator_id": evaluator_id,
            "temperature": 0,
            "redaction_status": "sanitized_public_staging",
            "problem": {
                "source_problem_id": obj.get("question_id"),
                "dataset_name": obj.get("dataset_name"),
                "source": obj.get("source"),
                "difficulty": obj.get("difficulty"),
                "difficulty_tier": obj.get("difficulty_tier"),
                "domain": obj.get("domain"),
                "problem": obj.get("problem") or obj.get("question") or obj.get("input"),
                "answer": obj.get("answer") or obj.get("label"),
            },
            "outcome": outcome,
            "trace": {
                "response": obj.get("response"),
                "planner_output": obj.get("planner_output"),
                "model_generation": obj.get("model_generation"),
                "final_answer": obj.get("final_answer"),
                "per_evaluation": obj.get("per_evaluation"),
                "logs": obj.get("logs"),
                "task_run_artifact": obj.get("task_run_artifact"),
            },
            "source_reference": {
                "raw_source_path": str(source_path.relative_to(repo_root)),
                "note": "Original chunk/worker path is kept only for reproducibility; public browsing follows Omni-MATH-2 problem ids under omnimath2/<protocol>/<tier>/<problem_id>/.",
            },
        }
    )


def protocol_trace_path(protocol_id: str, tier: str, problem_id: str, run_id: str) -> Path:
    return (
        Path("omnimath2")
        / protocol_id
        / tier_label(tier)
        / sanitize_path_token(problem_id)
        / f"{run_id}.trace.json"
    )


def protocol_outcome_path(protocol_id: str, tier: str, problem_id: str, run_id: str) -> Path:
    return (
        Path("omnimath2")
        / protocol_id
        / tier_label(tier)
        / sanitize_path_token(problem_id)
        / f"{run_id}.outcome.csv"
    )


def copy_sanitized_text(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(sanitize_str(src.read_text(encoding="utf-8")), encoding="utf-8")


def pick_config_files(repo_root: Path, source_root: Path, protocol_globs: dict[str, str]) -> dict[str, Path]:
    picked: dict[str, Path] = {}
    for protocol_id, glob_pattern in protocol_globs.items():
        matches = sorted(source_root.glob(glob_pattern))
        if matches:
            picked[protocol_id] = matches[0]
    return picked


def build_gpt_experiment(repo_root: Path, staging: Path) -> dict[str, Any]:
    experiment_id = GPT_EXPERIMENT_ID
    run_id = "gpt_oss_120b"
    exp_dir = staging / "omnimath2" / "runs" / run_id
    tables_dir = exp_dir / "tables"
    raw_root = repo_root / "results/trace/gpt_oss_120b"
    benchmark_path = repo_root / "results/emnlp_routing/benchmark/routing_benchmark.csv"
    benchmark_rows = read_csv(benchmark_path)
    official_lookup, official_records, official_id_by_legacy = official_omnimath2_records(repo_root)
    public_id_by_legacy = {
        row["problem_id"]: official_id_by_legacy.get(row["problem_id"], public_problem_id(row, official_lookup))
        for row in benchmark_rows
    }
    by_problem: dict[str, dict[str, Any]] = {r["problem_id"]: r for r in benchmark_rows}
    resolver_lookup = problem_lookup(
        benchmark_rows,
        tier_field="tier",
        problem_field="problem",
        source_field="source",
        problem_id_field="problem_id",
        sort_field="global_example_idx",
    )
    resolver_counters: dict[tuple[str, tuple[str, str, str]], int] = defaultdict(int)

    problem_index_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    protocol_rows: list[dict[str, Any]] = []
    trace_manifest_rows: list[dict[str, Any]] = []
    problem_outcomes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    problem_meta: dict[str, dict[str, Any]] = {}
    seen_problem_protocol: set[tuple[str, str]] = set()

    for row in benchmark_rows:
        legacy_problem_id = row["problem_id"]
        problem_id = public_id_by_legacy[legacy_problem_id]
        tier = row["tier"]
        group_dir = staging / "omnimath2" / "problems" / tier_label(tier) / sanitize_path_token(problem_id)
        problem_meta[problem_id] = {
            "experiment_id": experiment_id,
            "problem_id": problem_id,
            "legacy_problem_id": legacy_problem_id,
            "source_problem_id": problem_id,
            "benchmark_id": "omnimath2",
            "subset": row.get("source", ""),
            "tier": tier,
            "difficulty_tier": row.get("difficulty_tier", ""),
            "difficulty_score": row.get("difficulty", ""),
            "problem": row.get("problem", ""),
            "answer_available": "1",
        }
        problem_index_rows.append(problem_meta[problem_id])
        matched_rows.append(
            {
                "experiment_id": experiment_id,
                "problem_id": problem_id,
                "legacy_problem_id": legacy_problem_id,
                "split": "",
                "subset": row.get("source", ""),
                "difficulty_tier": row.get("difficulty_tier", ""),
                "baseline_correct": boolish(row.get("baseline_final_passed")),
                "single_correct": boolish(row.get("single_agent_final_passed")),
                "per_correct": boolish(row.get("per_final_passed")),
                "broadcast_correct": boolish(row.get("broadcast_final_passed")),
                "baseline_tokens": row.get("baseline_total_tokens", ""),
                "single_tokens": row.get("single_agent_total_tokens", ""),
                "per_tokens": row.get("per_total_tokens", ""),
                "broadcast_tokens": row.get("broadcast_total_tokens", ""),
                "oracle_cheapest_successful": row.get("cheapest_successful_protocol", ""),
                "oracle_token_cost": oracle_token_cost(row),
                "any_protocol_solved": "0"
                if row.get("cheapest_successful_protocol") == "none"
                else "1",
            }
        )
        oracle_rows.append(
            {
                "experiment_id": experiment_id,
                "problem_id": problem_id,
                "legacy_problem_id": legacy_problem_id,
                "oracle_cheapest_successful": row.get("cheapest_successful_protocol", ""),
                "oracle_token_cost": oracle_token_cost(row),
                "oracle_solved": "0"
                if row.get("cheapest_successful_protocol") == "none"
                else "1",
                "oracle_label_scope": "single_realization_observed_matched_run",
            }
        )
        group_dir.mkdir(parents=True, exist_ok=True)
        write_json(group_dir / "problem.json", problem_meta[problem_id])

    for tier_dir in sorted(raw_root.glob("tier*")):
        if not tier_dir.is_dir() or tier_dir.name == "report":
            continue
        tier = tier_dir.name
        for raw_protocol_dir, protocol_id in GPT_RAW_PROTOCOL_DIRS.items():
            proto_root = tier_dir / raw_protocol_dir
            if not proto_root.exists():
                continue
            for results_path in sorted(proto_root.glob("chunk*_part*/results.jsonl")):
                metric_rows = read_jsonl(first_metric_file(results_path.parent)) if first_metric_file(results_path.parent) else []
                with results_path.open("r", encoding="utf-8") as f:
                    for line_idx, line in enumerate(f):
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        legacy_problem_id = resolve_problem_id(
                            obj,
                            tier,
                            protocol_id,
                            resolver_lookup,
                            resolver_counters,
                        )
                        if legacy_problem_id not in by_problem:
                            continue
                        problem_id = public_id_by_legacy[legacy_problem_id]
                        if (problem_id, protocol_id) in seen_problem_protocol:
                            continue
                        seen_problem_protocol.add((problem_id, protocol_id))
                        metric = metric_rows[line_idx] if line_idx < len(metric_rows) else None
                        fallback = by_problem[legacy_problem_id]
                        columns = PROTOCOL_TO_COLUMNS[protocol_id]
                        outcome = metric_summary(metric)
                        if not outcome["final_correct"]:
                            outcome["final_correct"] = boolish(fallback.get(columns["correct"]))
                        if not outcome["total_tokens"]:
                            outcome["total_tokens"] = fallback.get(columns["tokens"], "")
                        if not outcome["model_calls"]:
                            outcome["model_calls"] = fallback.get(columns["calls"], "")
                        if not outcome["wall_time_seconds"]:
                            outcome["wall_time_seconds"] = fallback.get(columns["wall"], "")
                        outcome.update(
                            {
                                "legacy_problem_id": legacy_problem_id,
                                "final_answer": obj.get("final_answer", ""),
                                "answer": obj.get("answer") or obj.get("label") or "",
                                "answer_parse_status": "parsed",
                                "trace_path": str(protocol_trace_path(protocol_id, tier, problem_id, run_id)),
                            }
                        )
                        out_row = protocol_outcome_row(
                            experiment_id=experiment_id,
                            problem_id=problem_id,
                            protocol_id=protocol_id,
                            raw_protocol_name=raw_protocol_dir,
                            actor_set_id="gpt-oss-120b",
                            evaluator_id="gpt-oss-120b",
                            outcome=outcome,
                        )
                        protocol_rows.append(out_row)
                        problem_outcomes[problem_id].append(out_row)
                        trace_rel = Path(outcome["trace_path"])
                        trace_manifest_rows.append(
                            {
                                "experiment_id": experiment_id,
                                "problem_id": problem_id,
                                "legacy_problem_id": legacy_problem_id,
                                "protocol_id": protocol_id,
                                "trace_path": str(trace_rel),
                                "source_path": str(results_path.relative_to(repo_root)),
                                "source_line_number": line_idx + 1,
                                "redaction_status": "sanitized_public_staging",
                            }
                        )
                        write_json(
                            staging / trace_rel,
                            trace_payload(
                                experiment_id=experiment_id,
                                problem_id=problem_id,
                                protocol_id=protocol_id,
                                raw_protocol_name=raw_protocol_dir,
                                actor_set_id="gpt-oss-120b",
                                evaluator_id="gpt-oss-120b",
                                obj=obj,
                                outcome=outcome,
                                source_path=results_path,
                                repo_root=repo_root,
                            ),
                        )
                        write_csv(
                            staging / protocol_outcome_path(protocol_id, tier, problem_id, run_id),
                            [out_row],
                            PROTOCOL_OUTCOME_FIELDS,
                        )

    write_experiment_common(
        exp_dir=exp_dir,
        experiment_id=experiment_id,
        title="Omni-MATH 2 matched four-protocol run with gpt-oss-120b",
        benchmark_id="omnimath2",
        slice_id="competition-math-4181",
        actor_set_id="gpt-oss-120b",
        evaluator_id="gpt-oss-120b",
        n_problems=len(benchmark_rows),
        grouping="omnimath2/<protocol>/<tier>/<official_omnimath2_id>/<run_id>.trace.json",
        source_description="Full 4,181-problem Omni-MATH 2 competition math slice used for cost-aware protocol routing.",
        notes="All four protocols are observed for every problem. Temperature is 0 for actor and evaluator calls in configs.",
    )
    copy_configs(
        repo_root,
        exp_dir,
        pick_config_files(
            repo_root,
            raw_root,
            {
                "baseline_llm": "tier01/baseline_llm/chunk000_part000/config.yaml",
                "single_agent": "tier01/single_agent/chunk000_part000/config.yaml",
                "per": "tier01/per/chunk000_part000/config.yaml",
                "broadcast": "tier01/broadcast/chunk000_part000/config.yaml",
            },
        ),
    )
    write_omnimath2_problems_jsonl(staging, official_records)
    write_csv(
        staging / "omnimath2" / "tables" / "problem_index.csv",
        problem_index_rows,
        PROBLEM_INDEX_FIELDS,
    )
    write_tables(
        tables_dir,
        problem_index_rows,
        protocol_rows,
        matched_rows,
        aggregate_metrics(protocol_rows, experiment_id),
        oracle_rows,
        trace_manifest_rows,
    )
    return experiment_summary(
        experiment_id,
        "omnimath2",
        "competition-math-4181",
        len(benchmark_rows),
        "gpt-oss-120b",
        "gpt-oss-120b",
        run_date=GPT_RUN_DATE,
    )


def build_gemma_experiment(repo_root: Path, staging: Path) -> dict[str, Any]:
    experiment_id = GEMMA_EXPERIMENT_ID
    run_id = "gemma3_27b"
    exp_dir = staging / "omnimath2" / "runs" / run_id
    tables_dir = exp_dir / "tables"
    raw_root = repo_root / "results/trace/gemma3_27b"
    oracle_path = repo_root / "results/emnlp_routing/revision_strengthening/gemma3_actor_matched_subset_oracle.csv"
    oracle_source_rows = read_csv(oracle_path)
    official_lookup, _official_records, official_id_by_legacy = official_omnimath2_records(repo_root)
    full_benchmark_rows = read_csv(repo_root / "results/emnlp_routing/benchmark/routing_benchmark.csv")
    public_id_by_legacy = {
        row["problem_id"]: official_id_by_legacy.get(row["problem_id"], public_problem_id(row, official_lookup))
        for row in full_benchmark_rows
    }
    by_problem: dict[str, dict[str, str]] = {r["problem_id"]: r for r in oracle_source_rows}
    resolver_lookup = problem_lookup(
        oracle_source_rows,
        tier_field="difficulty_tier",
        problem_field="problem",
        source_field="source",
        problem_id_field="problem_id",
    )
    resolver_counters: dict[tuple[str, tuple[str, str, str]], int] = defaultdict(int)

    worker_metrics = build_gemma_worker_metric_map(raw_root, resolver_lookup)

    problem_index_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    protocol_rows: list[dict[str, Any]] = []
    trace_manifest_rows: list[dict[str, Any]] = []
    problem_outcomes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    problem_meta: dict[str, dict[str, Any]] = {}
    seen_problem_protocol: set[tuple[str, str]] = set()

    for row in oracle_source_rows:
        legacy_problem_id = row["problem_id"]
        problem_id = public_id_by_legacy.get(legacy_problem_id, sanitize_path_token(legacy_problem_id))
        tier = tier_label(row.get("difficulty_tier", ""))
        group_dir = staging / "omnimath2" / "problems" / tier / sanitize_path_token(problem_id)
        problem_meta[problem_id] = {
            "experiment_id": experiment_id,
            "problem_id": problem_id,
            "legacy_problem_id": legacy_problem_id,
            "source_problem_id": problem_id,
            "benchmark_id": "omnimath2",
            "subset": row.get("source", ""),
            "tier": tier,
            "difficulty_tier": row.get("difficulty_tier", ""),
            "difficulty_score": "",
            "problem": row.get("problem", ""),
            "answer_available": "1",
        }
        problem_index_rows.append(problem_meta[problem_id])
        matched_rows.append(
            {
                "experiment_id": experiment_id,
                "problem_id": problem_id,
                "legacy_problem_id": legacy_problem_id,
                "split": "",
                "subset": row.get("source", ""),
                "difficulty_tier": row.get("difficulty_tier", ""),
                "baseline_correct": boolish(row.get("baseline_llm")),
                "single_correct": boolish(row.get("single_agent")),
                "per_correct": boolish(row.get("per")),
                "broadcast_correct": boolish(row.get("broadcast")),
                "baseline_tokens": "",
                "single_tokens": "",
                "per_tokens": "",
                "broadcast_tokens": "",
                "oracle_cheapest_successful": row.get("gemma_oracle", ""),
                "oracle_token_cost": "",
                "any_protocol_solved": "0" if row.get("gemma_oracle") == "none" else "1",
            }
        )
        oracle_rows.append(
            {
                "experiment_id": experiment_id,
                "problem_id": problem_id,
                "legacy_problem_id": legacy_problem_id,
                "oracle_cheapest_successful": row.get("gemma_oracle", ""),
                "oracle_token_cost": "",
                "oracle_solved": "0" if row.get("gemma_oracle") == "none" else "1",
                "oracle_label_scope": "single_realization_observed_matched_run_reduced_tier_sample",
            }
        )
        group_dir.mkdir(parents=True, exist_ok=True)
        write_json(group_dir / "problem.json", problem_meta[problem_id])

    for tier_dir in sorted(raw_root.glob("tier_*")):
        if not tier_dir.is_dir():
            continue
        tier = tier_label(tier_dir.name)
        for raw_protocol_name, protocol_id in GEMMA_RAW_PROTOCOL_DIRS.items():
            results_path = tier_dir / "chunk000" / raw_protocol_name / "aggregate" / "results.jsonl"
            if not results_path.exists():
                continue
            with results_path.open("r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    legacy_problem_id = resolve_problem_id(
                        obj,
                        tier,
                        protocol_id,
                        resolver_lookup,
                        resolver_counters,
                    )
                    if legacy_problem_id not in by_problem:
                        continue
                    problem_id = public_id_by_legacy.get(legacy_problem_id, sanitize_path_token(legacy_problem_id))
                    if (problem_id, protocol_id) in seen_problem_protocol:
                        continue
                    seen_problem_protocol.add((problem_id, protocol_id))
                    metric = worker_metrics.get((legacy_problem_id, protocol_id))
                    outcome = metric_summary(metric)
                    if not outcome["final_correct"]:
                        outcome["final_correct"] = boolish(
                            by_problem[legacy_problem_id].get(protocol_id, "")
                        )
                    outcome.update(
                        {
                            "legacy_problem_id": legacy_problem_id,
                            "final_answer": obj.get("final_answer", ""),
                            "answer": obj.get("answer") or obj.get("label") or "",
                            "answer_parse_status": "parsed",
                            "trace_path": str(protocol_trace_path(protocol_id, tier, problem_id, run_id)),
                        }
                    )
                    out_row = protocol_outcome_row(
                        experiment_id=experiment_id,
                        problem_id=problem_id,
                        protocol_id=protocol_id,
                        raw_protocol_name=raw_protocol_name,
                        actor_set_id="gemma-3-27b",
                        evaluator_id="gpt-oss-120b",
                        outcome=outcome,
                    )
                    protocol_rows.append(out_row)
                    problem_outcomes[problem_id].append(out_row)
                    trace_rel = Path(outcome["trace_path"])
                    trace_manifest_rows.append(
                        {
                            "experiment_id": experiment_id,
                            "problem_id": problem_id,
                            "legacy_problem_id": legacy_problem_id,
                            "protocol_id": protocol_id,
                            "trace_path": str(trace_rel),
                            "source_path": str(results_path.relative_to(repo_root)),
                            "source_line_number": line_idx + 1,
                            "redaction_status": "sanitized_public_staging",
                        }
                    )
                    write_json(
                        staging / trace_rel,
                        trace_payload(
                            experiment_id=experiment_id,
                            problem_id=problem_id,
                            protocol_id=protocol_id,
                            raw_protocol_name=raw_protocol_name,
                            actor_set_id="gemma-3-27b",
                            evaluator_id="gpt-oss-120b",
                            obj=obj,
                            outcome=outcome,
                            source_path=results_path,
                            repo_root=repo_root,
                        ),
                    )
                    write_csv(
                        staging / protocol_outcome_path(protocol_id, tier, problem_id, run_id),
                        [out_row],
                        PROTOCOL_OUTCOME_FIELDS,
                    )

    write_experiment_common(
        exp_dir=exp_dir,
        experiment_id=experiment_id,
        title="Omni-MATH 2 reduced matched four-protocol run with Gemma-3-27B actors",
        benchmark_id="omnimath2",
        slice_id="tier-sampled-833",
        actor_set_id="gemma-3-27b",
        evaluator_id="gpt-oss-120b",
        n_problems=len(oracle_source_rows),
        grouping="omnimath2/<protocol>/<tier>/<official_omnimath2_id>/<run_id>.trace.json",
        source_description="Reduced tier-sampled Omni-MATH 2 actor-family scope check.",
        notes="Gemma-3-27B actors are judged by the same gpt-oss-120b evaluator family. Temperature is 0 in configs.",
    )
    copy_configs(
        repo_root,
        exp_dir,
        pick_config_files(
            repo_root,
            raw_root,
            {
                "baseline_llm": "tier_01/chunk000/baseline_gemma3_27b_eval_oss120b/worker_00/config.yaml",
                "single_agent": "tier_01/chunk000/single_agent_hint_gemma3_27b_eval_oss120b/worker_00/config.yaml",
                "per": "tier_01/chunk000/per_hint_gemma3_27b_eval_oss120b/worker_00/config.yaml",
                "broadcast": "tier_01/chunk000/broadcast_hint_gemma3_27b_eval_oss120b/worker_00/config.yaml",
            },
        ),
    )
    write_tables(
        tables_dir,
        problem_index_rows,
        protocol_rows,
        matched_rows,
        aggregate_metrics(protocol_rows, experiment_id),
        oracle_rows,
        trace_manifest_rows,
    )
    return experiment_summary(
        experiment_id,
        "omnimath2",
        "tier-sampled-833",
        len(oracle_source_rows),
        "gemma-3-27b",
        "gpt-oss-120b",
        run_date=GEMMA_RUN_DATE,
    )


def build_gemma_worker_metric_map(
    raw_root: Path,
    resolver_lookup: dict[str, dict[tuple[str, str, str], list[str]]],
) -> dict[tuple[str, str], dict[str, Any]]:
    metrics: dict[tuple[str, str], dict[str, Any]] = {}
    resolver_counters: dict[tuple[str, tuple[str, str, str]], int] = defaultdict(int)
    for raw_protocol_name, protocol_id in GEMMA_RAW_PROTOCOL_DIRS.items():
        for results_path in sorted(raw_root.glob(f"tier_*/chunk000/{raw_protocol_name}/worker_*/results.jsonl")):
            metric_path = first_metric_file(results_path.parent)
            metric_rows = read_jsonl(metric_path) if metric_path else []
            with results_path.open("r", encoding="utf-8") as f:
                tier = tier_label(results_path.parts[-5])
                for idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    problem_id = resolve_problem_id(
                        obj,
                        tier,
                        protocol_id,
                        resolver_lookup,
                        resolver_counters,
                    )
                    if idx < len(metric_rows):
                        metrics[(problem_id, protocol_id)] = metric_rows[idx]
    return metrics


def oracle_token_cost(row: dict[str, str]) -> str:
    protocol = row.get("cheapest_successful_protocol", "")
    if protocol == "baseline_llm":
        return row.get("baseline_total_tokens", "")
    if protocol == "single_agent":
        return row.get("single_agent_total_tokens", "")
    if protocol == "per":
        return row.get("per_total_tokens", "")
    if protocol == "broadcast":
        return row.get("broadcast_total_tokens", "")
    return ""


def protocol_outcome_row(
    *,
    experiment_id: str,
    problem_id: str,
    protocol_id: str,
    raw_protocol_name: str,
    actor_set_id: str,
    evaluator_id: str,
    outcome: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "problem_id": problem_id,
        "legacy_problem_id": outcome.get("legacy_problem_id", ""),
        "protocol_id": protocol_id,
        "raw_protocol_name": raw_protocol_name,
        "actor_set_id": actor_set_id,
        "evaluator_id": evaluator_id,
        "temperature": 0,
        "final_correct": outcome.get("final_correct", ""),
        "first_pass_correct": outcome.get("first_pass_correct", ""),
        "final_answer": outcome.get("final_answer", ""),
        "answer_parse_status": outcome.get("answer_parse_status", ""),
        "total_tokens": outcome.get("total_tokens", ""),
        "prompt_tokens": outcome.get("prompt_tokens", ""),
        "completion_tokens": outcome.get("completion_tokens", ""),
        "model_calls": outcome.get("model_calls", ""),
        "evaluator_calls": outcome.get("evaluator_calls", ""),
        "wall_time_seconds": outcome.get("wall_time_seconds", ""),
        "trace_path": outcome.get("trace_path", ""),
        "error_status": "",
    }


def aggregate_metrics(protocol_rows: list[dict[str, Any]], experiment_id: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in protocol_rows:
        grouped[row["protocol_id"]].append(row)
    out: list[dict[str, Any]] = []
    for protocol_id, rows in grouped.items():
        n = len(rows)
        final = sum(1 for r in rows if str(r.get("final_correct")) == "1")
        first_values = [r for r in rows if str(r.get("first_pass_correct", "")) in {"0", "1"}]
        first = sum(1 for r in first_values if str(r.get("first_pass_correct")) == "1")
        tokens = [float(r["total_tokens"]) for r in rows if str(r.get("total_tokens", "")).strip()]
        calls = [float(r["model_calls"]) for r in rows if str(r.get("model_calls", "")).strip()]
        eval_calls = [float(r["evaluator_calls"]) for r in rows if str(r.get("evaluator_calls", "")).strip()]
        wall = [float(r["wall_time_seconds"]) for r in rows if str(r.get("wall_time_seconds", "")).strip()]
        out.append(
            {
                "experiment_id": experiment_id,
                "protocol_id": protocol_id,
                "n": n,
                "final_pass_rate": f"{final / n:.6f}" if n else "",
                "first_pass_rate": f"{first / len(first_values):.6f}" if first_values else "",
                "avg_tokens": f"{sum(tokens) / len(tokens):.3f}" if tokens else "",
                "median_tokens": f"{median(tokens):.3f}" if tokens else "",
                "avg_model_calls": f"{sum(calls) / len(calls):.3f}" if calls else "",
                "avg_evaluator_calls": f"{sum(eval_calls) / len(eval_calls):.3f}" if eval_calls else "",
                "avg_wall_time_seconds": f"{sum(wall) / len(wall):.3f}" if wall else "",
                "cost_per_success": f"{sum(tokens) / final:.3f}" if tokens and final else "",
            }
        )
    order = {p: i for i, (p, *_rest) in enumerate(PROTOCOLS)}
    return sorted(out, key=lambda r: order.get(r["protocol_id"], 99))


def median(values: list[float]) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def write_tables(
    tables_dir: Path,
    problem_index_rows: list[dict[str, Any]],
    protocol_rows: list[dict[str, Any]],
    matched_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
    trace_manifest_rows: list[dict[str, Any]],
) -> None:
    write_csv(
        tables_dir / "problem_index.csv",
        problem_index_rows,
        PROBLEM_INDEX_FIELDS,
    )
    write_csv(
        tables_dir / "protocol_outcomes.csv",
        protocol_rows,
        [
            "experiment_id",
            "problem_id",
            "legacy_problem_id",
            "protocol_id",
            "raw_protocol_name",
            "actor_set_id",
            "evaluator_id",
            "temperature",
            "final_correct",
            "first_pass_correct",
            "final_answer",
            "answer_parse_status",
            "total_tokens",
            "prompt_tokens",
            "completion_tokens",
            "model_calls",
            "evaluator_calls",
            "wall_time_seconds",
            "trace_path",
            "error_status",
        ],
    )
    write_csv(
        tables_dir / "per_problem_matched.csv",
        matched_rows,
        [
            "experiment_id",
            "problem_id",
            "legacy_problem_id",
            "split",
            "subset",
            "difficulty_tier",
            "baseline_correct",
            "single_correct",
            "per_correct",
            "broadcast_correct",
            "baseline_tokens",
            "single_tokens",
            "per_tokens",
            "broadcast_tokens",
            "oracle_cheapest_successful",
            "oracle_token_cost",
            "any_protocol_solved",
        ],
    )
    write_csv(
        tables_dir / "aggregate_metrics.csv",
        aggregate_rows,
        [
            "experiment_id",
            "protocol_id",
            "n",
            "final_pass_rate",
            "first_pass_rate",
            "avg_tokens",
            "median_tokens",
            "avg_model_calls",
            "avg_evaluator_calls",
            "avg_wall_time_seconds",
            "cost_per_success",
        ],
    )
    write_csv(
        tables_dir / "routing_oracle.csv",
        oracle_rows,
        [
            "experiment_id",
            "problem_id",
            "legacy_problem_id",
            "oracle_cheapest_successful",
            "oracle_token_cost",
            "oracle_solved",
            "oracle_label_scope",
        ],
    )
    write_csv(
        tables_dir / "trace_manifest.csv",
        trace_manifest_rows,
        [
            "experiment_id",
            "problem_id",
            "legacy_problem_id",
            "protocol_id",
            "trace_path",
            "source_path",
            "source_line_number",
            "redaction_status",
        ],
    )


def write_problem_outcomes(
    exp_dir: Path,
    grouping_dir_name: str,
    problem_outcomes: dict[str, list[dict[str, Any]]],
    problem_meta: dict[str, dict[str, Any]],
    _source_rows: dict[str, Any],
) -> None:
    order = {p: i for i, (p, *_rest) in enumerate(PROTOCOLS)}
    for problem_id, rows in problem_outcomes.items():
        meta = problem_meta[problem_id]
        tier = tier_label(meta.get("tier") or meta.get("difficulty_tier"))
        problem_dir = exp_dir / grouping_dir_name / tier / sanitize_path_token(problem_id)
        write_csv(
            problem_dir / "outcomes.csv",
            sorted(rows, key=lambda r: order.get(r["protocol_id"], 99)),
            [
                "experiment_id",
                "problem_id",
                "protocol_id",
                "raw_protocol_name",
                "actor_set_id",
                "evaluator_id",
                "temperature",
                "final_correct",
                "first_pass_correct",
                "final_answer",
                "answer_parse_status",
                "total_tokens",
                "prompt_tokens",
                "completion_tokens",
                "model_calls",
                "evaluator_calls",
                "wall_time_seconds",
                "trace_path",
                "error_status",
            ],
        )


def write_omnimath2_problems_jsonl(staging: Path, records: list[dict[str, Any]]) -> None:
    out_path = staging / "omnimath2" / "problems.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in sorted(records, key=lambda r: int(r["id"]) if str(r["id"]).isdigit() else str(r["id"])):
            f.write(json.dumps(sanitize_obj(row), ensure_ascii=False) + "\n")


def write_experiment_common(
    *,
    exp_dir: Path,
    experiment_id: str,
    title: str,
    benchmark_id: str,
    slice_id: str,
    actor_set_id: str,
    evaluator_id: str,
    n_problems: int,
    grouping: str,
    source_description: str,
    notes: str,
) -> None:
    metadata = {
        "schema_version": "mas-protocol-dataset-v2",
        "experiment_id": experiment_id,
        "title": title,
        "benchmark_id": benchmark_id,
        "slice_id": slice_id,
        "actor_set_id": actor_set_id,
        "evaluator_id": evaluator_id,
        "protocols": [p[0] for p in PROTOCOLS],
        "temperature": 0,
        "n_problems": n_problems,
        "grouping": grouping,
        "source_description": source_description,
        "notes": notes,
    }
    write_json(exp_dir / "metadata.json", metadata)
    readme = f"""# {title}

Run id:

```text
{experiment_id}
```

## Setting

- Benchmark: `{benchmark_id}`
- Slice: `{slice_id}`
- Actor set: `{actor_set_id}`
- Evaluator: `{evaluator_id}`
- Protocols: `baseline_llm`, `single_agent`, `per`, `broadcast`
- Temperature: `0`
- Problems: `{n_problems}`
- Public trace layout: `{grouping}`

## Files

- `tables/problem_index.csv`: one row per problem.
- `tables/protocol_outcomes.csv`: one row per problem and protocol.
- `tables/per_problem_matched.csv`: wide matched table for routing analysis.
- `tables/aggregate_metrics.csv`: protocol-level accuracy and cost summary.
- `tables/routing_oracle.csv`: cheapest successful protocol label where available.
- `tables/trace_manifest.csv`: maps readable trace paths back to original raw paths.
- `configs/`: shared run configuration files, stored once per protocol.
- Trace JSON files live under `omnimath2/<protocol>/<tier>/<official_id>/`.
  Chunk and worker folders are intentionally removed.

## Notes

{notes}
"""
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "README.md").write_text(readme, encoding="utf-8")


def copy_configs(repo_root: Path, exp_dir: Path, config_map: dict[str, Path]) -> None:
    for protocol_id, src in config_map.items():
        if src.exists():
            copy_sanitized_text(src, exp_dir / "configs" / f"{protocol_id}.config.yaml")


def experiment_summary(
    experiment_id: str,
    benchmark_id: str,
    slice_id: str,
    n: int,
    actor_set_id: str,
    evaluator_id: str,
    *,
    run_date: str,
) -> dict[str, Any]:
    run_id = {
        "gpt-oss-120b": "gpt_oss_120b",
        "gemma-3-27b": "gemma3_27b",
    }.get(actor_set_id, actor_set_id.replace("-", "_"))
    run_path = f"omnimath2/runs/{run_id}"
    return {
        "experiment_id": experiment_id,
        "benchmark_id": benchmark_id,
        "slice_id": slice_id,
        "experiment_path": run_path,
        "n_problems": n,
        "split_policy": "matched_observed_protocol_runs",
        "actor_set_id": actor_set_id,
        "evaluator_id": evaluator_id,
        "protocols": "baseline_llm|single_agent|per|broadcast",
        "run_date": run_date,
        "run_status": "staged",
        "privacy_status": "private_staging",
        "aggregate_metrics_path": f"{run_path}/tables/aggregate_metrics.csv",
        "per_problem_matched_path": f"{run_path}/tables/per_problem_matched.csv",
        "trace_manifest_path": f"{run_path}/tables/trace_manifest.csv",
        "notes": "Protocol traces browse under omnimath2/<protocol>/<tier>/<official_omnimath2_id>/; chunk/worker folders are removed.",
    }


def write_registry(staging: Path, experiments: list[dict[str, Any]]) -> None:
    registry = staging / "registry"
    write_csv(
        registry / "experiments.csv",
        experiments,
        [
            "experiment_id",
            "benchmark_id",
            "slice_id",
            "experiment_path",
            "n_problems",
            "split_policy",
            "actor_set_id",
            "evaluator_id",
            "protocols",
            "run_date",
            "run_status",
            "privacy_status",
            "aggregate_metrics_path",
            "per_problem_matched_path",
            "trace_manifest_path",
            "notes",
        ],
    )
    write_csv(
        registry / "protocols.csv",
        [
            {
                "protocol_id": p,
                "display_name": name,
                "order_index": order,
                "short_description": desc,
                "uses_multi_agent_deliberation": "1" if p in {"per", "broadcast"} else "0",
                "expected_cost_level": order,
            }
            for p, name, order, desc in PROTOCOLS
        ],
        [
            "protocol_id",
            "display_name",
            "order_index",
            "short_description",
            "uses_multi_agent_deliberation",
            "expected_cost_level",
        ],
    )
    write_csv(
        registry / "benchmarks.csv",
        [
            {
                "benchmark_id": "omnimath2",
                "source_dataset": "Omni-MATH 2",
                "source_url_or_citation": "Ballon et al., 2026 / Omni-MATH 2",
                "slice_id": "competition-math-4181",
                "filter_logic": "Competition-level math slice used by the routing paper.",
                "n_total": 4181,
                "language": "en",
                "task_type": "math_reasoning",
                "answer_type": "short_answer_or_expression",
                "license_notes": "Check upstream Omni-MATH 2 terms before public redistribution of problem text.",
            },
            {
                "benchmark_id": "omnimath2",
                "source_dataset": "Omni-MATH 2",
                "source_url_or_citation": "Ballon et al., 2026 / Omni-MATH 2",
                "slice_id": "tier-sampled-833",
                "filter_logic": "Reduced tier-sampled actor-family scope check.",
                "n_total": 833,
                "language": "en",
                "task_type": "math_reasoning",
                "answer_type": "short_answer_or_expression",
                "license_notes": "Check upstream Omni-MATH 2 terms before public redistribution of problem text.",
            },
        ],
        [
            "benchmark_id",
            "source_dataset",
            "source_url_or_citation",
            "slice_id",
            "filter_logic",
            "n_total",
            "language",
            "task_type",
            "answer_type",
            "license_notes",
        ],
    )
    write_csv(
        registry / "models.csv",
        [
            {
                "model_id": "gpt-oss-120b",
                "display_name": "gpt-oss-120b",
                "provider_or_family": "OpenAI open-weight OSS family",
                "role": "actor/evaluator",
                "open_or_closed": "open_weight",
                "version_notes": "Used as actor and evaluator in primary Omni-MATH run.",
                "license_or_terms_notes": "Check model provider terms.",
            },
            {
                "model_id": "gemma-3-27b",
                "display_name": "Gemma-3-27B",
                "provider_or_family": "Gemma",
                "role": "actor",
                "open_or_closed": "open_weight",
                "version_notes": "Used as actor family in reduced Omni-MATH scope check.",
                "license_or_terms_notes": "Check model provider terms.",
            },
        ],
        [
            "model_id",
            "display_name",
            "provider_or_family",
            "role",
            "open_or_closed",
            "version_notes",
            "license_or_terms_notes",
        ],
    )
    write_json(
        registry / "schema_version.json",
        {
            "schema_version": "mas-protocol-dataset-v2",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def write_root_docs(staging: Path) -> None:
    readme = """---
license: other
language:
  - en
task_categories:
  - text-generation
  - question-answering
pretty_name: Matched MAS Protocol Traces
size_categories:
  - 1K<n<10K
---

# Matched MAS Protocol Traces

This private staging dataset organizes expensive multi-agent protocol traces as
a reusable Hugging Face data resource. The layout mirrors the original
Omni-MATH-2 release at the problem-record level, then adds protocol traces.
Each run uses the same four protocol family:

- `baseline_llm`
- `single_agent`
- `per`
- `broadcast`

Unlike a raw trace dump, this release is organized around benchmark records,
per-protocol traces, matched outcome tables, aggregate metrics, configs, and
checksums.

## How To Read The Dataset

Start with:

```text
registry/experiments.csv
```

Then open the Omni-MATH-2 benchmark folder:

```text
omnimath2/problems.jsonl
omnimath2/tables/problem_index.csv
```

For run-specific tables:

```text
omnimath2/runs/gpt_oss_120b/tables/aggregate_metrics.csv
omnimath2/runs/gpt_oss_120b/tables/per_problem_matched.csv
omnimath2/runs/gemma3_27b/tables/aggregate_metrics.csv
```

For trace browsing, use:

```text
omnimath2/<protocol>/<tier>/<official_omnimath2_id>/<run_id>.trace.json
```

For example:

```text
omnimath2/baseline_llm/tier_01/2676/gpt_oss_120b.trace.json
omnimath2/single_agent/tier_01/2676/gpt_oss_120b.trace.json
omnimath2/per/tier_01/2676/gpt_oss_120b.trace.json
omnimath2/broadcast/tier_01/2676/gpt_oss_120b.trace.json
```

Chunk and worker folders from the execution system are intentionally removed
from the public browsing hierarchy. Original raw paths are preserved only in
`omnimath2/runs/<run_id>/tables/trace_manifest.csv`.

## Current Experiments

See `registry/experiments.csv` for the canonical list.

## Configuration

Run configuration files are stored once per experiment under:

```text
omnimath2/runs/<run_id>/configs/
```

All currently staged protocol runs use temperature `0`.

Problem folder names use the official Omni-MATH-2 `id` field. Because this
release uses a filtered 4,181-problem slice of the 4,428-row upstream dataset,
ids are not expected to be contiguous within each difficulty tier.

## Privacy

This is a private staging layout. Before public release, run an anonymization
scan for personal paths, usernames, tokens, private endpoints, and cache files.
"""
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "README.md").write_text(readme, encoding="utf-8")
    upload_guide = """# Upload Guide

This folder is a local Hugging Face dataset staging tree. Review the
anonymization report, checksums, and source-license notes before uploading.

Recommended upload command (set `HF_DATASET_REPO` to your own dataset repo):

```bash
hf auth whoami
hf upload-large-folder "$HF_DATASET_REPO" \\
  dataset_release/huggingface_v2 \\
  --type dataset
```

Before a real upload, regenerate full checksums:

```bash
python scripts/dataset_release/prepare_hf_protocol_dataset_v2.py --clean
```

The quick staging run may use `--skip-checksums`; do not use skipped checksums
for a public release.
"""
    (staging / "UPLOAD_GUIDE.md").write_text(upload_guide, encoding="utf-8")
    (staging / "docs").mkdir(exist_ok=True)
    (staging / "docs" / "anonymization_report.md").write_text(
        "# Anonymization Report\n\nStatus: private staging. Run the safety scan before public release.\n",
        encoding="utf-8",
    )
    (staging / "docs" / "schema.md").write_text(
        """# Schema

## Registry

- `registry/experiments.csv`: one row per experiment, with benchmark slice,
  actor set, evaluator, protocol list, and paths to the main tables.
- `registry/benchmarks.csv`: benchmark/source/filter information.
- `registry/models.csv`: model family and role information.
- `registry/protocols.csv`: canonical protocol IDs and descriptions.

## Benchmark And Run Tables

`omnimath2/problems.jsonl` mirrors the original Omni-MATH-2 row-style release
using the official `id` field where available.

Each run under `omnimath2/runs/<run_id>/` has:

- `tables/problem_index.csv`: one row per problem.
- `tables/protocol_outcomes.csv`: one row per problem and protocol.
- `tables/per_problem_matched.csv`: one wide row per problem for routing.
- `tables/aggregate_metrics.csv`: one row per protocol.
- `tables/routing_oracle.csv`: cheapest successful protocol labels when
  matched outcomes are available.
- `tables/trace_manifest.csv`: maps public clear trace paths back to original
  raw chunk/worker paths.

## Trace Browsing

Omni-MATH traces use:

```text
omnimath2/<protocol>/<tier>/<official_omnimath2_id>/<run_id>.trace.json
```

The problem folder also contains one small `<run_id>.outcome.csv` file for
that protocol/run.
""",
        encoding="utf-8",
    )
    (staging / "omnimath2").mkdir(parents=True, exist_ok=True)
    (staging / "omnimath2" / "README.md").write_text(
        """# Omni-MATH-2 Protocol Trace Release

This folder mirrors the original Omni-MATH-2 release style at the problem
record level, then adds matched traces for four protocol wrappers.

## Benchmark Records

- `problems.jsonl` contains the filtered Omni-MATH-2 problem records with the
  official upstream `id`, `problem`, `solution`, `answer`, `source`, `domain`,
  `difficulty`, and derived `difficulty_tier`.
- `tables/problem_index.csv` is the benchmark-level index used by the trace
  release. It keeps `problem_id` as the official Omni-MATH-2 id and stores the
  old internal routing id as `legacy_problem_id`.

The official ids come from the upstream Omni-MATH-2 `id` field. They are not
contiguous within a tier because this release is a filtered slice.

## Protocol Trace Layout

Traces are organized by protocol first:

```text
omnimath2/baseline_llm/<tier>/<official_id>/<run_id>.trace.json
omnimath2/single_agent/<tier>/<official_id>/<run_id>.trace.json
omnimath2/per/<tier>/<official_id>/<run_id>.trace.json
omnimath2/broadcast/<tier>/<official_id>/<run_id>.trace.json
```

Each trace has a companion `<run_id>.outcome.csv` with the correctness, token,
model-call, evaluator-call, wall-time, and trace-path fields for that
problem/protocol/run.

## Runs And Configurations

Run-level metadata, configs, aggregate metrics, matched per-problem outcomes,
oracle labels, and trace manifests live under:

```text
omnimath2/runs/gpt_oss_120b/
omnimath2/runs/gemma3_27b/
```

Current runs:

- `gpt_oss_120b`: full 4,181-problem Omni-MATH-2 filtered four-protocol run,
  with gpt-oss-120b as actor and evaluator.
- `gemma3_27b`: reduced 833-problem tier-sampled scope check, with
  Gemma-3-27B actors and gpt-oss-120b evaluator.

All staged protocol runs use temperature `0`. Chunk and worker folders from the
execution system are removed from public browsing paths; original raw paths are
kept only in `omnimath2/runs/<run_id>/tables/trace_manifest.csv`.
""",
        encoding="utf-8",
    )


def copy_release_docs(repo_root: Path, staging: Path) -> None:
    source_dir = repo_root / "results/emnlp_routing/dataset_release"
    for name in ["LICENSE", "NOTICE", "CITATION.cff"]:
        src = source_dir / name
        if src.exists():
            copy_sanitized_text(src, staging / name)


def write_release_manifest(staging: Path) -> None:
    files = [p for p in staging.rglob("*") if p.is_file() and not p.name == "checksums.sha256"]
    total_bytes = sum(p.stat().st_size for p in files)
    write_json(
        staging / "registry" / "release_manifest.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "schema_version": "mas-protocol-dataset-v2",
            "files": len(files),
            "size_bytes": total_bytes,
            "privacy_status": "private_staging",
        },
    )


def write_checksums(staging: Path, skip: bool) -> None:
    checksum_path = staging / "registry" / "checksums.sha256"
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    if skip:
        checksum_path.write_text(
            "# Checksums skipped for quick local staging. Regenerate before upload.\n",
            encoding="utf-8",
        )
        return
    with checksum_path.open("w", encoding="utf-8") as f:
        for path in sorted(p for p in staging.rglob("*") if p.is_file() and p != checksum_path):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            f.write(f"{digest}  {path.relative_to(staging)}\n")


def run_safety_scan(staging: Path) -> list[str]:
    """Scan the staging tree for local paths, usernames, and credential shapes.

    Set SCIAGENTTRACE_SCAN_USERNAMES to a comma-separated list of local account
    names to also flag (e.g. "alice,bob"); nothing site-specific is hardcoded.
    """
    findings: list[str] = []
    patterns = [
        re.compile(r"/Users/"),
        re.compile(r"/home/"),
        re.compile(r"/lus/"),
        re.compile(r"/grand/"),
        re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"),
        re.compile(r"(?i)authorization\s*[:=]"),
        re.compile(r"(?i)api[_-]?key\s*[:=]"),
    ]
    for name in os.environ.get("SCIAGENTTRACE_SCAN_USERNAMES", "").split(","):
        name = name.strip()
        if name:
            patterns.append(re.compile(re.escape(name), re.IGNORECASE))
    for path in staging.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".png", ".pdf"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern in patterns:
            if pattern.search(text):
                findings.append(f"{path.relative_to(staging)} :: {pattern.pattern}")
                break
    report = staging / "docs" / "anonymization_report.md"
    if findings:
        report.write_text(
            "# Anonymization Report\n\nStatus: findings need review before public release.\n\n"
            + "\n".join(f"- `{x}`" for x in findings[:500])
            + ("\n" if len(findings) <= 500 else f"\n\nTruncated {len(findings) - 500} additional findings.\n"),
            encoding="utf-8",
        )
    else:
        report.write_text(
            "# Anonymization Report\n\nStatus: no obvious personal paths, tokens, or local usernames found by the staging scan.\n",
            encoding="utf-8",
        )
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("dataset_release/huggingface_v2"),
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip-checksums", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    staging = (repo_root / args.staging_dir).resolve()
    if args.clean and staging.exists():
        shutil.rmtree(staging)

    write_root_docs(staging)
    copy_release_docs(repo_root, staging)
    experiments = [
        build_gpt_experiment(repo_root, staging),
        build_gemma_experiment(repo_root, staging),
    ]
    write_registry(staging, experiments)
    write_release_manifest(staging)
    findings = run_safety_scan(staging)
    write_checksums(staging, skip=args.skip_checksums)
    print(f"Prepared HF protocol dataset staging at {staging}")
    print(f"Experiments: {len(experiments)}")
    print(f"Safety scan findings: {len(findings)}")
    if findings:
        print("See docs/anonymization_report.md before public release.")


if __name__ == "__main__":
    main()
