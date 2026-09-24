#!/usr/bin/env python3
"""Aggregate science-QA JSONL worker runs into reviewer-friendly summaries."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_DATASETS = ("jeebench", "scibench", "mascqa_internal")
DEFAULT_PROTOCOLS = (
    "baseline_llm",
    "single_agent_hint_llm",
    "per_hint_llm",
    "broadcast_hint_llm",
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Malformed JSON in {path}:{line_no}: {exc}") from exc
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())


def result_sources(protocol_root: Path) -> List[Path]:
    if not protocol_root.exists():
        return []
    return sorted(
        (
            child / "results.jsonl"
            for child in protocol_root.iterdir()
            if child.is_dir() and child.name != "worker_logs" and (child / "results.jsonl").exists()
        ),
        key=lambda path: path.parent.name,
    )


def row_key(row: Dict[str, Any]) -> str:
    # `question_id` is assigned by the AgentVerse run loader and is unique
    # within a dataset slice. Some upstream science benchmarks reuse public ids,
    # so using `id` first would incorrectly drop completed examples.
    for key in ("question_id", "id"):
        value = row.get(key)
        if value not in (None, ""):
            return f"{key}:{value}"
    return json.dumps(row.get("input", ""), sort_keys=True)


def row_sort_key(row: Dict[str, Any]) -> Tuple[str, int, str]:
    qid = row.get("question_id")
    try:
        qid_int = int(qid)
    except (TypeError, ValueError):
        qid_int = 10**9
    return str(row.get("id") or ""), qid_int, str(row.get("_worker") or "")


def final_answer(row: Dict[str, Any]) -> str:
    per_eval = row.get("per_evaluation")
    if isinstance(per_eval, dict):
        for key in ("judge_student_final_answer", "final_answer", "rule_prediction"):
            value = per_eval.get(key)
            if value not in (None, ""):
                return str(value).strip()
    for key in ("final_answer", "answer", "response"):
        value = row.get(key)
        if value not in (None, ""):
            text = str(value).strip()
            boxed = re.search(r"\\boxed\{([^{}]+)\}", text)
            if boxed:
                return boxed.group(1).strip()
            return text
    return ""


def correctness(row: Dict[str, Any]) -> int:
    per_eval = row.get("per_evaluation")
    if isinstance(per_eval, dict) and per_eval.get("correctness") is not None:
        return int(bool(per_eval.get("correctness")))
    if row.get("correctness") is not None:
        return int(bool(row.get("correctness")))
    return int(final_answer(row).strip() == str(row.get("label") or row.get("answer") or "").strip())


def log_stats(log_path: Optional[Path]) -> Dict[str, Any]:
    if log_path is None or not log_path.exists():
        return {
            "worker_done": False,
            "traceback_count": 0,
            "rate_limit_count": 0,
            "api_connection_error_count": 0,
            "timeout_mention_count": 0,
        }
    text = log_path.read_text(errors="ignore")
    return {
        "worker_done": "[worker-done]" in text,
        "traceback_count": text.count("Traceback") + text.count("[worker-error]"),
        "rate_limit_count": text.count("RateLimitError"),
        "api_connection_error_count": text.count("APIConnectionError"),
        "timeout_mention_count": text.lower().count("timeout"),
    }


def collect_rows(protocol_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    worker_manifest: List[Dict[str, Any]] = []
    keyed: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    duplicate_count = 0
    log_root = protocol_root / "worker_logs"

    for results_path in result_sources(protocol_root):
        rows = read_jsonl(results_path)
        worker = results_path.parent.name
        log_path = log_root / f"{worker}.log"
        stats = log_stats(log_path)
        worker_manifest.append(
            {
                "worker": worker,
                "results_path": str(results_path),
                "log_path": str(log_path) if log_path.exists() else "",
                "rows": len(rows),
                **stats,
            }
        )
        mtime = results_path.stat().st_mtime
        for row in rows:
            row = dict(row)
            row["_worker"] = worker
            row["_results_path"] = str(results_path)
            key = row_key(row)
            if key in keyed:
                duplicate_count += 1
            if key not in keyed or mtime >= keyed[key][0]:
                keyed[key] = (mtime, row)

    rows = sorted((payload for _, payload in keyed.values()), key=row_sort_key)
    return rows, sorted(worker_manifest, key=lambda item: item["worker"]), duplicate_count


def summarize(rows: List[Dict[str, Any]], *, dataset: str, protocol: str, expected_total: int, worker_manifest: List[Dict[str, Any]], duplicate_count: int) -> Dict[str, Any]:
    total = len(rows)
    correct = sum(correctness(row) for row in rows)
    return {
        "dataset": dataset,
        "protocol": protocol,
        "total": total,
        "expected_total": expected_total,
        "missing": max(expected_total - total, 0),
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "complete": total == expected_total,
        "duplicate_rows_removed": duplicate_count,
        "worker_logs": len(worker_manifest),
        "workers_done": sum(1 for row in worker_manifest if row.get("worker_done")),
        "traceback_count": sum(int(row.get("traceback_count") or 0) for row in worker_manifest),
        "rate_limit_count": sum(int(row.get("rate_limit_count") or 0) for row in worker_manifest),
        "api_connection_error_count": sum(int(row.get("api_connection_error_count") or 0) for row in worker_manifest),
        "timeout_mention_count": sum(int(row.get("timeout_mention_count") or 0) for row in worker_manifest),
    }


def prediction_row(row: Dict[str, Any], *, dataset: str, protocol: str) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "protocol": protocol,
        "id": row.get("id") or "",
        "question_id": row.get("question_id") or "",
        "source": row.get("source") or "",
        "subject": row.get("subject") or row.get("topic") or "",
        "question_type": row.get("question_type") or "",
        "prediction": final_answer(row),
        "label": row.get("label") or row.get("answer") or "",
        "correct": correctness(row),
        "worker": row.get("_worker") or "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/science-benchmarks"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--protocols", nargs="+", default=list(DEFAULT_PROTOCOLS))
    parser.add_argument("--slice-name", default="text_only")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    output_root = (args.output_root or (run_root / "analysis")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_summaries: List[Dict[str, Any]] = []
    completion_rows: List[Dict[str, Any]] = []

    for dataset in args.datasets:
        expected_total = count_jsonl(args.data_root / dataset / args.slice_name / "all.jsonl")
        for protocol in args.protocols:
            protocol_root = run_root / dataset / protocol
            rows, worker_manifest, duplicate_count = collect_rows(protocol_root)
            summary = summarize(
                rows,
                dataset=dataset,
                protocol=protocol,
                expected_total=expected_total,
                worker_manifest=worker_manifest,
                duplicate_count=duplicate_count,
            )
            dataset_summaries.append(summary)
            completion_rows.append(
                {
                    "dataset": dataset,
                    "protocol": protocol,
                    "protocol_root": str(protocol_root),
                    **summary,
                }
            )

            out = output_root / "by_dataset" / dataset / protocol
            write_jsonl(out / "results.jsonl", rows)
            write_jsonl(out / "incorrect_examples.jsonl", [row for row in rows if not correctness(row)])
            write_json(out / "accuracy_summary.json", summary)
            write_csv(out / "worker_manifest.csv", worker_manifest, list(worker_manifest[0].keys()) if worker_manifest else ["worker"])
            prediction_rows = [prediction_row(row, dataset=dataset, protocol=protocol) for row in rows]
            write_csv(
                out / "predictions.csv",
                prediction_rows,
                [
                    "dataset",
                    "protocol",
                    "id",
                    "question_id",
                    "source",
                    "subject",
                    "question_type",
                    "prediction",
                    "label",
                    "correct",
                    "worker",
                ],
            )

    write_csv(
        output_root / "protocol_dataset_summary.csv",
        dataset_summaries,
        [
            "dataset",
            "protocol",
            "total",
            "expected_total",
            "missing",
            "correct",
            "accuracy",
            "complete",
            "duplicate_rows_removed",
            "worker_logs",
            "workers_done",
            "traceback_count",
            "rate_limit_count",
            "api_connection_error_count",
            "timeout_mention_count",
        ],
    )
    write_csv(output_root / "completion_manifest.csv", completion_rows, list(completion_rows[0].keys()) if completion_rows else ["dataset"])

    by_protocol: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in dataset_summaries:
        by_protocol[row["protocol"]].append(row)
    protocol_overall: List[Dict[str, Any]] = []
    for protocol, rows in sorted(by_protocol.items()):
        total = sum(int(row["total"]) for row in rows)
        expected = sum(int(row["expected_total"]) for row in rows)
        correct = sum(int(row["correct"]) for row in rows)
        protocol_overall.append(
            {
                "protocol": protocol,
                "total": total,
                "expected_total": expected,
                "missing": max(expected - total, 0),
                "correct": correct,
                "accuracy": correct / total if total else 0.0,
                "complete": total == expected,
            }
        )
    write_csv(
        output_root / "protocol_overall_summary.csv",
        protocol_overall,
        ["protocol", "total", "expected_total", "missing", "correct", "accuracy", "complete"],
    )

    write_json(
        output_root / "run_summary.json",
        {
            "run_root": str(run_root),
            "data_root": str(args.data_root.resolve()),
            "slice_name": args.slice_name,
            "datasets": args.datasets,
            "protocols": args.protocols,
            "complete": all(row["total"] == row["expected_total"] for row in dataset_summaries),
            "protocol_overall": protocol_overall,
        },
    )
    (output_root / "README.md").write_text(
        "# Science-QA Run Analysis\n\n"
        "- `protocol_dataset_summary.csv`: one row per dataset/protocol.\n"
        "- `protocol_overall_summary.csv`: one row per protocol over all datasets.\n"
        "- `completion_manifest.csv`: completion and error/backoff audit.\n"
        "- `by_dataset/<dataset>/<protocol>/`: merged results, predictions, incorrect examples, and worker manifest.\n",
        encoding="utf-8",
    )

    incomplete = [row for row in dataset_summaries if row["total"] != row["expected_total"]]
    print(f"[science-qa-aggregate] wrote {output_root}")
    if incomplete:
        print("[science-qa-aggregate] incomplete runs:")
        for row in incomplete:
            print(f"  {row['dataset']}/{row['protocol']}: {row['total']}/{row['expected_total']}")
        if args.fail_on_incomplete:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
