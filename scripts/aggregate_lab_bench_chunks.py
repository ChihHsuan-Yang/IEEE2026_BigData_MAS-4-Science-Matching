#!/usr/bin/env python3
"""Aggregate chunked LAB-Bench runs into per-subset protocol folders.

Supported full-run layouts:

    <run_root>/<model_family>/<protocol>/chunk_000/results.jsonl
    <run_root>/<model_family>/<protocol>/chunk_001/results.jsonl
    ...

or one protocol job directory per protocol:

    <run_root>/<protocol_run>/<model_family>/<protocol>/chunk_000/results.jsonl
    ...

Local smoke/debug runs with ``<protocol>/results.jsonl`` or
``<protocol>/example_000/results.jsonl`` are also accepted.

The script writes a reviewer-friendly analysis layout:

    <output_root>/by_subset/<subset>/<protocol>/
      results.jsonl
      correctness.jsonl
      predictions.csv
      incorrect_examples.jsonl
      accuracy_summary.json
      subtask_metrics.csv
      chunk_manifest.csv
      README.md

It also writes run-level summary tables under ``<output_root>``. The raw
chunk directories are left untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PROTOCOLS = (
    "baseline_llm",
    "single_agent_hint_llm",
    "per_hint_llm",
    "broadcast_hint_llm",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def chunk_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(?:chunk|example)_(\d+)", path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def row_sort_key(row: dict[str, Any]) -> tuple[str, str, int, str, str, str]:
    subset = str(row.get("lab_bench_subset") or "")
    subtask = str(row.get("subtask") or "")
    try:
        qid = int(row.get("question_id") or 0)
    except (TypeError, ValueError):
        qid = 0
    return (
        subset,
        subtask,
        qid,
        str(row.get("id") or ""),
        str(row.get("_run_group") or ""),
        str(row.get("_chunk") or ""),
    )


def final_answer(row: dict[str, Any]) -> str:
    for key in ("final_answer", "answer", "response"):
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        boxed = re.search(r"\\boxed\{([A-Z])\}", text)
        if boxed:
            return boxed.group(1)
        if re.fullmatch(r"[A-Z]", text):
            return text
    return ""


def correctness(row: dict[str, Any]) -> int:
    per_eval = row.get("per_evaluation")
    if isinstance(per_eval, dict) and per_eval.get("correctness") is not None:
        return int(bool(per_eval.get("correctness")))
    if row.get("correctness") is not None:
        return int(bool(row.get("correctness")))
    return int(final_answer(row) == str(row.get("label") or row.get("answer") or "").strip())


def manifest_subset_counts(rows: list[dict[str, Any]]) -> str:
    subset_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        subset_counts[str(row.get("lab_bench_subset") or "UNKNOWN")] += 1
    return json.dumps(dict(sorted(subset_counts.items())), sort_keys=True)


def protocol_roots(run_root: Path, model_family: str, protocol: str) -> list[tuple[str, Path]]:
    """Find protocol result roots under both direct and per-job layouts."""
    candidates: list[tuple[str, Path]] = []
    direct = run_root / model_family / protocol
    if direct.exists():
        candidates.append((".", direct))
    for child in sorted((p for p in run_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        nested = child / model_family / protocol
        if nested.exists():
            candidates.append((child.name, nested))

    seen: set[Path] = set()
    unique: list[tuple[str, Path]] = []
    for run_group, root in candidates:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append((run_group, root))
    return unique


def result_sources(protocol_root: Path) -> list[tuple[str, str, Path]]:
    """Return (source_kind, unit_name, results_path) triples."""
    sources: list[tuple[str, str, Path]] = []
    direct_results = protocol_root / "results.jsonl"
    if direct_results.exists():
        sources.append(("direct", "direct", direct_results))
    child_sources = [
        child
        for child in protocol_root.iterdir()
        if child.is_dir() and (child / "results.jsonl").exists()
    ]
    for child in sorted(child_sources, key=chunk_sort_key):
        source_kind = "chunk" if child.name.startswith("chunk_") else "worker"
        sources.append((source_kind, child.name, child / "results.jsonl"))
    return sources


def collect_protocol_rows(run_root: Path, model_family: str, protocol: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    roots = protocol_roots(run_root, model_family, protocol)
    if not roots:
        raise SystemExit(
            "Missing protocol result directory. Looked for "
            f"{run_root / model_family / protocol} and {run_root}/*/{model_family}/{protocol}"
        )

    all_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for run_group, protocol_root in roots:
        sources = result_sources(protocol_root)
        if not sources:
            manifest_rows.append(
                {
                    "protocol": protocol,
                    "run_group": run_group,
                    "chunk": "",
                    "source_kind": "none",
                    "results_path": str(protocol_root),
                    "num_examples": 0,
                    "status": "no_results_jsonl",
                    "subset_counts": "{}",
                }
            )
            continue
        for source_kind, unit_name, results_path in sources:
            rows = read_jsonl(results_path)
            for row in rows:
                row.setdefault("_chunk", unit_name)
                row.setdefault("_source_kind", source_kind)
                row.setdefault("_run_group", run_group)
            all_rows.extend(rows)
            manifest_rows.append(
                {
                    "protocol": protocol,
                    "run_group": run_group,
                    "chunk": unit_name,
                    "source_kind": source_kind,
                    "results_path": str(results_path),
                    "num_examples": len(rows),
                    "status": "ok",
                    "subset_counts": manifest_subset_counts(rows),
                }
            )
    return sorted(all_rows, key=row_sort_key), manifest_rows


def summarize(rows: list[dict[str, Any]], *, protocol: str, subset: str | None = None) -> dict[str, Any]:
    total = len(rows)
    correct = sum(correctness(row) for row in rows)
    subtask_rows = []
    by_subtask: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subtask[str(row.get("subtask") or "UNKNOWN")].append(row)
    for subtask, sub_rows in sorted(by_subtask.items()):
        sub_correct = sum(correctness(row) for row in sub_rows)
        sub_total = len(sub_rows)
        subtask_rows.append(
            {
                "protocol": protocol,
                "subset": subset or "ALL",
                "subtask": subtask,
                "correct": sub_correct,
                "total": sub_total,
                "accuracy": sub_correct / sub_total if sub_total else "",
            }
        )
    return {
        "protocol": protocol,
        "subset": subset or "ALL",
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "subtasks": subtask_rows,
    }


def prediction_row(row: dict[str, Any], *, protocol: str) -> dict[str, Any]:
    per_eval = row.get("per_evaluation") if isinstance(row.get("per_evaluation"), dict) else {}
    return {
        "protocol": protocol,
        "id": row.get("id") or "",
        "question_id": row.get("question_id") or "",
        "subset": row.get("lab_bench_subset") or "",
        "subtask": row.get("subtask") or "",
        "prediction": final_answer(row),
        "label": row.get("label") or row.get("answer") or "",
        "correct": correctness(row),
        "run_group": row.get("_run_group") or "",
        "chunk": row.get("_chunk") or "",
        "source_kind": row.get("_source_kind") or "",
        "judge": per_eval.get("judge_device") or per_eval.get("mode") or "",
        "answer_text": row.get("answer_text") or "",
    }


def correctness_row(row: dict[str, Any], *, protocol: str) -> dict[str, Any]:
    return {
        "question_id": row.get("question_id"),
        "id": row.get("id"),
        "subset": row.get("lab_bench_subset"),
        "subtask": row.get("subtask"),
        "final_answer": final_answer(row),
        "ground_truth": row.get("label") or row.get("answer"),
        "correctness": correctness(row),
        "protocol": protocol,
        "run_group": row.get("_run_group"),
        "chunk": row.get("_chunk"),
        "source_kind": row.get("_source_kind"),
    }


def duplicate_question_rows(rows: list[dict[str, Any]], *, protocol: str, subset: str | None = None) -> list[dict[str, Any]]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row_id = str(row.get("id") or row.get("question_id") or "")
        if row_id:
            by_id[row_id].append(row)
    duplicates: list[dict[str, Any]] = []
    for row_id, duplicate_rows in sorted(by_id.items()):
        if len(duplicate_rows) <= 1:
            continue
        duplicates.append(
            {
                "protocol": protocol,
                "subset": subset or "ALL",
                "id": row_id,
                "count": len(duplicate_rows),
                "locations": ";".join(
                    f"{row.get('_run_group') or '.'}/{row.get('_chunk') or ''}" for row in duplicate_rows
                ),
            }
        )
    return duplicates


def write_subset_bundle(
    *,
    output_root: Path,
    protocol: str,
    subset: str,
    rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    subset_dir = output_root / "by_subset" / subset / protocol
    sorted_rows = sorted(rows, key=row_sort_key)
    summary = summarize(sorted_rows, protocol=protocol, subset=subset)
    predictions = [prediction_row(row, protocol=protocol) for row in sorted_rows]
    correctness_rows = [correctness_row(row, protocol=protocol) for row in sorted_rows]
    incorrect = [row for row in sorted_rows if not correctness(row)]

    subset_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(subset_dir / "results.jsonl", sorted_rows)
    write_jsonl(subset_dir / "correctness.jsonl", correctness_rows)
    write_jsonl(subset_dir / "incorrect_examples.jsonl", incorrect)
    write_json(subset_dir / "accuracy_summary.json", {k: v for k, v in summary.items() if k != "subtasks"})
    write_csv(
        subset_dir / "predictions.csv",
        predictions,
        [
            "protocol",
            "id",
            "question_id",
            "subset",
            "subtask",
            "prediction",
            "label",
            "correct",
            "run_group",
            "chunk",
            "source_kind",
            "judge",
            "answer_text",
        ],
    )
    write_csv(
        subset_dir / "subtask_metrics.csv",
        summary["subtasks"],
        ["protocol", "subset", "subtask", "correct", "total", "accuracy"],
    )
    subset_manifest = [
        row
        for row in manifest_rows
        if row.get("status") == "ok" and subset in json.loads(str(row.get("subset_counts") or "{}"))
    ]
    write_csv(
        subset_dir / "chunk_manifest.csv",
        subset_manifest,
        ["protocol", "run_group", "chunk", "source_kind", "results_path", "num_examples", "status", "subset_counts"],
    )
    duplicate_rows = duplicate_question_rows(sorted_rows, protocol=protocol, subset=subset)
    if duplicate_rows:
        write_csv(
            subset_dir / "duplicate_questions.csv",
            duplicate_rows,
            ["protocol", "subset", "id", "count", "locations"],
        )
    readme = [
        f"# LAB-Bench {subset} / {protocol}",
        "",
        f"- correct: {summary['correct']}",
        f"- total: {summary['total']}",
        f"- accuracy: {summary['accuracy']:.6f}",
        f"- source chunks: {len(subset_manifest)}",
        "",
        "Files:",
        "- `results.jsonl`: merged raw result rows for this subset and protocol.",
        "- `correctness.jsonl`: compact correctness rows.",
        "- `predictions.csv`: one row per example for spreadsheet inspection.",
        "- `incorrect_examples.jsonl`: raw rows where the evaluator marked the final answer incorrect.",
        "- `subtask_metrics.csv`: accuracy by LAB-Bench subtask.",
        "- `chunk_manifest.csv`: chunks contributing examples to this subset.",
        "- `duplicate_questions.csv`: written only if the same question appears more than once.",
    ]
    (subset_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return {k: v for k, v in summary.items() if k != "subtasks"}


def write_protocol_bundle(
    *,
    output_root: Path,
    protocol: str,
    rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    protocol_dir = output_root / "by_protocol" / protocol
    summary = summarize(rows, protocol=protocol)
    write_jsonl(protocol_dir / "results.jsonl", rows)
    write_jsonl(protocol_dir / "correctness.jsonl", [correctness_row(row, protocol=protocol) for row in rows])
    write_json(protocol_dir / "accuracy_summary.json", {k: v for k, v in summary.items() if k != "subtasks"})
    write_csv(
        protocol_dir / "predictions.csv",
        [prediction_row(row, protocol=protocol) for row in rows],
        [
            "protocol",
            "id",
            "question_id",
            "subset",
            "subtask",
            "prediction",
            "label",
            "correct",
            "run_group",
            "chunk",
            "source_kind",
            "judge",
            "answer_text",
        ],
    )
    write_csv(
        protocol_dir / "subtask_metrics.csv",
        summary["subtasks"],
        ["protocol", "subset", "subtask", "correct", "total", "accuracy"],
    )
    write_csv(
        protocol_dir / "chunk_manifest.csv",
        manifest_rows,
        ["protocol", "run_group", "chunk", "source_kind", "results_path", "num_examples", "status", "subset_counts"],
    )
    duplicate_rows = duplicate_question_rows(rows, protocol=protocol)
    if duplicate_rows:
        write_csv(
            protocol_dir / "duplicate_questions.csv",
            duplicate_rows,
            ["protocol", "subset", "id", "count", "locations"],
        )
    return {k: v for k, v in summary.items() if k != "subtasks"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Root containing either <model_family>/<protocol>/... or <protocol_run>/<model_family>/<protocol>/...",
    )
    parser.add_argument("--model-family", default="gpt_oss_120b")
    parser.add_argument("--protocols", default=",".join(DEFAULT_PROTOCOLS), help="Comma-separated protocol directory names.")
    parser.add_argument("--output-root", type=Path, default=None, help="Default: <run_root>/analysis")
    parser.add_argument("--expected-total", type=int, default=None, help="Warn if any protocol total differs from this count.")
    parser.add_argument("--fail-on-mismatch", action="store_true", help="Exit nonzero after writing outputs if expected-total mismatches.")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    output_root = (args.output_root or (run_root / "analysis")).resolve()
    protocols = tuple(part.strip() for part in args.protocols.split(",") if part.strip())

    output_root.mkdir(parents=True, exist_ok=True)
    all_subset_summaries: list[dict[str, Any]] = []
    all_protocol_summaries: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    coverage_warnings: list[dict[str, Any]] = []

    for protocol in protocols:
        rows, manifest_rows = collect_protocol_rows(run_root, args.model_family, protocol)
        protocol_summary = write_protocol_bundle(
            output_root=output_root,
            protocol=protocol,
            rows=rows,
            manifest_rows=manifest_rows,
        )
        all_protocol_summaries.append(protocol_summary)
        if args.expected_total is not None and protocol_summary["total"] != args.expected_total:
            coverage_warnings.append(
                {
                    "protocol": protocol,
                    "expected_total": args.expected_total,
                    "observed_total": protocol_summary["total"],
                    "delta": protocol_summary["total"] - args.expected_total,
                }
            )
        completion_rows.extend(manifest_rows)

        by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_subset[str(row.get("lab_bench_subset") or "UNKNOWN")].append(row)
        for subset, subset_rows in sorted(by_subset.items()):
            all_subset_summaries.append(
                write_subset_bundle(
                    output_root=output_root,
                    protocol=protocol,
                    subset=subset,
                    rows=subset_rows,
                    manifest_rows=manifest_rows,
                )
            )

    write_csv(
        output_root / "protocol_overall_summary.csv",
        all_protocol_summaries,
        ["protocol", "subset", "correct", "total", "accuracy"],
    )
    write_csv(
        output_root / "protocol_subset_summary.csv",
        all_subset_summaries,
        ["protocol", "subset", "correct", "total", "accuracy"],
    )
    write_csv(
        output_root / "completion_manifest.csv",
        completion_rows,
        ["protocol", "run_group", "chunk", "source_kind", "results_path", "num_examples", "status", "subset_counts"],
    )
    if coverage_warnings:
        write_csv(
            output_root / "coverage_warnings.csv",
            coverage_warnings,
            ["protocol", "expected_total", "observed_total", "delta"],
        )
    readme = [
        "# LAB-Bench Chunk Aggregation",
        "",
        f"- run_root: `{run_root}`",
        f"- model_family: `{args.model_family}`",
        f"- protocols: {', '.join(protocols)}",
        "",
        "Main tables:",
        "- `protocol_overall_summary.csv`: one row per protocol over the full slice.",
        "- `protocol_subset_summary.csv`: one row per protocol and LAB-Bench subset.",
        "- `completion_manifest.csv`: chunk-level input coverage and status.",
        "- `coverage_warnings.csv`: written only if `--expected-total` finds a protocol row-count mismatch.",
        "",
        "Folder layout:",
        "- `by_protocol/<protocol>/`: merged full-slice files for one protocol.",
        "- `by_subset/<subset>/<protocol>/`: subset-specific analysis bundle for one protocol.",
    ]
    (output_root / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(f"[lab-bench-aggregate] wrote {output_root}")
    for row in all_protocol_summaries:
        print(f"  {row['protocol']}: {row['correct']}/{row['total']} = {row['accuracy']:.4f}")
    for row in coverage_warnings:
        print(
            "[lab-bench-aggregate] WARNING "
            f"{row['protocol']} total={row['observed_total']} expected={row['expected_total']}"
        )
    if coverage_warnings and args.fail_on_mismatch:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
