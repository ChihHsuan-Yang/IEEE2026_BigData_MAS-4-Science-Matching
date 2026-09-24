#!/usr/bin/env python3
"""Stage one validated classic-protocol run for the v2 Hugging Face builder."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_DIRS = {
    "baseline_llm": "baseline_llm",
    "single_agent": "single_agent",
    "per": "planner_executor_reviewer",
    "broadcast": "broadcast",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_no}")
            rows.append(row)
    return rows


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json_compact(row) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def problem_signature(row: dict[str, Any]) -> tuple[str, str]:
    source_id = str(row.get("id") or row.get("problem_id") or "")
    text = str(row.get("input") or row.get("question") or row.get("problem") or "")
    if not source_id or not text:
        raise ValueError("Row is missing its source id or problem text")
    return source_id, hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_question_id(value: Any) -> str:
    if value in (None, ""):
        raise ValueError("Result row is missing question_id")
    text = str(value).strip()
    return str(int(text)) if text.lstrip("-").isdigit() else text


def final_answer(row: dict[str, Any]) -> str:
    evaluation = row.get("per_evaluation")
    if isinstance(evaluation, dict):
        for key in ("judge_student_final_answer", "final_answer", "rule_prediction"):
            if evaluation.get(key) not in (None, ""):
                return str(evaluation[key])
    for key in ("final_answer", "answer", "response"):
        if row.get(key) not in (None, ""):
            return str(row[key])
    return ""


def outcome_row(
    row: dict[str, Any], *, benchmark_id: str, actor_model_id: str
) -> dict[str, Any]:
    evaluation = row.get("per_evaluation")
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    correctness = evaluation.get("correctness", row.get("correctness"))
    if correctness not in (0, 1, False, True):
        raise ValueError(f"Missing binary correctness for question_id={row.get('question_id')}")
    return {
        "problem_uid": str(row.get("id") or row.get("problem_id") or ""),
        "benchmark_id": benchmark_id,
        "model_family": actor_model_id,
        "final_correct": int(bool(correctness)),
        "final_answer": final_answer(row),
        "eval_mode": evaluation.get("mode"),
        "subject": row.get("subject"),
        "question_type": row.get("question_type"),
        "source": row.get("source"),
    }


def stage_run(
    *,
    canonical_results: Path,
    canonical_provenance: Path,
    target_dataset: Path,
    source_manifest: Path,
    output_root: Path,
    benchmark_id: str,
    protocol_id: str,
    actor_model_id: str,
    evaluator_model_id: str,
    compute_site: str,
    inference_backend: str,
    source_run_uri: str,
    source_repo_commit: str,
    target_index_base: int = 0,
    base_source_repo_commit: str | None = None,
    recovery_source_repo_commit: str | None = None,
    base_inference_backend: str | None = None,
    recovery_inference_backend: str | None = None,
) -> dict[str, Any]:
    results = read_jsonl(canonical_results)
    provenance = read_jsonl(canonical_provenance)
    targets = read_jsonl(target_dataset)
    if not (len(results) == len(provenance) == len(targets)):
        raise ValueError(
            "Canonical result, provenance, and target counts differ: "
            f"{len(results)}, {len(provenance)}, {len(targets)}"
        )
    if protocol_id not in PROTOCOL_DIRS:
        raise ValueError(f"Unknown protocol_id: {protocol_id}")

    staged_records: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    trace_manifest: list[dict[str, Any]] = []
    for index, (result, source, target) in enumerate(zip(results, provenance, targets)):
        expected_qid = str(index + target_index_base)
        result_qid = normalize_question_id(result.get("question_id"))
        if result_qid != expected_qid or str(source.get("key")) != expected_qid:
            raise ValueError(
                f"Question-order mismatch at target index {index}: "
                f"result={result_qid}, provenance={source.get('key')}, expected={expected_qid}"
            )
        if problem_signature(result) != problem_signature(target):
            raise ValueError(f"Problem mismatch for question_id={result_qid}")
        if result.get("error"):
            raise ValueError(f"Operational error row is not releasable: question_id={result_qid}")

        artifact = Path(str(source.get("resolved_task_run_artifact") or ""))
        if not artifact.is_absolute() or not artifact.exists():
            raise ValueError(f"Missing resolved raw trace for question_id={result_qid}: {artifact}")
        trace_manifest.append(
            {
                "question_id": result_qid,
                "source_kind": source.get("source_kind"),
                "sha256": sha256_file(artifact),
                "bytes": artifact.stat().st_size,
            }
        )

        source_kind = str(source.get("source_kind") or "base")
        if source_kind == "recovery":
            row_source_commit = recovery_source_repo_commit or source_repo_commit
            row_backend = recovery_inference_backend or inference_backend
        else:
            row_source_commit = base_source_repo_commit or source_repo_commit
            row_backend = base_inference_backend or inference_backend

        staged = {
            key: value
            for key, value in result.items()
            if key not in {"_results_path", "_worker", "task_run_artifact"}
        }
        staged.update(
            {
                "cluster": compute_site,
                "backend": row_backend,
                "actor_model_id": actor_model_id,
                "evaluator_id": evaluator_model_id,
                "source_commit": row_source_commit,
                "source_run_uri": source_run_uri,
                "source_import_version": "canonical-classic-stage-v1",
                "source_trace_sha256": trace_manifest[-1]["sha256"],
            }
        )
        staged_records.append(staged)
        outcomes.append(
            outcome_row(
                staged,
                benchmark_id=benchmark_id,
                actor_model_id=actor_model_id,
            )
        )

    protocol_dir = PROTOCOL_DIRS[protocol_id]
    model_root = output_root / protocol_dir / benchmark_id / actor_model_id
    records_path = model_root / "traces/records.jsonl"
    outcomes_path = model_root / "labels/outcomes.jsonl"
    trace_manifest_path = model_root / "source_trace_manifest.jsonl"
    provenance_path = model_root / "run_provenance.json"
    write_jsonl(records_path, staged_records)
    write_jsonl(outcomes_path, outcomes)
    write_jsonl(trace_manifest_path, trace_manifest)

    frozen = {
        "schema_version": "canonical-classic-stage-v1",
        "imported_at_utc": utc_now(),
        "frozen": True,
        "source_complete": True,
        "compute_site": compute_site,
        "inference_backend": inference_backend,
        "source_run_uri": source_run_uri,
        "source_repo_commit": source_repo_commit,
        "source_repo_commits": sorted(
            {
                base_source_repo_commit or source_repo_commit,
                recovery_source_repo_commit or source_repo_commit,
            }
        ),
        "inference_backends": sorted(
            {
                base_inference_backend or inference_backend,
                recovery_inference_backend or inference_backend,
            }
        ),
        "source_run_summary_sha256": sha256_file(source_manifest),
        "benchmark_id": benchmark_id,
        "protocol": protocol_id,
        "actor_model_id": actor_model_id,
        "evaluator_model_id": evaluator_model_id,
        "canonical_rows": len(staged_records),
        "target_index_base": target_index_base,
        "records_sha256": sha256_file(records_path),
        "outcomes_sha256": sha256_file(outcomes_path),
        "source_trace_manifest_sha256": sha256_file(trace_manifest_path),
        "canonical_results_sha256": sha256_file(canonical_results),
        "canonical_provenance_sha256": sha256_file(canonical_provenance),
        "target_dataset_sha256": sha256_file(target_dataset),
    }
    write_json(provenance_path, frozen)
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-results", type=Path, required=True)
    parser.add_argument("--canonical-provenance", type=Path, required=True)
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--protocol-id", choices=sorted(PROTOCOL_DIRS), required=True)
    parser.add_argument("--actor-model-id", required=True)
    parser.add_argument("--evaluator-model-id", required=True)
    parser.add_argument("--compute-site", required=True)
    parser.add_argument("--inference-backend", required=True)
    parser.add_argument("--source-run-uri", required=True)
    parser.add_argument("--source-repo-commit", required=True)
    parser.add_argument("--target-index-base", type=int, default=0)
    parser.add_argument("--base-source-repo-commit")
    parser.add_argument("--recovery-source-repo-commit")
    parser.add_argument("--base-inference-backend")
    parser.add_argument("--recovery-inference-backend")
    args = parser.parse_args()

    try:
        report = stage_run(**vars(args))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        "[canonical-classic-stage] "
        f"{report['benchmark_id']}/{report['protocol']} "
        f"{report['actor_model_id']} rows={report['canonical_rows']} frozen=True"
    )


if __name__ == "__main__":
    main()
