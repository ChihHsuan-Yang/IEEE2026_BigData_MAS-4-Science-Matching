#!/usr/bin/env python3
"""Audit an in-flight classic-protocol recovery without calling a model."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scripts.dataset_release.merge_classic_recovery import normalize_key, target_keys
except ModuleNotFoundError:  # Direct execution adds this script's directory to sys.path.
    from merge_classic_recovery import normalize_key, target_keys


def load_target(path: Path, key: str, target_index_base: int) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object in {path}:{line_no}")
            rows.append(value)
    return rows, target_keys(rows, key, target_index_base=target_index_base)


def resolve_artifact(row: dict[str, Any], source: Path) -> Path | None:
    value = row.get("task_run_artifact")
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    raw_results = row.get("_results_path")
    base = Path(str(raw_results)).parent if raw_results else source.parent
    return (base / path).resolve()


def audit_recovery(
    *,
    recovery_root: Path,
    target_dataset: Path,
    key: str = "question_id",
    target_index_base: int = 0,
) -> dict[str, Any]:
    _, ordered_target_keys = load_target(target_dataset, key, target_index_base)
    target_set = set(ordered_target_keys)
    sources = sorted(recovery_root.rglob("results.jsonl"), key=str)

    rows_total = 0
    parse_errors: list[dict[str, Any]] = []
    missing_keys: list[dict[str, Any]] = []
    error_rows: list[str] = []
    missing_artifacts: list[str] = []
    keys: list[str] = []

    for source in sources:
        with source.open(encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                rows_total += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    parse_errors.append(
                        {"path": str(source), "line": line_no, "error": str(exc)}
                    )
                    continue
                if not isinstance(row, dict):
                    parse_errors.append(
                        {"path": str(source), "line": line_no, "error": "not_a_json_object"}
                    )
                    continue
                try:
                    row_key = normalize_key(row.get(key))
                except ValueError as exc:
                    missing_keys.append(
                        {"path": str(source), "line": line_no, "error": str(exc)}
                    )
                    continue
                keys.append(row_key)
                if row.get("error") or row.get("error_type"):
                    error_rows.append(row_key)
                artifact = resolve_artifact(row, source)
                if artifact is None or not artifact.exists():
                    missing_artifacts.append(row_key)

    counts = Counter(keys)
    duplicate_keys = sorted(key for key, count in counts.items() if count > 1)
    unique_keys = set(counts)
    outside_target = sorted(unique_keys - target_set)
    missing_target = [key_value for key_value in ordered_target_keys if key_value not in unique_keys]
    valid_unique = unique_keys - set(error_rows) - set(missing_artifacts) - set(outside_target)
    complete = not any(
        (
            parse_errors,
            missing_keys,
            error_rows,
            duplicate_keys,
            missing_artifacts,
            outside_target,
            missing_target,
        )
    ) and len(valid_unique) == len(ordered_target_keys)

    return {
        "schema_version": "classic-recovery-progress-v1",
        "complete": complete,
        "recovery_root": str(recovery_root.resolve()),
        "target_dataset": str(target_dataset.resolve()),
        "key": key,
        "target_index_base": target_index_base,
        "target_records": len(ordered_target_keys),
        "result_source_files": len(sources),
        "rows_total": rows_total,
        "unique_result_keys": len(unique_keys),
        "valid_unique_keys": len(valid_unique),
        "parse_error_count": len(parse_errors),
        "missing_key_count": len(missing_keys),
        "error_row_count": len(error_rows),
        "duplicate_key_count": len(duplicate_keys),
        "missing_artifact_count": len(missing_artifacts),
        "outside_target_count": len(outside_target),
        "missing_target_count": len(missing_target),
        "parse_errors": parse_errors,
        "missing_keys": missing_keys,
        "error_row_keys": sorted(set(error_rows)),
        "duplicate_keys": duplicate_keys,
        "missing_artifact_keys": sorted(set(missing_artifacts)),
        "outside_target_keys": outside_target,
        "missing_target_keys": missing_target,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--key", default="question_id")
    parser.add_argument("--target-index-base", type=int, default=0)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    report = audit_recovery(
        recovery_root=args.recovery_root,
        target_dataset=args.target_dataset,
        key=args.key,
        target_index_base=args.target_index_base,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
