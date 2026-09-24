#!/usr/bin/env python3
"""Merge targeted classic-protocol recovery rows into an exact canonical result set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_no}")
            rows.append(value)
    return rows


def normalize_key(value: Any) -> str:
    if value is None or value == "":
        raise ValueError("Result row is missing its merge key")
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        return str(int(text))
    return text


def target_keys(
    rows: list[dict[str, Any]], key: str, *, target_index_base: int = 0
) -> list[str]:
    keys: list[str] = []
    for index, row in enumerate(rows):
        value = row.get(key)
        if value in (None, "") and key == "question_id":
            value = index + target_index_base
        keys.append(normalize_key(value))
    if len(keys) != len(set(keys)):
        raise ValueError(f"Target dataset has duplicate {key} values")
    return keys


def load_unique_rows(path: Path, key: str) -> dict[str, dict[str, Any]]:
    keyed: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        row_key = normalize_key(row.get(key))
        if row_key in keyed:
            raise ValueError(f"Duplicate {key}={row_key!r} in {path}")
        keyed[row_key] = row
    return keyed


def recovery_sources(roots: Iterable[Path], explicit: Iterable[Path]) -> list[Path]:
    sources = {path.resolve() for path in explicit}
    for root in roots:
        sources.update(path.resolve() for path in root.rglob("results.jsonl"))
    return sorted(sources, key=str)


def resolve_artifact(row: dict[str, Any], *, source: Path, row_key: str) -> Path:
    artifact = row.get("task_run_artifact")
    if not artifact:
        raise ValueError(f"{source}: {row_key} has no task_run_artifact")
    artifact_path = Path(str(artifact))
    if not artifact_path.is_absolute():
        raw_results = row.get("_results_path")
        base_dir = Path(str(raw_results)).parent if raw_results else source.parent
        artifact_path = base_dir / artifact_path
    artifact_path = artifact_path.resolve()
    if not artifact_path.exists():
        raise ValueError(f"{source}: {row_key} artifact does not exist: {artifact_path}")
    return artifact_path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_recovery(
    *,
    base_results: Path,
    recovery_paths: list[Path],
    target_dataset: Path,
    output_root: Path,
    key: str = "question_id",
    target_index_base: int = 0,
    require_artifacts: bool = False,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    target_rows = read_jsonl(target_dataset)
    ordered_keys = target_keys(target_rows, key, target_index_base=target_index_base)
    target_set = set(ordered_keys)
    base = load_unique_rows(base_results, key)

    outside_target = sorted(set(base) - target_set)
    if outside_target:
        raise ValueError(f"Base results contain {len(outside_target)} keys outside the target set")

    merged = dict(base)
    provenance: dict[str, dict[str, Any]] = {
        row_key: {"key": row_key, "source_kind": "base", "source_path": str(base_results.resolve())}
        for row_key in base
    }
    ignored_existing: list[dict[str, str]] = []
    recovery_added = 0
    seen_recovery: dict[str, Path] = {}

    for source in recovery_paths:
        rows = load_unique_rows(source, key)
        for row_key, row in rows.items():
            if row_key not in target_set:
                raise ValueError(f"{source}: {key}={row_key!r} is outside the target set")
            if row_key in seen_recovery:
                raise ValueError(
                    f"Recovery key {row_key!r} appears in both {seen_recovery[row_key]} and {source}"
                )
            seen_recovery[row_key] = source
            if row_key in merged:
                ignored_existing.append({"key": row_key, "source_path": str(source)})
                continue
            merged[row_key] = row
            provenance[row_key] = {
                "key": row_key,
                "source_kind": "recovery",
                "source_path": str(source),
            }
            recovery_added += 1

    if require_artifacts:
        for row_key, row in merged.items():
            artifact_path = resolve_artifact(
                row,
                source=Path(provenance[row_key]["source_path"]),
                row_key=row_key,
            )
            provenance[row_key]["resolved_task_run_artifact"] = str(artifact_path)

    missing_keys = [row_key for row_key in ordered_keys if row_key not in merged]
    complete = not missing_keys and len(merged) == len(ordered_keys)
    if not complete and not allow_incomplete:
        raise ValueError(
            f"Canonical merge is incomplete: {len(merged)}/{len(ordered_keys)} rows; "
            f"missing {len(missing_keys)}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    ordered_rows = [merged[row_key] for row_key in ordered_keys if row_key in merged]
    ordered_provenance = [provenance[row_key] for row_key in ordered_keys if row_key in provenance]
    missing_target_rows = [target_rows[index] for index, row_key in enumerate(ordered_keys) if row_key in missing_keys]

    results_path = output_root / "results.jsonl"
    provenance_path = output_root / "provenance.jsonl"
    missing_path = output_root / "missing_target_rows.jsonl"
    audit_path = output_root / "audit.json"
    write_jsonl(results_path, ordered_rows)
    write_jsonl(provenance_path, ordered_provenance)
    write_jsonl(missing_path, missing_target_rows)

    audit = {
        "schema_version": "classic-recovery-merge-v1",
        "complete": complete,
        "key": key,
        "target_index_base": target_index_base,
        "target_records": len(ordered_keys),
        "base_records": len(base),
        "recovery_source_files": len(recovery_paths),
        "recovery_records_seen": len(seen_recovery),
        "recovery_records_added": recovery_added,
        "recovery_records_ignored_existing": len(ignored_existing),
        "canonical_records": len(ordered_rows),
        "missing_records": len(missing_keys),
        "missing_keys": missing_keys,
        "ignored_existing": ignored_existing,
        "base_results": str(base_results.resolve()),
        "target_dataset": str(target_dataset.resolve()),
        "recovery_sources": [str(path) for path in recovery_paths],
        "require_artifacts": require_artifacts,
    }
    write_json(audit_path, audit)

    checksum_paths = [results_path, provenance_path, missing_path, audit_path]
    (output_root / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, action="append", default=[])
    parser.add_argument("--recovery-results", type=Path, action="append", default=[])
    parser.add_argument("--target-dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--key", default="question_id")
    parser.add_argument(
        "--target-index-base",
        type=int,
        default=0,
        help="Index assigned when target rows lack question_id (science-QA runs use 1)",
    )
    parser.add_argument("--require-artifacts", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    sources = recovery_sources(args.recovery_root, args.recovery_results)
    if not sources:
        raise SystemExit("No recovery results.jsonl files were found")
    try:
        audit = merge_recovery(
            base_results=args.base_results,
            recovery_paths=sources,
            target_dataset=args.target_dataset,
            output_root=args.output_root,
            key=args.key,
            target_index_base=args.target_index_base,
            require_artifacts=args.require_artifacts,
            allow_incomplete=args.allow_incomplete,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        "[classic-recovery-merge] "
        f"canonical={audit['canonical_records']}/{audit['target_records']} "
        f"recovery_added={audit['recovery_records_added']} complete={audit['complete']}"
    )


if __name__ == "__main__":
    main()
