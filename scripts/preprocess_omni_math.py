#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def difficulty_to_tier(value: float) -> int:
    """
    Omni-MATH advertises 10 difficulty levels, while the released file stores
    averaged floating-point scores. We map each example to the nearest tier.
    """
    return max(1, min(10, int(round(float(value)))))


def normalize_record(record: Dict[str, Any], dataset_name: str, judge_type: str) -> Dict[str, Any]:
    raw_difficulty = float(record["difficulty"])
    return {
        "question": record["problem"],
        "answer": None,
        "answer_number": str(record["answer"]).strip(),
        "equation_solution": record.get("solution"),
        "difficulty": raw_difficulty,
        "difficulty_tier": difficulty_to_tier(raw_difficulty),
        "raw_difficulty": raw_difficulty,
        "domain": record.get("domain", []),
        "source": record.get("source", ""),
        "dataset_name": dataset_name,
        "judge_type": judge_type,
    }


def normalize_omni_math_2_record(
    record: Dict[str, Any],
    dataset_name: str = "omni-math-2-filtered",
    judge_type: str = "omni-judge",
) -> Optional[Dict[str, Any]]:
    if record.get("tags"):
        return None

    raw_difficulty = float(record["difficulty"])
    return {
        "question": record["problem"],
        "answer": None,
        "answer_number": str(record["answer"]).strip(),
        "equation_solution": record.get("solution"),
        "difficulty": raw_difficulty,
        "difficulty_tier": difficulty_to_tier(raw_difficulty),
        "raw_difficulty": raw_difficulty,
        "domain": record.get("domain", []),
        "source": record.get("source", ""),
        "dataset_name": dataset_name,
        "judge_type": judge_type,
    }


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_bundle(records: List[Dict[str, Any]], raw_path: Path, out_dir: Path, summary_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(raw_path, raw_dir / raw_path.name)

    all_path = out_dir / "all.jsonl"
    write_jsonl(all_path, records)

    by_tier: Dict[int, List[Dict[str, Any]]] = {tier: [] for tier in range(1, 11)}
    raw_counter = Counter()
    for record in records:
        by_tier[int(record["difficulty_tier"])].append(record)
        raw_counter[str(record["raw_difficulty"])] += 1

    for tier in range(1, 11):
        write_jsonl(out_dir / f"tier_{tier:02d}.jsonl", by_tier[tier])

    summary = {
        "total_examples": len(records),
        "counts_by_tier": {str(tier): len(by_tier[tier]) for tier in range(1, 11)},
        "counts_by_raw_difficulty": dict(sorted(raw_counter.items(), key=lambda kv: float(kv[0]))),
        "files": {
            "all": str(all_path),
            "tiers": {str(tier): str(out_dir / f"tier_{tier:02d}.jsonl") for tier in range(1, 11)},
        },
    }
    (out_dir / summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Omni-MATH into MGSM-style tiered JSONL files.")
    parser.add_argument("--omni-math-in", help="Path to the official Omni-MATH JSONL file.")
    parser.add_argument("--omni-math-rule-in", help="Path to the official omni-math-rule JSONL file.")
    parser.add_argument(
        "--omni-math-2-in",
        help="Path to the Omni-MATH-2 JSONL file from Hugging Face.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="AgentVerse data directory.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not any((args.omni_math_in, args.omni_math_rule_in, args.omni_math_2_in)):
        parser.error("Provide at least one input dataset path.")

    if args.omni_math_in:
        omni_math_in = Path(args.omni_math_in).resolve()
        omni_math_rows = [
            normalize_record(row, "omni-math", "omni-judge")
            for row in read_jsonl(omni_math_in)
        ]
        write_bundle(
            omni_math_rows,
            raw_path=omni_math_in,
            out_dir=data_dir / "omni-math",
            summary_name="summary.json",
        )

    if args.omni_math_rule_in:
        omni_math_rule_in = Path(args.omni_math_rule_in).resolve()
        omni_math_rule_rows = [
            normalize_record(row, "omni-math-rule", "rule")
            for row in read_jsonl(omni_math_rule_in)
        ]
        write_bundle(
            omni_math_rule_rows,
            raw_path=omni_math_rule_in,
            out_dir=data_dir / "omni-math-rule",
            summary_name="summary.json",
        )

    if args.omni_math_2_in:
        omni_math_2_in = Path(args.omni_math_2_in).resolve()
        omni_math_2_rows = []
        for row in read_jsonl(omni_math_2_in):
            normalized = normalize_omni_math_2_record(row)
            if normalized is not None:
                omni_math_2_rows.append(normalized)

        write_bundle(
            omni_math_2_rows,
            raw_path=omni_math_2_in,
            out_dir=data_dir / "omni-math-2-filtered",
            summary_name="summary.json",
        )


if __name__ == "__main__":
    main()
