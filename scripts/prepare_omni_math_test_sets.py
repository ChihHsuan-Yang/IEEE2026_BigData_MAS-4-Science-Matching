#!/usr/bin/env python3
"""
Build the small Omni-MATH test sets used for benchmark verification.

Each output file contains exactly five problems so you can run quick PER smoke tests
without touching the full tier files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TEST_DIR = DATA_DIR / "omni-math" / "test"
NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


PRESETS: Dict[str, Dict[str, str]] = {
    "tier1_omni_math_rule_numeric": {
        "dataset_family": "omni_math_rule",
        "filename": "tier1_omni_math_rule_numeric_5.jsonl",
        "description": "Tier-1 omni-math-rule with the local numeric verifier.",
    },
    "tier1_omni_math_rule_verifier": {
        "dataset_family": "omni_math_rule",
        "filename": "tier1_omni_math_rule_verifier_5.jsonl",
        "description": "Tier-1 omni-math-rule with the official rule-based verifier.",
    },
    "tier1_omni_math_rule_omni_judge": {
        "dataset_family": "omni_math_rule",
        "filename": "tier1_omni_math_rule_omni_judge_5.jsonl",
        "description": "Tier-1 omni-math-rule with the official Omni-Judge evaluator.",
    },
    "tier1_omni_math_omni_judge": {
        "dataset_family": "omni_math",
        "filename": "tier1_omni_math_omni_judge_5.jsonl",
        "description": "Tier-1 Omni-MATH with the official Omni-Judge evaluator.",
    },
    "tier1_omni_math_omni_verifier": {
        "dataset_family": "omni_math",
        "filename": "tier1_omni_math_omni_verifier_5.jsonl",
        "description": "Tier-1 Omni-MATH with the official learned verifier alias.",
    },
    "tier1_omni_math_numeric": {
        "dataset_family": "omni_math",
        "filename": "tier1_omni_math_numeric_5.jsonl",
        "description": "Tier-1 Omni-MATH with the local numeric verifier.",
    },
    "tier2_omni_math_small": {
        "dataset_family": "omni_math",
        "filename": "tier2_omni_math_5.jsonl",
        "description": "Tier-2 Omni-MATH small five-problem dataset with plain numeric answers.",
    },
    "tier3_omni_math_small": {
        "dataset_family": "omni_math",
        "filename": "tier3_omni_math_5.jsonl",
        "description": "Tier-3 Omni-MATH small five-problem dataset with plain numeric answers.",
    },
    "tier6_omni_math_small": {
        "dataset_family": "omni_math",
        "filename": "tier6_omni_math_5.jsonl",
        "description": "Tier-6 Omni-MATH small five-problem dataset with plain numeric answers.",
    },
    "tier6_omni_math_symbolic_small": {
        "dataset_family": "omni_math",
        "filename": "tier6_omni_math_symbolic_5.jsonl",
        "description": "Tier-6 Omni-MATH small five-problem dataset with non-numeric symbolic answers.",
    },
}


def _read_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _select_numeric_rows(path: Path, count: int = 5) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in _read_jsonl(path):
        answer = str(row.get("answer_number", "")).strip()
        if NUMERIC_RE.match(answer):
            rows.append(row)
        if len(rows) == count:
            break

    if len(rows) < count:
        raise RuntimeError(f"Only found {len(rows)} numeric rows in {path}, expected {count}.")

    return rows


def _select_non_numeric_rows(path: Path, count: int = 5) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in _read_jsonl(path):
        answer = str(row.get("answer_number", "")).strip()
        if answer and not NUMERIC_RE.match(answer):
            rows.append(row)
        if len(rows) == count:
            break

    if len(rows) < count:
        raise RuntimeError(
            f"Only found {len(rows)} non-numeric rows in {path}, expected {count}."
        )

    return rows


def main() -> None:
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    selected = {
        "omni_math": _select_numeric_rows(DATA_DIR / "omni-math" / "tier_01.jsonl"),
        "omni_math_rule": _select_numeric_rows(DATA_DIR / "omni-math-rule" / "tier_01.jsonl"),
        "omni_math_tier2": _select_numeric_rows(DATA_DIR / "omni-math" / "tier_02.jsonl"),
        "omni_math_tier3": _select_numeric_rows(DATA_DIR / "omni-math" / "tier_03.jsonl"),
        "omni_math_tier6": _select_numeric_rows(DATA_DIR / "omni-math" / "tier_06.jsonl"),
        "omni_math_tier6_symbolic": _select_non_numeric_rows(
            DATA_DIR / "omni-math" / "tier_06.jsonl"
        ),
    }

    manifest = {
        "generated_by": str(Path(__file__).relative_to(ROOT)),
        "count_per_dataset": 5,
        "selection_policy": {
            "tier1": "first five Tier-1 rows with plain numeric answers",
            "tier2": "first five Tier-2 rows with plain numeric answers",
            "tier3": "first five Tier-3 rows with plain numeric answers",
            "tier6": "first five Tier-6 rows with plain numeric answers",
            "tier6_symbolic": "first five Tier-6 rows with non-numeric answers",
        },
        "presets": [],
    }

    for preset_name, spec in PRESETS.items():
        if preset_name == "tier2_omni_math_small":
            base_rows = selected["omni_math_tier2"]
        elif preset_name == "tier3_omni_math_small":
            base_rows = selected["omni_math_tier3"]
        elif preset_name == "tier6_omni_math_small":
            base_rows = selected["omni_math_tier6"]
        elif preset_name == "tier6_omni_math_symbolic_small":
            base_rows = selected["omni_math_tier6_symbolic"]
        else:
            base_rows = selected[spec["dataset_family"]]
        rows: List[Dict[str, object]] = []
        for idx, row in enumerate(base_rows, start=1):
            copied = dict(row)
            copied["test_case"] = preset_name
            copied["test_index"] = idx
            copied["test_split"] = "omni-math/test"
            rows.append(copied)

        output_path = TEST_DIR / spec["filename"]
        _write_jsonl(output_path, rows)

        manifest["presets"].append(
            {
                "name": preset_name,
                "file": str(output_path.relative_to(ROOT)),
                "dataset_family": spec["dataset_family"],
                "description": spec["description"],
                "questions": [row["question"] for row in rows],
            }
        )

    with (TEST_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {len(PRESETS)} test datasets to {TEST_DIR}")


if __name__ == "__main__":
    main()
