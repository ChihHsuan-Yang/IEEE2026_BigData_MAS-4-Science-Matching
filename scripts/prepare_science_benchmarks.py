#!/usr/bin/env python3
"""Prepare JEEBench, SciBench, and MaScQA for AgentVerse JSONL runs.

The output schema is intentionally compatible with the generic JSONL loader:
each row has at least `input` and `answer`.  Additional metadata is preserved so
trace releases can cite source benchmark, license, and answer type.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = ROOT / "data" / "source_benchmarks"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "science-benchmarks"
SOURCE_REPOS = {
    "jeebench": "https://github.com/dair-iitd/jeebench.git",
    "scibench": "https://github.com/mandyyyyii/scibench.git",
    "mascqa_internal": "https://github.com/M3RG-IITD/MaScQA.git",
}
SOURCE_DIRS = {
    "jeebench": "jeebench",
    "scibench": "scibench",
    "mascqa_internal": "mascqa",
}

MISSING_ARTIFACT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bas shown in (?:the )?figure\b",
        r"\bshown in (?:the )?figure\b",
        r"\bgiven in (?:the )?figure\b",
        r"\bshown below\b",
        r"\bsee figure\b",
        r"\bfollowing figure\b",
        r"\bdiagram shown\b",
        r"\bin the image\b",
        r"\bfrom the image\b",
        r"\bfigure\s*\d",
        r"\bfig\.\s*\d",
        r"\bdata tables\b",
    ]
]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def fetch_sources(source_root: Path, datasets: list[str]) -> None:
    source_root.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        repo_url = SOURCE_REPOS[dataset]
        target = source_root / SOURCE_DIRS[dataset]
        if target.exists():
            continue
        print(f"[prepare] cloning {repo_url} -> {target}")
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(target)], check=True)


def chunk_rows(rows: list[dict[str, Any]], out_dir: Path, *, chunk_size: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("chunk_*.jsonl"):
        old.unlink()
    for idx in range(0, len(rows), chunk_size):
        write_jsonl(out_dir / f"chunk_{idx // chunk_size:03d}.jsonl", rows[idx : idx + chunk_size])


def missing_artifact_reasons(text: str) -> list[str]:
    reasons: list[str] = []
    for pattern in MISSING_ARTIFACT_PATTERNS:
        match = pattern.search(text)
        if match:
            reasons.append(match.group(0))
    return sorted(set(reasons))


def annotate_text_only(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    text_only_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    for row in rows:
        reasons = missing_artifact_reasons(row.get("input", ""))
        row["llm_only_text_suitable"] = not reasons
        row["text_only_exclusion_reasons"] = reasons
        if reasons:
            excluded_rows.append(row)
        else:
            text_only_rows.append(row)
    return text_only_rows, excluded_rows


def write_dataset_outputs(out: Path, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    text_only_rows, excluded_rows = annotate_text_only(rows)
    write_jsonl(out / "all.jsonl", rows)
    write_jsonl(out / "smoke.jsonl", rows[:5])
    chunk_rows(rows, out / "chunks_25", chunk_size=25)

    text_only_out = out / "text_only"
    write_jsonl(text_only_out / "all.jsonl", text_only_rows)
    write_jsonl(text_only_out / "smoke.jsonl", text_only_rows[:5])
    chunk_rows(text_only_rows, text_only_out / "chunks_25", chunk_size=25)
    write_jsonl(text_only_out / "excluded_missing_artifact.jsonl", excluded_rows)
    return text_only_rows, excluded_rows


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = clean_text(value)
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def format_input(*, benchmark: str, instructions: str, problem: str, metadata: dict[str, Any]) -> str:
    meta_lines = [f"{key}: {value}" for key, value in metadata.items() if value not in (None, "")]
    meta_block = "\n".join(meta_lines)
    return clean_text(
        f"{benchmark} scientific reasoning task for an LLM-only agent.\n\n"
        f"{instructions}\n\n"
        f"{meta_block}\n\n"
        f"Problem:\n{problem}"
    )


def prepare_jeebench(source_root: Path, output_root: Path) -> list[dict[str, Any]]:
    zip_path = source_root / "jeebench" / "data.zip"
    with zipfile.ZipFile(zip_path) as archive:
        raw = json.loads(archive.read("data/dataset.json"))

    rows: list[dict[str, Any]] = []
    for item in raw:
        answer = normalize_answer(item.get("gold"))
        qtype = clean_text(item.get("type"))
        instructions = (
            "Answer the JEE problem. If the question is multiple-choice, output the correct "
            "option letter(s) only, e.g. A or ABD. If the question requests a numeric value, "
            "output the requested value. End with a boxed final answer."
        )
        row = {
            "id": f"jeebench:{item.get('description')}:{item.get('index')}",
            "dataset_name": "jeebench",
            "source": "JEEBench",
            "license": "MIT",
            "release_compatibility": "cc-by-sa-compatible",
            "subject": item.get("subject"),
            "question_type": qtype,
            "problem_index": item.get("index"),
            "input": format_input(
                benchmark="JEEBench",
                instructions=instructions,
                problem=clean_text(item.get("question")),
                metadata={
                    "Exam": item.get("description"),
                    "Subject": item.get("subject"),
                    "Question type": qtype,
                },
            ),
            "answer": answer,
            "answer_number": answer,
            "reference_solution": "",
        }
        rows.append(row)

    out = output_root / "jeebench"
    write_dataset_outputs(out, rows)
    return rows


def _load_scibench_solution_map(dataset_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    mapping: dict[tuple[str, str], dict[str, Any]] = {}
    for sol_path in dataset_dir.glob("*_sol.json"):
        for item in json.loads(sol_path.read_text(encoding="utf-8")):
            key = (clean_text(item.get("source")), clean_text(item.get("problemid")))
            mapping[key] = item
    return mapping


def prepare_scibench(source_root: Path, output_root: Path) -> list[dict[str, Any]]:
    dataset_dir = source_root / "scibench" / "dataset" / "original"
    solution_map = _load_scibench_solution_map(dataset_dir)
    rows: list[dict[str, Any]] = []

    for path in sorted(dataset_dir.glob("*.json")):
        if path.name.endswith("_sol.json"):
            continue
        items = json.loads(path.read_text(encoding="utf-8"))
        for item in items:
            source = clean_text(item.get("source") or path.stem)
            problem_id = clean_text(item.get("problemid"))
            solution = solution_map.get((source, problem_id), {})
            answer_number = normalize_answer(item.get("answer_number"))
            unit = clean_text(item.get("unit"))
            answer = clean_text(f"{answer_number} {unit}").strip()
            instructions = (
                "Solve the college-level scientific problem. Output the requested numeric, "
                "symbolic, or short textual answer. Include units when the problem or reference "
                "answer requires units. End with a boxed final answer."
            )
            rows.append(
                {
                    "id": f"scibench:{source}:{problem_id}",
                    "dataset_name": "scibench",
                    "source": source,
                    "license": "MIT",
                    "release_compatibility": "cc-by-sa-compatible",
                    "question_type": "numeric_or_short_answer",
                    "problem_id": problem_id,
                    "unit": unit,
                    "input": format_input(
                        benchmark="SciBench",
                        instructions=instructions,
                        problem=clean_text(item.get("problem_text")),
                        metadata={
                            "Source textbook": source,
                            "Problem id": problem_id,
                        },
                    ),
                    "answer": answer,
                    "answer_number": answer_number,
                    "reference_solution": clean_text(solution.get("solution")),
                }
            )

    out = output_root / "scibench"
    write_dataset_outputs(out, rows)
    return rows


def _parse_question_cell(value: Any) -> str:
    text = clean_text(value)
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return clean_text("\n".join(str(x) for x in parsed if str(x).strip()))
        except Exception:
            return text
    return text


def _xlsx_cell_text(cell: ET.Element, shared_strings: list[str], ns: dict[str, str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return clean_text("".join(t.text or "" for t in cell.findall(".//main:t", ns)))

    value = cell.find("main:v", ns)
    if value is None or value.text is None:
        return ""

    raw = value.text
    if cell_type == "s":
        try:
            return shared_strings[int(raw)]
        except (ValueError, IndexError):
            return raw
    return raw


def _read_xlsx_first_sheet(path: Path) -> list[dict[str, str]]:
    """Read simple XLSX tables without openpyxl.

    MaScQA stores the needed QA table as a regular first-sheet workbook. This
    fallback avoids adding a runtime dependency to the smoke-test environment.
    """

    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("main:si", ns):
                shared_strings.append(clean_text("".join(t.text or "" for t in item.findall(".//main:t", ns))))

        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        table: list[list[str]] = []
        for row in sheet.findall(".//main:sheetData/main:row", ns):
            values: list[str] = []
            current_col = 0
            for cell in row.findall("main:c", ns):
                ref = cell.attrib.get("r", "")
                letters = "".join(ch for ch in ref if ch.isalpha())
                if letters:
                    col_idx = 0
                    for ch in letters:
                        col_idx = col_idx * 26 + (ord(ch.upper()) - ord("A") + 1)
                    while current_col < col_idx - 1:
                        values.append("")
                        current_col += 1
                values.append(_xlsx_cell_text(cell, shared_strings, ns))
                current_col += 1
            table.append(values)

    if not table:
        return []
    headers = [clean_text(x) for x in table[0]]
    records: list[dict[str, str]] = []
    for row in table[1:]:
        record = {header: row[idx] if idx < len(row) else "" for idx, header in enumerate(headers) if header}
        if any(value for value in record.values()):
            records.append(record)
    return records


def prepare_mascqa(source_root: Path, output_root: Path) -> list[dict[str, Any]]:
    xlsx_path = source_root / "mascqa" / "notebooks" / "all_QA.xlsx"
    try:
        import pandas as pd
        frame_iter = pd.read_excel(xlsx_path).to_dict(orient="records")
    except ImportError:
        frame_iter = _read_xlsx_first_sheet(xlsx_path)
    rows: list[dict[str, Any]] = []

    for item in frame_iter:
        qid = normalize_answer(item.get("Question Info"))
        answer = normalize_answer(item.get("Correct Answer"))
        if not qid or not answer:
            continue
        qtype = normalize_answer(item.get("Question Type"))
        question = _parse_question_cell(item.get("QUESTION"))
        option_value = normalize_answer(item.get("Corresponding Value"))
        instructions = (
            "Answer the materials-science question. If options are given, output the correct "
            "option letter(s). For matching questions with listed options, output the option "
            "letter(s), not the full mapping text. If a numeric value is requested, output the "
            "requested value. End with a boxed final answer."
        )
        rows.append(
            {
                "id": f"mascqa:{qid}",
                "dataset_name": "mascqa",
                "source": "MaScQA",
                "license": "CC-BY-NC-SA-4.0",
                "release_compatibility": "internal-only-not-cc-by-sa-compatible",
                "question_type": qtype,
                "topic": normalize_answer(item.get("TOPIC")),
                "input": format_input(
                    benchmark="MaScQA",
                    instructions=instructions,
                    problem=question,
                    metadata={
                        "Topic": normalize_answer(item.get("TOPIC")),
                        "Question type": qtype,
                        "Question id": qid,
                    },
                ),
                "answer": answer,
                "answer_number": answer,
                "reference_solution": option_value,
            }
        )

    out = output_root / "mascqa_internal"
    write_dataset_outputs(out, rows)
    return rows


def _dataset_counts(output_root: Path, dataset: str, count: int) -> dict[str, Any]:
    text_only_path = output_root / dataset / "text_only" / "all.jsonl"
    excluded_path = output_root / dataset / "text_only" / "excluded_missing_artifact.jsonl"
    text_only_count = sum(1 for _ in text_only_path.open(encoding="utf-8")) if text_only_path.exists() else 0
    excluded_count = sum(1 for _ in excluded_path.open(encoding="utf-8")) if excluded_path.exists() else 0
    return {
        "count": count,
        "text_only_count": text_only_count,
        "artifact_excluded_count": excluded_count,
    }


def write_manifest(output_root: Path, counts: dict[str, int]) -> None:
    manifest = {
        "schema": "agentverse_generic_jsonl",
        "required_fields": ["input", "answer"],
        "datasets": {
            "jeebench": {
                **_dataset_counts(output_root, "jeebench", counts.get("jeebench", 0)),
                "license": "MIT",
                "release_compatibility": "compatible_with_cc_by_sa_4_0_trace_release",
            },
            "scibench": {
                **_dataset_counts(output_root, "scibench", counts.get("scibench", 0)),
                "license": "MIT",
                "release_compatibility": "compatible_with_cc_by_sa_4_0_trace_release",
            },
            "mascqa_internal": {
                **_dataset_counts(output_root, "mascqa_internal", counts.get("mascqa_internal", 0)),
                "license": "CC-BY-NC-SA-4.0",
                "release_compatibility": "internal_only_not_compatible_with_cc_by_sa_4_0_release",
            },
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_readme(output_root: Path, counts: dict[str, int]) -> None:
    jee_counts = _dataset_counts(output_root, "jeebench", counts.get("jeebench", 0))
    sci_counts = _dataset_counts(output_root, "scibench", counts.get("scibench", 0))
    mascqa_counts = _dataset_counts(output_root, "mascqa_internal", counts.get("mascqa_internal", 0))
    readme = f"""# Science Benchmark JSONL Slices

Generated by `scripts/prepare_science_benchmarks.py`.

Each row uses the generic AgentVerse JSONL schema with at least:

- `input`: full task text shown to the agent
- `answer`: evaluator-only reference answer
- `reference_solution`: optional evaluator-only solution or answer text
- `dataset_name`, `source`, `license`, `release_compatibility`: provenance fields

## Datasets

| Dataset | Rows | Text-only rows | Excluded as hidden-artifact dependent | License | Release handling |
| --- | ---: | ---: | ---: | --- | --- |
| JEEBench | {jee_counts['count']} | {jee_counts['text_only_count']} | {jee_counts['artifact_excluded_count']} | MIT | Compatible with CC-BY-SA-4.0 trace release with attribution. |
| SciBench | {sci_counts['count']} | {sci_counts['text_only_count']} | {sci_counts['artifact_excluded_count']} | MIT | Compatible with CC-BY-SA-4.0 trace release with attribution. |
| MaScQA | {mascqa_counts['count']} | {mascqa_counts['text_only_count']} | {mascqa_counts['artifact_excluded_count']} | CC-BY-NC-SA-4.0 | Internal experiments only; do not include raw prompts or traces in a CC-BY-SA release without separate permission or a compatible release plan. |

## Text-Only Slices

Each dataset folder also contains a `text_only/` slice for LLM-only agents. This
slice excludes examples that appear to require a missing figure, image, diagram,
or external data table. Excluded rows are written to
`text_only/excluded_missing_artifact.jsonl` with `text_only_exclusion_reasons`.

The filter is intentionally conservative and phrase-based. It removes explicit
references such as "shown in the figure", "from the image", "see figure",
"Figure 2-3", and "data tables", while keeping ordinary text-only phrases such
as "given below" when the needed values are included in the prompt.

## Committed Files

The public-release-compatible dataset folders, `jeebench/` and `scibench/`,
contain:

- `all.jsonl`: full converted slice
- `smoke.jsonl`: first five examples for smoke tests
- `chunks_25/`: 25-example chunks for small batch jobs
- `text_only/`: LLM-only no-hidden-artifact slice with its own `all.jsonl`,
  `smoke.jsonl`, `chunks_25/`, and excluded-row audit file

The `mascqa_internal/` folder intentionally commits only release notes and
ignore rules. Generate its JSONL files locally for internal experiments.

## Notes

JEEBench includes both single-answer and multi-answer multiple-choice examples.
SciBench is mostly numeric or short-answer scientific problem solving with units.
MaScQA mixes multiple-choice, matching, and numeric materials-science questions.

## Regeneration

For public-release-compatible slices:

```bash
python scripts/prepare_science_benchmarks.py --fetch-sources --datasets jeebench scibench
```

For internal MaScQA experiments, run the same script with `mascqa_internal`.
MaScQA is licensed CC-BY-NC-SA-4.0, so generated prompts/traces should not be
included in a CC-BY-SA release unless a separate compatible release plan or
permission is obtained.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--fetch-sources",
        action="store_true",
        help="Clone missing upstream benchmark repos into --source-root.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["jeebench", "scibench", "mascqa_internal"],
        default=["jeebench", "scibench", "mascqa_internal"],
        help="Datasets to prepare. MaScQA is internal-only because of its NC license.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = list(dict.fromkeys(args.datasets))
    if args.fetch_sources:
        fetch_sources(source_root, datasets)

    counts: dict[str, int] = {}
    if "jeebench" in datasets:
        counts["jeebench"] = len(prepare_jeebench(source_root, output_root))
    if "scibench" in datasets:
        counts["scibench"] = len(prepare_scibench(source_root, output_root))
    if "mascqa_internal" in datasets:
        counts["mascqa_internal"] = len(prepare_mascqa(source_root, output_root))
    write_manifest(output_root, counts)
    write_readme(output_root, counts)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
