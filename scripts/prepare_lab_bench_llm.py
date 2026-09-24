#!/usr/bin/env python3
"""Prepare LLM-only LAB-Bench slices for AgentVerse.

The script downloads the public Hugging Face LAB-Bench rows through the
datasets-server API and writes JSONL files compatible with the generic
AgentVerse benchmark loader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import textwrap
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable


DATASET = "futurehouse/lab-bench"
SPLIT = "train"
STRICT_IN_PROMPT_SUBSETS = ("CloningScenarios", "ProtocolQA", "SeqQA")
TEXT_NO_TOOL_SUBSETS = STRICT_IN_PROMPT_SUBSETS + ("DbQA", "LitQA2", "SuppQA")
EXCLUDED_LLM_ONLY_SUBSETS = {
    "FigQA": {"count": 181, "reason": "requires figure image assets"},
    "TableQA": {"count": 244, "reason": "requires table image assets rather than clean text tables"},
}
PAGE_SIZE = 100
DEFAULT_CHUNK_SIZE = 100
DEFAULT_EXTRA_CHUNK_SIZES = (15,)


def _stable_seed(*parts: str) -> int:
    payload = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _request_json(url: str, *, retries: int = 5, sleep_s: float = 2.0) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except Exception as exc:  # pragma: no cover - network resilience
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(sleep_s * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}") from last_error


def fetch_rows(config: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": config,
                "split": SPLIT,
                "offset": offset,
                "length": PAGE_SIZE,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        payload = _request_json(url)
        batch = payload.get("rows") or []
        if not batch:
            break
        rows.extend(item["row"] for item in batch)
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(0.25)
    return rows


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", "\n").split())


def _block_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _answer_options(row: dict[str, Any], subset: str) -> tuple[list[dict[str, Any]], str]:
    ideal = _clean_text(row.get("ideal"))
    distractors = [_clean_text(item) for item in row.get("distractors", [])]
    raw_options = [{"text": ideal, "is_correct": True}] + [
        {"text": distractor, "is_correct": False} for distractor in distractors
    ]
    rng = random.Random(_stable_seed(subset, str(row.get("id", "")), ideal))
    rng.shuffle(raw_options)

    labeled: list[dict[str, Any]] = []
    answer_letter = ""
    for idx, option in enumerate(raw_options):
        letter = chr(ord("A") + idx)
        labeled.append(
            {
                "label": letter,
                "text": option["text"],
                "is_correct": bool(option["is_correct"]),
            }
        )
        if option["is_correct"]:
            answer_letter = letter
    return labeled, answer_letter


def _context_block(row: dict[str, Any], subset: str) -> str:
    parts: list[str] = []
    if subset == "ProtocolQA":
        protocol = _block_text(row.get("protocol"))
        if protocol:
            parts.append("Protocol:\n" + protocol)
    elif subset == "SuppQA":
        paper_title = _clean_text(row.get("paper-title"))
        if paper_title:
            parts.append(f"Paper title: {paper_title}")
    return "\n\n".join(parts)


def build_prompt(row: dict[str, Any], subset: str, slice_name: str, options: list[dict[str, Any]]) -> str:
    question = _block_text(row.get("question"))
    context = _context_block(row, subset)
    option_lines = "\n".join(f"{option['label']}. {option['text']}" for option in options)
    boxed_options = ", ".join(f"\\boxed{{{option['label']}}}" for option in options)
    slice_note = (
        "This is the strict in-prompt-evidence slice: answer using only the "
        "question and any context shown below."
        if slice_name == "llm_strict"
        else "This is the broader text-only, no-tool slice: answer from the prompt and model knowledge only; do not use retrieval, databases, or external tools."
    )

    blocks = [
        "LAB-Bench multiple-choice task for an LLM-only agent.",
        slice_note,
        f"Subset: {subset}",
    ]
    subtask = _clean_text(row.get("subtask"))
    if subtask:
        blocks.append(f"Subtask: {subtask}")
    if context:
        blocks.append(context)
    blocks.extend(
        [
            "Question:\n" + question,
            "Options:\n" + option_lines,
            textwrap.dedent(
                r"""
                Final-answer rules:
                - Choose exactly one option.
                - End with exactly one final line containing only one of the listed option letters in boxed form: {boxed_options}.
                - Do not put explanation after the boxed final answer.
                """
            ).strip().format(boxed_options=boxed_options),
        ]
    )
    return "\n\n".join(blocks)


def normalize_row(row: dict[str, Any], subset: str, slice_name: str) -> dict[str, Any]:
    options, answer_letter = _answer_options(row, subset)
    answer_text = next(option["text"] for option in options if option["is_correct"])
    prompt = build_prompt(row, subset, slice_name, options)
    original_id = str(row.get("id", ""))
    return {
        "id": f"lab-bench:{subset}:{original_id}",
        "input": prompt,
        "question": prompt,
        "answer": answer_letter,
        "answer_number": answer_letter,
        "label": answer_letter,
        "answer_text": answer_text,
        "reference_solution": f"Correct option: {answer_letter}. {answer_text}",
        "options": options,
        "dataset_name": "lab-bench-public",
        "lab_bench_slice": slice_name,
        "lab_bench_subset": subset,
        "subtask": row.get("subtask"),
        "source": row.get("source"),
        "original_id": original_id,
        "domain": ["Biology", "LAB-Bench", subset],
        "prompt_condition": "llm-only text prompt; no tools; no retrieval",
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_shuffled_chunks(slice_dir: Path, shuffled: list[dict[str, Any]], chunk_size: int) -> int:
    chunks_dir = slice_dir / f"chunks_{chunk_size:02d}_shuffled"
    for idx in range(0, len(shuffled), chunk_size):
        chunk_rows = shuffled[idx : idx + chunk_size]
        write_jsonl(chunks_dir / f"chunk_{idx // chunk_size:03d}.jsonl", chunk_rows)
    return (len(shuffled) + chunk_size - 1) // chunk_size


def write_slice(
    root: Path,
    slice_name: str,
    subsets: tuple[str, ...],
    raw_rows: dict[str, list[dict[str, Any]]],
    chunk_sizes: tuple[int, ...],
) -> dict[str, Any]:
    slice_dir = root / slice_name
    normalized_by_subset: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    for subset in subsets:
        normalized = [normalize_row(row, subset, slice_name) for row in raw_rows[subset]]
        normalized_by_subset[subset] = normalized
        all_rows.extend(normalized)
        write_jsonl(slice_dir / "by_subset" / f"{subset}.jsonl", normalized)

    write_jsonl(slice_dir / "all.jsonl", all_rows)

    shuffled = list(all_rows)
    random.Random(_stable_seed(slice_name, "chunk-shuffle-v1")).shuffle(shuffled)
    num_chunks_by_size = {
        str(chunk_size): _write_shuffled_chunks(slice_dir, shuffled, chunk_size)
        for chunk_size in chunk_sizes
    }

    first700 = shuffled[: min(700, len(shuffled))]
    write_jsonl(slice_dir / "first700_shuffled.jsonl", first700)

    return {
        "slice": slice_name,
        "subsets": {subset: len(rows) for subset, rows in normalized_by_subset.items()},
        "total": len(all_rows),
        "chunk_sizes": list(chunk_sizes),
        "num_chunks_by_size": num_chunks_by_size,
        "first700_count": len(first700),
    }


def write_readme(root: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# LAB-Bench Public LLM-Only Slices",
        "",
        "These files are generated by `scripts/prepare_lab_bench_llm.py` from the public Hugging Face dataset `futurehouse/lab-bench`: https://huggingface.co/datasets/futurehouse/lab-bench.",
        "",
        "## Why We Filter",
        "",
        "The paper protocols currently use text-only LLM agents and OpenAI-compatible chat completions against a served endpoint. The public LAB-Bench release includes figure and table questions whose rows point to image assets. Those are excluded here so the benchmark condition remains text-only and comparable across actor model families.",
        "",
        "We keep two slices:",
        "",
        "- `llm_strict`: `CloningScenarios`, `ProtocolQA`, and `SeqQA`. These questions place the main evidence in the prompt and are the safest LLM-only validation slice.",
        "- `text_no_tool`: the strict slice plus `DbQA`, `LitQA2`, and `SuppQA`. This is a broader closed-book/no-tool stress test; it should not be described as a database, literature-retrieval, or supplementary-material tool-use evaluation.",
        "",
        "Excluded public subsets:",
    ]
    for subset, info in EXCLUDED_LLM_ONLY_SUBSETS.items():
        lines.append(f"- `{subset}`: {info['reason']} ({info['count']} public examples).")
    lines.extend(["", "## Generated Files", ""])
    for summary in summaries:
        lines.append(f"### `{summary['slice']}`")
        lines.append("")
        lines.append(f"- Total examples: {summary['total']}")
        lines.append("- Shuffled chunk sets:")
        for chunk_size in summary["chunk_sizes"]:
            count = summary["num_chunks_by_size"][str(chunk_size)]
            lines.append(f"  - `chunks_{chunk_size:02d}_shuffled`: {count} chunks")
        lines.append(f"- `first700_shuffled.jsonl`: {summary['first700_count']} examples for a first-pass run")
        lines.append("- Subset counts:")
        for subset, count in summary["subsets"].items():
            lines.append(f"  - `{subset}`: {count}")
        lines.append("")
    lines.extend(
        [
            "## Prompt And Label Format",
            "",
            "Each JSONL row contains `input`, `answer`, `answer_number`, and `label`. The gold label is the shuffled multiple-choice letter. The prompt requires the agent to end with exactly one listed boxed option letter, for example `\\boxed{A}`. The evaluator-only `reference_solution` records the correct option letter and option text.",
            "",
            "The public LAB-Bench canary field is intentionally not included in the agent-facing prompt. Source DOI fields and held-out key-passage fields, when present in the source rows, are not added to the prompt either; they are metadata/evaluator-side material, not agent evidence.",
            "",
            "The option order is deterministically shuffled from the LAB-Bench row id and ideal answer, so rerunning the script produces stable files.",
            "",
            "## Regeneration",
            "",
            "```bash",
            "python scripts/prepare_lab_bench_llm.py --output data/lab-bench-public",
            "```",
            "",
        ]
    )
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/lab-bench-public")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--extra-chunk-sizes",
        default=",".join(str(size) for size in DEFAULT_EXTRA_CHUNK_SIZES),
        help="Comma-separated additional chunk sizes to materialize, for example 15 for ~50 chunks on the strict slice.",
    )
    args = parser.parse_args()

    root = Path(args.output)
    extra_chunk_sizes = tuple(
        int(part.strip())
        for part in str(args.extra_chunk_sizes or "").split(",")
        if part.strip()
    )
    chunk_sizes = tuple(dict.fromkeys((args.chunk_size, *extra_chunk_sizes)))
    all_subsets = tuple(dict.fromkeys(TEXT_NO_TOOL_SUBSETS))
    raw_rows = {subset: fetch_rows(subset) for subset in all_subsets}

    summaries = [
        write_slice(root, "llm_strict", STRICT_IN_PROMPT_SUBSETS, raw_rows, chunk_sizes),
        write_slice(root, "text_no_tool", TEXT_NO_TOOL_SUBSETS, raw_rows, chunk_sizes),
    ]

    manifest = {
        "dataset": DATASET,
        "split": SPLIT,
        "generated_slices": summaries,
        "excluded_for_llm_only": EXCLUDED_LLM_ONLY_SUBSETS,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(root, summaries)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
