#!/usr/bin/env python3
"""Aggregate science-QA reviewer uptake diagnostics from JSONL logs.

This adapter mirrors the coupling-paper review-conditioned summaries, but reads
the deduplicated science-QA `analysis/by_dataset/.../results.jsonl` files.  It
uses saved runtime evaluator labels whenever a candidate was submitted, then
falls back to conservative deterministic answer matching.
"""

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_DATASETS = ("jeebench", "scibench", "mascqa_internal")
DEFAULT_PROTOCOLS = ("per_hint_llm", "broadcast_hint_llm")

REVIEW_POSITION_RE = re.compile(r"\bReview Position:\s*([A-Za-z_]+)", re.IGNORECASE)
CANDIDATE_UNDER_REVIEW_RE = re.compile(
    r"Candidate (?:answer )?under (?:group )?review:\s*(?:Source:\s*[^\\\n]+?\s*)?"
    r"(?:Candidate Answer:\s*)?(\\boxed\{.*?\}|[^\n]+)",
    re.IGNORECASE | re.DOTALL,
)
CANDIDATE_UPDATE_RE = re.compile(
    r"Candidate updated from\s*(\\boxed\{.*?\})\s*to\s*(\\boxed\{.*?\})",
    re.IGNORECASE | re.DOTALL,
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit("Malformed JSON in %s:%s: %s" % (path, line_no, exc))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def extract_boxed(text: Any) -> str:
    if text is None:
        return ""
    raw = str(text)
    marker = r"\boxed{"
    last_match = ""
    start = 0
    while True:
        idx = raw.find(marker, start)
        if idx == -1:
            break
        cursor = idx + len(marker)
        depth = 1
        chunks = []
        while cursor < len(raw) and depth > 0:
            char = raw[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    break
            chunks.append(char)
            cursor += 1
        if depth == 0:
            last_match = "".join(chunks).strip()
        start = idx + 1
    return last_match


def extract_answer(value: Any) -> str:
    if value is None:
        return ""
    boxed = extract_boxed(value)
    if boxed:
        return boxed
    text = str(value).strip()
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1].rstrip(".") if lines else text


def answer_from_log(log: Dict[str, Any]) -> str:
    explicit = log.get("candidate_answer")
    if explicit not in (None, ""):
        answer = extract_answer(explicit)
        if answer and not is_control_text(answer):
            return answer
    for key in ("content", "response", "model_generation"):
        value = log.get(key)
        boxed = extract_boxed(value)
        if boxed and not is_control_text(boxed):
            return boxed
        answer = short_unboxed_answer(value)
        if answer:
            return answer
    return ""


def candidate_under_review(content: Any) -> str:
    text = str(content or "")
    for marker in ("Candidate Answer:", "Candidate under review:"):
        idx = text.lower().find(marker.lower())
        if idx != -1:
            segment = text[idx + len(marker) :]
            segment = re.split(r"\bReview Position:\b|\bPeer Review:\b", segment, maxsplit=1)[0]
            answer = extract_answer(segment)
            if answer:
                return answer
    match = CANDIDATE_UNDER_REVIEW_RE.search(text)
    if match:
        answer = extract_answer(match.group(1))
        if answer:
            return answer
    return extract_answer(text)


def short_unboxed_answer(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if is_control_text(text):
        return ""
    lines = [line.strip().rstrip(".") for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    last = lines[-1]
    if is_control_text(last):
        return ""
    # Accept compact unboxed answers such as `BD` or `50.7 atm`, but avoid
    # treating long reasoning paragraphs as extracted candidate answers.
    if len(last) <= 80 and len(last.split()) <= 8:
        return last
    return ""


def is_control_text(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return True
    return text.startswith("[route:") or text.startswith("[no extracted final answer")


def normalize_answer(value: Any) -> str:
    text = extract_answer(value)
    text = text.replace("\u0008", "\\b")
    text = text.strip()
    text = re.sub(r"\\(?:mathrm|text)\{([^{}]*)\}", r"\1", text)
    text = text.replace("\\,", "").replace("\\ ", "")
    text = text.replace(" ", "")
    text = text.strip("$. ")
    return text.lower()


def letter_key(value: Any) -> str:
    text = extract_answer(value).strip().upper()
    text = re.sub(r"[^A-Z]", "", text)
    if 1 <= len(text) <= 8:
        return "".join(sorted(text))
    return ""


def label_answer(record: Dict[str, Any]) -> str:
    for key in ("label", "answer_number", "answer"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def final_answer(record: Dict[str, Any]) -> str:
    per_eval = record.get("per_evaluation") or {}
    if isinstance(per_eval, dict):
        for key in ("judge_student_final_answer", "final_answer", "rule_prediction"):
            value = per_eval.get(key)
            if value not in (None, ""):
                return str(value).strip()
    for key in ("final_answer", "response", "model_generation"):
        answer = extract_answer(record.get(key))
        if answer:
            return answer
    return ""


def final_correct(record: Dict[str, Any]) -> Optional[bool]:
    per_eval = record.get("per_evaluation") or {}
    if isinstance(per_eval, dict) and per_eval.get("correctness") is not None:
        return bool(per_eval.get("correctness"))
    if record.get("correctness") is not None:
        return bool(record.get("correctness"))
    return deterministic_correct(record, final_answer(record))


def deterministic_correct(record: Dict[str, Any], candidate: str) -> Optional[bool]:
    if not candidate:
        return None
    label = label_answer(record)
    if not label:
        return None
    cand_letters = letter_key(candidate)
    label_letters = letter_key(label)
    if cand_letters and label_letters:
        return cand_letters == label_letters
    return normalize_answer(candidate) == normalize_answer(label)


def build_runtime_label_map(logs: List[Dict[str, Any]]) -> Dict[str, Optional[bool]]:
    labels = {}
    pending = ""
    for log in logs:
        stage = str(log.get("stage") or "")
        content = str(log.get("content") or "")
        if "evaluation_submission" in stage:
            pending = answer_from_log(log)
            continue
        if "evaluation_result" in stage or stage == "evaluation" or "evaluation_post" in stage:
            passed = None
            if log.get("correctness") is not None:
                passed = bool(log.get("correctness"))
            elif re.search(r"\bPASS\b", content, re.IGNORECASE):
                passed = True
            elif re.search(r"\bFAIL\b", content, re.IGNORECASE):
                passed = False
            if pending:
                key = normalize_answer(pending)
                existing = labels.get(key)
                if existing is not None and existing != passed:
                    labels[key] = None
                else:
                    labels[key] = passed
            pending = ""
    return labels


def candidate_correctness(record: Dict[str, Any], candidate: str) -> Tuple[Optional[bool], str]:
    if not candidate:
        return None, "missing"
    runtime = build_runtime_label_map(record.get("logs") or [])
    key = normalize_answer(candidate)
    if key in runtime and runtime[key] is not None:
        return bool(runtime[key]), "runtime_evaluator"
    if normalize_answer(candidate) == normalize_answer(final_answer(record)):
        fc = final_correct(record)
        if fc is not None:
            return fc, "final_runtime_evaluator"
    det = deterministic_correct(record, candidate)
    if det is not None:
        return bool(det), "deterministic"
    return None, "unlabeled"


def per_review_action(content: Any) -> Optional[str]:
    text = str(content or "")
    lowered = text.lower()
    if "[agree]" in lowered or "[submit]" in lowered:
        return "agree"
    if "[route:" in lowered or "fix instruction:" in lowered or "diagnosis:" in lowered:
        return "revise"
    if "needs_revision" in lowered or "propose_correction" in lowered:
        return "revise"
    return None


def broadcast_review_action(content: Any) -> Optional[str]:
    match = REVIEW_POSITION_RE.search(str(content or ""))
    if not match:
        return None
    position = match.group(1).strip().lower()
    if position == "approve":
        return "agree"
    return "revise"


def is_candidate_event(log: Dict[str, Any]) -> bool:
    stage = str(log.get("stage") or "")
    typ = str(log.get("type") or "")
    if typ not in {"message", "summary"}:
        return False
    return any(
        marker in stage
        for marker in (
            "executor",
            "candidate_review",
            "evaluation_submission",
            "planner_post_eval",
            "executor_post_eval",
            "repair",
            "poll",
            "discussion",
        )
    )


def find_next_candidate(
    logs: List[Dict[str, Any]],
    start_index: int,
    current: str = "",
    stop_on_review: bool = False,
) -> str:
    current_key = normalize_answer(current)
    for log in logs[start_index:]:
        stage = str(log.get("stage") or "")
        if stop_on_review and stage.startswith("reviewer"):
            return ""
        if "evaluation_result" in stage:
            continue
        answer = answer_from_log(log) if is_candidate_event(log) else ""
        if answer and normalize_answer(answer) != current_key:
            return answer
    return ""


def extract_per_episodes(dataset: str, record: Dict[str, Any], source_file: Path) -> List[Dict[str, Any]]:
    episodes = []
    logs = record.get("logs") or []
    last_candidate = ""
    after_fail = False
    for index, log in enumerate(logs):
        stage = str(log.get("stage") or "")
        typ = str(log.get("type") or "")
        content = str(log.get("content") or "")
        if "evaluation_result" in stage or stage == "evaluation" or "evaluation_post" in stage:
            if re.search(r"\bFAIL\b", content, re.IGNORECASE) or log.get("correctness") is False:
                after_fail = True
            elif re.search(r"\bPASS\b", content, re.IGNORECASE) or log.get("correctness") is True:
                after_fail = False
        if not (typ == "message" and stage.startswith("reviewer")):
            if not stage.startswith("reviewer") and is_candidate_event(log):
                answer = answer_from_log(log)
                if answer:
                    last_candidate = answer
            continue

        action = per_review_action(content)
        if action is None:
            continue
        marked_before = ""
        if "Candidate Answer:" in content or "Candidate under review:" in content:
            marked_before = candidate_under_review(content)
        reviewer_candidate = answer_from_log(log)
        before = marked_before or (reviewer_candidate if action == "agree" else "") or last_candidate
        after = before
        if action == "revise":
            if reviewer_candidate and normalize_answer(reviewer_candidate) != normalize_answer(before):
                after = reviewer_candidate
            else:
                after = find_next_candidate(logs, index + 1, before, stop_on_review=True) or before
        pre_correct, pre_source = candidate_correctness(record, before)
        post_correct, post_source = candidate_correctness(record, after)
        episodes.append(
            episode_row(
                dataset,
                "per_hint_llm",
                record,
                source_file,
                index,
                stage,
                str(log.get("sender") or ""),
                action,
                before,
                after,
                pre_correct,
                post_correct,
                pre_source,
                post_source,
                after_fail,
                content,
            )
        )
        if after:
            last_candidate = after
    return episodes


def extract_broadcast_episodes(dataset: str, record: Dict[str, Any], source_file: Path) -> List[Dict[str, Any]]:
    episodes = []
    logs = record.get("logs") or []
    after_fail = False
    index = 0
    while index < len(logs):
        log = logs[index]
        stage = str(log.get("stage") or "")
        content = str(log.get("content") or "")
        if "evaluation_result" in stage or stage == "evaluation":
            if re.search(r"\bFAIL\b", content, re.IGNORECASE) or log.get("correctness") is False:
                after_fail = True
            elif re.search(r"\bPASS\b", content, re.IGNORECASE) or log.get("correctness") is True:
                after_fail = False
        if not (stage.startswith("candidate_review") and str(log.get("type") or "") == "message"):
            index += 1
            continue
        before = candidate_under_review(content)
        if not before:
            index += 1
            continue

        j = index + 1
        saw_review = False
        any_revise = False
        review_texts = []
        reviewers = []
        after = before
        while j < len(logs):
            next_log = logs[j]
            next_stage = str(next_log.get("stage") or "")
            next_content = str(next_log.get("content") or "")
            if next_stage.startswith("candidate_review") and str(next_log.get("type") or "") == "message":
                break
            if "evaluation_submission" in next_stage or next_stage == "evaluation":
                break
            update = CANDIDATE_UPDATE_RE.search(next_content)
            if update:
                after = extract_answer(update.group(2)) or after
                j += 1
                break
            if next_stage.startswith("approval") and str(next_log.get("type") or "") == "message":
                action = broadcast_review_action(next_content)
                if action:
                    reviewed = candidate_under_review(next_content)
                    if reviewed and normalize_answer(reviewed) != normalize_answer(before):
                        break
                    saw_review = True
                    any_revise = any_revise or action == "revise"
                    review_texts.append("%s: %s" % (next_log.get("sender") or "", next_content))
                    reviewers.append(str(next_log.get("sender") or ""))
            j += 1

        if saw_review:
            action = "revise" if any_revise else "agree"
            if action == "revise" and normalize_answer(after) == normalize_answer(before):
                after = find_next_distinct_review_candidate(logs, j, before) or after
            pre_correct, pre_source = candidate_correctness(record, before)
            post_correct, post_source = candidate_correctness(record, after)
            episodes.append(
                episode_row(
                    dataset,
                    "broadcast_hint_llm",
                    record,
                    source_file,
                    index,
                    stage,
                    "broadcast_group:" + ",".join(reviewers),
                    action,
                    before,
                    after,
                    pre_correct,
                    post_correct,
                    pre_source,
                    post_source,
                    after_fail,
                    "\n\n".join(review_texts),
                )
            )
        index = max(j, index + 1)
    return episodes


def find_next_distinct_review_candidate(
    logs: List[Dict[str, Any]],
    start_index: int,
    current: str,
) -> str:
    current_key = normalize_answer(current)
    for log in logs[start_index:]:
        stage = str(log.get("stage") or "")
        if "evaluation_submission" in stage or stage == "evaluation":
            return ""
        if stage.startswith("candidate_review") and str(log.get("type") or "") == "message":
            answer = candidate_under_review(log.get("content"))
            if answer and normalize_answer(answer) != current_key:
                return answer
    return ""


def episode_row(
    dataset: str,
    protocol: str,
    record: Dict[str, Any],
    source_file: Path,
    log_index: int,
    stage: str,
    reviewer: str,
    action: str,
    before: str,
    after: str,
    pre_correct: Optional[bool],
    post_correct: Optional[bool],
    pre_source: str,
    post_source: str,
    after_fail: bool,
    review_feedback: str,
) -> Dict[str, Any]:
    before_key = normalize_answer(before)
    after_key = normalize_answer(after)
    return {
        "dataset": dataset,
        "protocol": protocol,
        "id": record.get("id") or "",
        "question_id": record.get("question_id") or "",
        "source_file": str(source_file),
        "log_index": log_index,
        "stage": stage,
        "reviewer": reviewer,
        "action": action,
        "answer_before": before,
        "answer_after": after,
        "answer_before_key": before_key,
        "answer_after_key": after_key,
        "answer_changed": before_key != after_key if before_key and after_key else "",
        "label": label_answer(record),
        "pre_correct": pre_correct,
        "post_correct": post_correct,
        "pre_correct_source": pre_source,
        "post_correct_source": post_source,
        "after_evaluator_fail": after_fail,
        "final_correct": final_correct(record),
        "review_feedback_excerpt": re.sub(r"\s+", " ", str(review_feedback or "")).strip()[:500],
    }


def first_candidate(protocol: str, record: Dict[str, Any]) -> str:
    logs = record.get("logs") or []
    for log in logs:
        stage = str(log.get("stage") or "")
        typ = str(log.get("type") or "")
        if protocol == "per_hint_llm" and typ == "message" and stage.startswith("executor"):
            return answer_from_log(log)
        if protocol == "broadcast_hint_llm":
            if typ == "summary" and stage.startswith("poll"):
                answer = answer_from_log(log)
                if answer:
                    return answer
            if typ == "message" and stage.startswith("discussion"):
                answer = answer_from_log(log)
                if answer:
                    return answer
            if typ == "message" and stage.startswith("candidate_review"):
                answer = candidate_under_review(log.get("content"))
                if answer:
                    return answer
    return final_answer(record)


def summarize_outer(records_by_protocol: Dict[str, List[Tuple[str, Path, Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    rows = []
    for protocol, records in sorted(records_by_protocol.items()):
        examples = len(records)
        first_available = 0
        first_correct = 0
        final_ok = 0
        wrong_initial = 0
        changed = 0
        repaired = 0
        neglected = 0
        try_but_fail = 0
        for dataset, _path, record in records:
            del dataset
            first = first_candidate(protocol, record)
            fcand, _src = candidate_correctness(record, first)
            ffinal = final_correct(record)
            if first:
                first_available += 1
                if fcand is True:
                    first_correct += 1
                if fcand is False:
                    wrong_initial += 1
                    if normalize_answer(final_answer(record)) != normalize_answer(first):
                        changed += 1
                    if ffinal is True:
                        repaired += 1
                    elif normalize_answer(final_answer(record)) == normalize_answer(first):
                        neglected += 1
                    else:
                        try_but_fail += 1
            if ffinal is True:
                final_ok += 1
        rows.append(
            {
                "protocol": protocol,
                "examples": examples,
                "first_candidate_available": first_available,
                "first_candidate_correct_rate": rate(first_correct, first_available),
                "final_correct_rate": rate(final_ok, examples),
                "wrong_initial_count": wrong_initial,
                "changed_rate_given_wrong_initial": rate(changed, wrong_initial),
                "repair_rate_given_wrong_initial": rate(repaired, wrong_initial),
                "neglect_rate_given_wrong_initial": rate(neglected, wrong_initial),
                "try_but_fail_rate_given_wrong_initial": rate(try_but_fail, wrong_initial),
            }
        )
    return rows


def summarize_reviews(episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for key, group in sorted(group_by(episodes, ("dataset", "protocol")).items()):
        rows.append(review_summary_row(key[0], key[1], group))
    for protocol, group in sorted(group_by(episodes, ("protocol",)).items()):
        rows.append(review_summary_row("ALL", protocol[0], group))
    return rows


def review_summary_row(dataset: str, protocol: str, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    eligible = [e for e in episodes if e["pre_correct"] in (True, False) and e["action"] in {"revise", "agree"}]
    positive = [e for e in eligible if e["action"] == "revise"]
    pre_wrong = [e for e in eligible if e["pre_correct"] is False]
    pre_right = [e for e in eligible if e["pre_correct"] is True]
    useful = [e for e in positive if e["pre_correct"] is False]
    misleading = [e for e in positive if e["pre_correct"] is True]
    changed_useful = [e for e in useful if e["answer_changed"] is True]
    repaired_useful = [e for e in useful if e["post_correct"] is True]
    neglected_useful = [e for e in useful if e["answer_changed"] is False]
    try_but_fail = [e for e in useful if e["answer_changed"] is True and e["post_correct"] is not True]
    misleading_resisted = [e for e in misleading if e["post_correct"] is True]
    misleading_harm = [e for e in misleading if e["post_correct"] is False]
    return {
        "dataset": dataset,
        "protocol": protocol,
        "review_episodes": len(episodes),
        "eligible_review_episodes": len(eligible),
        "positive_reviews": len(positive),
        "pre_wrong_reviews": len(pre_wrong),
        "pre_right_reviews": len(pre_right),
        "reviewer_precision": rate(len(useful), len(positive)),
        "reviewer_recall": rate(len(useful), len(pre_wrong)),
        "useful_critique_episodes": len(useful),
        "coupling_rate": rate(len(changed_useful), len(useful)),
        "reviewer_guided_repair_rate": rate(len(repaired_useful), len(useful)),
        "neglect_rate": rate(len(neglected_useful), len(useful)),
        "try_but_fail_rate": rate(len(try_but_fail), len(useful)),
        "misleading_episodes": len(misleading),
        "misleading_resistance_rate": rate(len(misleading_resisted), len(misleading)),
        "misleading_harm_rate": rate(len(misleading_harm), len(misleading)),
    }


def group_by(rows: Iterable[Dict[str, Any]], keys: Tuple[str, ...]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    return grouped


def rate(num: int, den: int) -> Optional[float]:
    return (float(num) / float(den)) if den else None


def pct(value: Optional[float]) -> str:
    return "n/a" if value is None else "%.1f%%" % (100.0 * value)


def render_markdown(outcome_rows: List[Dict[str, Any]], review_rows: List[Dict[str, Any]]) -> str:
    overall = [row for row in review_rows if row["dataset"] == "ALL"]
    lines = [
        "# Science-QA Coupling Diagnostics",
        "",
        "This folder contains post-hoc reviewer uptake diagnostics for the science-QA",
        "runs (JEEBench, SciBench, and internal MaScQA). The adapter reads the",
        "deduplicated `analysis/by_dataset/.../results.jsonl` logs and does not rerun",
        "the task-solving protocols.",
        "",
        "Scope note: candidate correctness reuses saved runtime evaluator labels when",
        "a candidate was submitted. Otherwise it falls back to conservative",
        "deterministic answer matching. Treat these numbers as a mechanism-aligned",
        "science-domain trend check; an exact paper-strength version can rerun the",
        "candidate-level evaluator for every intermediate candidate.",
        "",
        "## Outcome",
        "",
        markdown_table(
            [
                [
                    row["protocol"],
                    str(row["examples"]),
                    pct(row["final_correct_rate"]),
                    str(row["wrong_initial_count"]),
                    pct(row["repair_rate_given_wrong_initial"]),
                    pct(row["neglect_rate_given_wrong_initial"]),
                ]
                for row in outcome_rows
            ],
            ["Protocol", "N", "Final correct", "Wrong first", "Repair", "Neglect"],
        ),
        "",
        "## Review-Conditioned Uptake",
        "",
        markdown_table(
            [
                [
                    row["protocol"],
                    str(row["eligible_review_episodes"]),
                    pct(row["reviewer_precision"]),
                    pct(row["reviewer_recall"]),
                    str(row["useful_critique_episodes"]),
                    pct(row["coupling_rate"]),
                    pct(row["reviewer_guided_repair_rate"]),
                    pct(row["neglect_rate"]),
                ]
                for row in overall
            ],
            [
                "Protocol",
                "Eligible reviews",
                "Precision",
                "Recall",
                "Useful critiques",
                "Coupling",
                "Guided repair",
                "Neglect",
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def markdown_table(rows: List[List[str]], headers: List[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--protocols", nargs="+", default=list(DEFAULT_PROTOCOLS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    analysis_root = args.analysis_root.expanduser().resolve()
    output_dir = (args.output_dir or (analysis_root / "coupling_science_qa")).expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit("Output directory exists; pass --overwrite: %s" % output_dir)
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    records_by_protocol = defaultdict(list)
    episodes = []
    for dataset in args.datasets:
        for protocol in args.protocols:
            path = analysis_root / "by_dataset" / dataset / protocol / "results.jsonl"
            if not path.exists():
                continue
            records = read_jsonl(path)
            for record in records:
                records_by_protocol[protocol].append((dataset, path, record))
                if protocol == "per_hint_llm":
                    episodes.extend(extract_per_episodes(dataset, record, path))
                elif protocol == "broadcast_hint_llm":
                    episodes.extend(extract_broadcast_episodes(dataset, record, path))

    outer_rows = summarize_outer(records_by_protocol)
    review_rows = summarize_reviews(episodes)

    write_csv(
        output_dir / "science_qa_outer_transition_summary.csv",
        outer_rows,
        [
            "protocol",
            "examples",
            "first_candidate_available",
            "first_candidate_correct_rate",
            "final_correct_rate",
            "wrong_initial_count",
            "changed_rate_given_wrong_initial",
            "repair_rate_given_wrong_initial",
            "neglect_rate_given_wrong_initial",
            "try_but_fail_rate_given_wrong_initial",
        ],
    )
    write_csv(
        output_dir / "science_qa_review_conditioned_summary.csv",
        review_rows,
        [
            "dataset",
            "protocol",
            "review_episodes",
            "eligible_review_episodes",
            "positive_reviews",
            "pre_wrong_reviews",
            "pre_right_reviews",
            "reviewer_precision",
            "reviewer_recall",
            "useful_critique_episodes",
            "coupling_rate",
            "reviewer_guided_repair_rate",
            "neglect_rate",
            "try_but_fail_rate",
            "misleading_episodes",
            "misleading_resistance_rate",
            "misleading_harm_rate",
        ],
    )
    write_csv(
        output_dir / "science_qa_review_episodes.csv",
        episodes,
        [
            "dataset",
            "protocol",
            "id",
            "question_id",
            "source_file",
            "log_index",
            "stage",
            "reviewer",
            "action",
            "answer_before",
            "answer_after",
            "answer_before_key",
            "answer_after_key",
            "answer_changed",
            "label",
            "pre_correct",
            "post_correct",
            "pre_correct_source",
            "post_correct_source",
            "after_evaluator_fail",
            "final_correct",
            "review_feedback_excerpt",
        ],
    )
    write_json(
        output_dir / "run_manifest.json",
        {
            "analysis_root": str(analysis_root),
            "datasets": list(args.datasets),
            "protocols": list(args.protocols),
            "review_episodes": len(episodes),
        },
    )
    (output_dir / "README.md").write_text(render_markdown(outer_rows, review_rows), encoding="utf-8")
    print("Wrote science-QA coupling diagnostics to %s" % output_dir)


if __name__ == "__main__":
    main()
