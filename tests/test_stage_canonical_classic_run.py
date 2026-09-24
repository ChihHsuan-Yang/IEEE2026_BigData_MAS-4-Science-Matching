import json

import pytest

from scripts.dataset_release.stage_canonical_classic_run import stage_run


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_stage_run_writes_frozen_builder_input_without_private_paths(tmp_path):
    artifact = tmp_path / "raw" / "trace.txt"
    artifact.parent.mkdir()
    artifact.write_text("trace")
    target = tmp_path / "target.jsonl"
    results = tmp_path / "canonical" / "results.jsonl"
    source = tmp_path / "canonical" / "provenance.jsonl"
    source_manifest = tmp_path / "canonical" / "audit.json"
    _write_jsonl(target, [{"id": "p", "input": "question"}])
    _write_jsonl(
        results,
        [
            {
                "id": "p",
                "question_id": 1,
                "input": "question",
                "per_evaluation": {"correctness": 1, "mode": "llm"},
                "task_run_artifact": "task_runs/trace.txt",
                "_results_path": "/private/results.jsonl",
                "logs": [],
            }
        ],
    )
    _write_jsonl(
        source,
        [
            {
                "key": "1",
                "source_kind": "recovery",
                "resolved_task_run_artifact": str(artifact),
            }
        ],
    )
    source_manifest.write_text("{}\n")

    report = stage_run(
        canonical_results=results,
        canonical_provenance=source,
        target_dataset=target,
        source_manifest=source_manifest,
        output_root=tmp_path / "staging",
        benchmark_id="scibench",
        protocol_id="broadcast",
        actor_model_id="gemma_4_31b",
        evaluator_model_id="gpt_oss_120b",
        compute_site="site-a",
        inference_backend="backend-a_openai_compatible",
        source_run_uri="runstore://site-a/example",
        source_repo_commit="abc123",
        target_index_base=1,
    )

    model_root = tmp_path / "staging/broadcast/scibench/gemma_4_31b"
    staged = json.loads((model_root / "traces/records.jsonl").read_text())
    assert report["frozen"] is True
    assert report["canonical_rows"] == 1
    assert "_results_path" not in staged
    assert "task_run_artifact" not in staged
    assert staged["source_trace_sha256"]
    assert json.loads((model_root / "run_provenance.json").read_text())["source_complete"] is True


def test_stage_run_preserves_base_and_recovery_provenance(tmp_path):
    target = tmp_path / "target.jsonl"
    results = tmp_path / "results.jsonl"
    provenance = tmp_path / "provenance.jsonl"
    manifest = tmp_path / "audit.json"
    artifacts = []
    for name in ("base", "recovery"):
        artifact = tmp_path / f"{name}.txt"
        artifact.write_text(name)
        artifacts.append(artifact)
    _write_jsonl(
        target,
        [
            {"id": "a", "input": "question a"},
            {"id": "b", "input": "question b"},
        ],
    )
    _write_jsonl(
        results,
        [
            {"id": "a", "question_id": 1, "input": "question a", "per_evaluation": {"correctness": 1}},
            {"id": "b", "question_id": 2, "input": "question b", "per_evaluation": {"correctness": 0}},
        ],
    )
    _write_jsonl(
        provenance,
        [
            {"key": "1", "source_kind": "base", "resolved_task_run_artifact": str(artifacts[0])},
            {"key": "2", "source_kind": "recovery", "resolved_task_run_artifact": str(artifacts[1])},
        ],
    )
    manifest.write_text("{}\n")

    report = stage_run(
        canonical_results=results,
        canonical_provenance=provenance,
        target_dataset=target,
        source_manifest=manifest,
        output_root=tmp_path / "staging",
        benchmark_id="scibench",
        protocol_id="broadcast",
        actor_model_id="gemma_4_31b",
        evaluator_model_id="gpt_oss_120b",
        compute_site="site-a",
        inference_backend="mixed",
        source_run_uri="runstore://site-a/example",
        source_repo_commit="composite",
        target_index_base=1,
        base_source_repo_commit="base-sha",
        recovery_source_repo_commit="recovery-sha",
        base_inference_backend="backend-b",
        recovery_inference_backend="backend-a",
    )

    model_root = tmp_path / "staging/broadcast/scibench/gemma_4_31b"
    staged = [json.loads(line) for line in (model_root / "traces/records.jsonl").read_text().splitlines()]
    assert [(row["source_commit"], row["backend"]) for row in staged] == [
        ("base-sha", "backend-b"),
        ("recovery-sha", "backend-a"),
    ]
    assert report["source_repo_commits"] == ["base-sha", "recovery-sha"]
    assert report["inference_backends"] == ["backend-a", "backend-b"]


def test_stage_run_rejects_question_order_mismatch(tmp_path):
    artifact = tmp_path / "trace.txt"
    artifact.write_text("trace")
    target = tmp_path / "target.jsonl"
    results = tmp_path / "results.jsonl"
    provenance = tmp_path / "provenance.jsonl"
    manifest = tmp_path / "audit.json"
    _write_jsonl(target, [{"id": "p", "input": "question"}])
    _write_jsonl(
        results,
        [
            {
                "id": "p",
                "question_id": 2,
                "input": "question",
                "per_evaluation": {"correctness": 1},
            }
        ],
    )
    _write_jsonl(
        provenance,
        [{"key": "2", "resolved_task_run_artifact": str(artifact)}],
    )
    manifest.write_text("{}\n")

    with pytest.raises(ValueError, match="Question-order mismatch"):
        stage_run(
            canonical_results=results,
            canonical_provenance=provenance,
            target_dataset=target,
            source_manifest=manifest,
            output_root=tmp_path / "staging",
            benchmark_id="scibench",
            protocol_id="broadcast",
            actor_model_id="gemma_4_31b",
            evaluator_model_id="gpt_oss_120b",
            compute_site="site-a",
            inference_backend="backend-a",
            source_run_uri="runstore://site-a/example",
            source_repo_commit="abc123",
            target_index_base=1,
        )
