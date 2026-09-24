import json

import pytest

from scripts.dataset_release.merge_classic_recovery import merge_recovery


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_merge_fills_only_missing_rows_in_target_order(tmp_path):
    target = tmp_path / "target.jsonl"
    base = tmp_path / "base.jsonl"
    recovery = tmp_path / "recovery.jsonl"
    output = tmp_path / "canonical"
    _write_jsonl(target, [{"id": "a"}, {"id": "b"}, {"id": "c"}])
    _write_jsonl(base, [{"question_id": 0, "value": "base-a"}, {"question_id": 2, "value": "base-c"}])
    _write_jsonl(
        recovery,
        [
            {"question_id": 1, "value": "recovery-b"},
            {"question_id": 2, "value": "must-not-override"},
        ],
    )

    audit = merge_recovery(
        base_results=base,
        recovery_paths=[recovery],
        target_dataset=target,
        output_root=output,
    )

    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert [row["value"] for row in rows] == ["base-a", "recovery-b", "base-c"]
    assert audit["complete"] is True
    assert audit["recovery_records_added"] == 1
    assert audit["recovery_records_ignored_existing"] == 1
    assert (output / "checksums.sha256").exists()


def test_merge_fails_closed_when_target_is_still_incomplete(tmp_path):
    target = tmp_path / "target.jsonl"
    base = tmp_path / "base.jsonl"
    recovery = tmp_path / "recovery.jsonl"
    _write_jsonl(target, [{"id": "a"}, {"id": "b"}])
    _write_jsonl(base, [{"question_id": 0}])
    _write_jsonl(recovery, [])

    with pytest.raises(ValueError, match="incomplete"):
        merge_recovery(
            base_results=base,
            recovery_paths=[recovery],
            target_dataset=target,
            output_root=tmp_path / "out",
        )


def test_merge_rejects_duplicate_recovery_keys_across_files(tmp_path):
    target = tmp_path / "target.jsonl"
    base = tmp_path / "base.jsonl"
    recovery_a = tmp_path / "recovery-a.jsonl"
    recovery_b = tmp_path / "recovery-b.jsonl"
    _write_jsonl(target, [{"id": "a"}])
    _write_jsonl(base, [])
    _write_jsonl(recovery_a, [{"question_id": 0, "value": "a"}])
    _write_jsonl(recovery_b, [{"question_id": 0, "value": "b"}])

    with pytest.raises(ValueError, match="appears in both"):
        merge_recovery(
            base_results=base,
            recovery_paths=[recovery_a, recovery_b],
            target_dataset=target,
            output_root=tmp_path / "out",
        )


def test_merge_supports_one_based_runner_question_ids(tmp_path):
    target = tmp_path / "target.jsonl"
    base = tmp_path / "base.jsonl"
    recovery = tmp_path / "recovery.jsonl"
    output = tmp_path / "out"
    _write_jsonl(target, [{"id": "a"}, {"id": "b"}])
    _write_jsonl(base, [{"question_id": 1, "value": "a"}])
    _write_jsonl(recovery, [{"question_id": 2, "value": "b"}])

    audit = merge_recovery(
        base_results=base,
        recovery_paths=[recovery],
        target_dataset=target,
        output_root=output,
        target_index_base=1,
    )

    assert audit["complete"] is True
    assert audit["target_index_base"] == 1


def test_relative_artifact_resolves_from_raw_results_path(tmp_path):
    target = tmp_path / "target.jsonl"
    base = tmp_path / "base.jsonl"
    recovery = tmp_path / "recovery.jsonl"
    raw_worker = tmp_path / "raw" / "worker"
    raw_worker.mkdir(parents=True)
    artifact = raw_worker / "task_runs" / "example.txt"
    artifact.parent.mkdir()
    artifact.write_text("trace")
    raw_results = raw_worker / "results.jsonl"
    _write_jsonl(target, [{"id": "a"}])
    _write_jsonl(
        base,
        [
            {
                "question_id": 0,
                "task_run_artifact": "task_runs/example.txt",
                "_results_path": str(raw_results),
            }
        ],
    )
    _write_jsonl(recovery, [])

    audit = merge_recovery(
        base_results=base,
        recovery_paths=[recovery],
        target_dataset=target,
        output_root=tmp_path / "out",
        require_artifacts=True,
    )

    provenance = json.loads((tmp_path / "out" / "provenance.jsonl").read_text())
    assert audit["complete"] is True
    assert provenance["resolved_task_run_artifact"] == str(artifact.resolve())
