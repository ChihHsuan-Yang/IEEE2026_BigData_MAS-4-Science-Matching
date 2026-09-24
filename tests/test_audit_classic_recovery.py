import json

from scripts.dataset_release.audit_classic_recovery import audit_recovery


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_audit_reports_complete_recovery(tmp_path):
    target = tmp_path / "target.jsonl"
    recovery = tmp_path / "recovery"
    write_jsonl(target, [{"question_id": 1}, {"question_id": 2}])
    for question_id in (1, 2):
        worker = recovery / f"worker-{question_id}"
        artifact = worker / "trace.txt"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("trace")
        write_jsonl(
            worker / "results.jsonl",
            [{"question_id": question_id, "task_run_artifact": "trace.txt"}],
        )

    report = audit_recovery(
        recovery_root=recovery,
        target_dataset=target,
        target_index_base=1,
    )

    assert report["complete"] is True
    assert report["valid_unique_keys"] == 2
    assert report["missing_target_count"] == 0


def test_audit_rejects_error_rows_and_missing_artifacts(tmp_path):
    target = tmp_path / "target.jsonl"
    recovery = tmp_path / "recovery"
    write_jsonl(target, [{"question_id": 1}, {"question_id": 2}])
    write_jsonl(
        recovery / "worker-1" / "results.jsonl",
        [{"question_id": 1, "task_run_artifact": "missing.txt"}],
    )
    write_jsonl(
        recovery / "worker-2" / "results.jsonl",
        [{"question_id": 2, "error": "endpoint_failure"}],
    )

    report = audit_recovery(
        recovery_root=recovery,
        target_dataset=target,
        target_index_base=1,
    )

    assert report["complete"] is False
    assert report["error_row_count"] == 1
    assert report["missing_artifact_count"] == 2
