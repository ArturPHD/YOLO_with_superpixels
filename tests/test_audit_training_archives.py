from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.audit_training_archives import audit, summarize_run


def write_results(path: Path, epochs: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "time",
                "metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
            ],
        )
        writer.writeheader()
        for epoch in range(epochs):
            writer.writerow(
                {
                    "epoch": epoch,
                    "time": (epoch + 1) * 10,
                    "metrics/precision(B)": 0.4 + epoch * 0.1,
                    "metrics/recall(B)": 0.3,
                    "metrics/mAP50(B)": 0.5 + epoch * 0.1,
                    "metrics/mAP50-95(B)": 0.2 + epoch * 0.1,
                }
            )


def test_results_without_completion_marker_are_unverified(tmp_path: Path) -> None:
    run = tmp_path / "legacy"
    write_results(run / "results.csv")

    summary = summarize_run(run)

    assert summary["status"] == "unverified_results_only"
    assert summary["best_mAP50_95"] == pytest.approx(0.3)
    assert summary["final_mAP50_95"] == pytest.approx(0.3)


def test_complete_marker_and_args_establish_budget_status(tmp_path: Path) -> None:
    run = tmp_path / "complete"
    write_results(run / "results.csv", epochs=2)
    (run / "RUN_COMPLETE.txt").write_text("job=12_0\ntag=actual10\nwall_seconds=123\n")
    (run / "args.yaml").write_text("epochs: 2\nmodel: yolo12n.pt\n")

    summary = summarize_run(run)

    assert summary["status"] == "complete"
    assert summary["job"] == "12_0"
    assert summary["budget_reached"] is True
    assert summary["completion_wall_seconds"] == 123.0


def test_failed_marker_wins_over_partial_results_and_audit_writes_reports(tmp_path: Path) -> None:
    failed = tmp_path / "failed"
    write_results(failed / "results.csv")
    (failed / "RUN_FAILED.txt").write_text("exit_code=137\n")
    output = tmp_path / "report"

    rows = audit(tmp_path, output)

    assert rows[0]["status"] == "failed"
    assert rows[0]["failure_exit_code"] == "137"
    assert (output / "training_archive_audit.json").is_file()
    assert (output / "training_archive_audit.csv").is_file()
