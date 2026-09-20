"""Summarize archived Ultralytics training runs without treating partial CSVs as complete.

Cluster launcher scripts write ``RUN_META.txt`` before training and either
``RUN_COMPLETE.txt`` or ``RUN_FAILED.txt`` afterwards. Older runs sometimes
only contain ``results.csv``; those remain explicitly ``unverified`` because a
partial run has the same file shape as a completed run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

MAP5095 = "metrics/mAP50-95(B)"
RESULT_FIELDS = (
    "epoch",
    "time",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    MAP5095,
)


def parse_key_value_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    return values


def safe_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in {None, ""} else None
    except ValueError:
        return None


def read_results(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"results_present": False, "epoch_count": 0}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"results_present": True, "epoch_count": 0}

    def metric(row: dict[str, str]) -> float:
        return safe_float(row.get(MAP5095)) or float("-inf")

    first, final, best = rows[0], rows[-1], max(rows, key=metric)
    summary: dict[str, Any] = {
        "results_present": True,
        "epoch_count": len(rows),
        "first_epoch": safe_float(first.get("epoch")),
        "last_epoch": safe_float(final.get("epoch")),
        "best_epoch": safe_float(best.get("epoch")),
    }
    for field in RESULT_FIELDS:
        safe_name = field.replace("metrics/", "").replace("(B)", "").replace("/", "_").replace("-", "_")
        summary[f"best_{safe_name}"] = safe_float(best.get(field))
        summary[f"final_{safe_name}"] = safe_float(final.get(field))
    return summary


def read_args(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "args.yaml"
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def summarize_run(run_dir: Path) -> dict[str, Any]:
    complete = parse_key_value_file(run_dir / "RUN_COMPLETE.txt")
    failed = parse_key_value_file(run_dir / "RUN_FAILED.txt")
    meta = parse_key_value_file(run_dir / "RUN_META.txt")
    results = read_results(run_dir / "results.csv")
    args = read_args(run_dir)
    if complete and results["results_present"]:
        status = "complete"
    elif failed:
        status = "failed"
    elif results["results_present"]:
        status = "unverified_results_only"
    else:
        status = "incomplete"

    requested_epochs = args.get("epochs")
    if requested_epochs is not None:
        try:
            requested_epochs = int(requested_epochs)
        except (TypeError, ValueError):
            requested_epochs = None
    last_epoch = results.get("last_epoch")
    budget_reached = (
        None
        if requested_epochs is None or last_epoch is None
        else int(last_epoch) + 1 >= requested_epochs
    )
    return {
        "run_dir": str(run_dir),
        "status": status,
        "job": complete.get("job") or meta.get("job"),
        "tag": complete.get("tag") or meta.get("tag"),
        "model": complete.get("model") or meta.get("model") or args.get("model"),
        "data": complete.get("data") or meta.get("data") or args.get("data"),
        "requested_epochs": requested_epochs,
        "budget_reached": budget_reached,
        "completion_wall_seconds": safe_float(complete.get("wall_seconds")),
        "failure_exit_code": failed.get("exit_code"),
        **results,
    }


def audit(root: Path, output: Path) -> list[dict[str, Any]]:
    root, output = root.resolve(), output.resolve()
    runs: list[dict[str, Any]] = []
    for results_csv in root.rglob("results.csv"):
        if output in results_csv.parents:
            continue
        runs.append(summarize_run(results_csv.parent))
    runs.sort(key=lambda row: row["run_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_archive_audit.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    keys = sorted({key for row in runs for key in row})
    with (output / "training_archive_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Directory containing archived training runs.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for JSON and CSV summaries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = audit(args.root, args.output)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"audited {len(rows)} runs: {counts}")
    print(args.output / "training_archive_audit.csv")


if __name__ == "__main__":
    main()
