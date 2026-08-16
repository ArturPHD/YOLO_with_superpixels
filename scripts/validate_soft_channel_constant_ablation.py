"""Validate trained soft-channel checkpoints with constant extra-channel ablations.

This script answers a narrow diagnostic question: after fine-tuning a model with
soft superpixel channels, how much does validation quality change if those
channels are replaced by constants instead of being recomputed from the image?
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from yolo_superpixels import SuperpixelYOLO  # noqa: E402

SOFT_MODES = {"soft_interior", "soft_boundary", "soft_boundary_interior"}
METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)


@dataclass(frozen=True)
class AblationCase:
    name: str
    extra_channels_enabled: bool
    disabled_values: tuple[float, ...] | None = None


@dataclass(frozen=True)
class CheckpointSpec:
    run_dir: Path
    weights: Path
    weight_kind: str
    config: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-root",
        type=Path,
        nargs="+",
        required=True,
        help="One or more roots containing Ultralytics experiment directories.",
    )
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML.")
    parser.add_argument("--weights", choices=["best", "last", "both"], default="best")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "soft_channel_ablation")
    parser.add_argument("--name", default="constant_ablation")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=None, help="Debug only: limit checkpoints.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config", payload)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid superpixel config at {path}.")
    return config


def discover_checkpoints(roots: list[Path], weight_kind: str) -> list[CheckpointSpec]:
    kinds = ("best", "last") if weight_kind == "both" else (weight_kind,)
    specs: list[CheckpointSpec] = []
    seen: set[tuple[Path, str]] = set()
    for root in roots:
        for config_path in sorted(root.rglob("superpixel_config.json")):
            config = load_config(config_path)
            if config.get("channel_mode") not in SOFT_MODES:
                continue
            run_dir = config_path.parent
            for kind in kinds:
                weights = run_dir / "weights" / f"{kind}.pt"
                key = (weights.resolve(), kind)
                if weights.exists() and key not in seen:
                    specs.append(CheckpointSpec(run_dir=run_dir, weights=weights, weight_kind=kind, config=config))
                    seen.add(key)
    return sorted(specs, key=lambda spec: (str(spec.run_dir), spec.weight_kind))


def cases_for_mode(mode: str) -> list[AblationCase]:
    if mode == "soft_interior":
        return [
            AblationCase("real_soft_interior", True),
            AblationCase("interior_0", False, (0.0,)),
            AblationCase("interior_1", False, (1.0,)),
        ]
    if mode == "soft_boundary":
        return [
            AblationCase("real_soft_boundary", True),
            AblationCase("boundary_0", False, (0.0,)),
            AblationCase("boundary_1", False, (1.0,)),
        ]
    if mode == "soft_boundary_interior":
        return [
            AblationCase("real_boundary_interior", True),
            AblationCase("boundary_0_interior_0", False, (0.0, 0.0)),
            AblationCase("boundary_1_interior_1", False, (1.0, 1.0)),
            AblationCase("boundary_1_interior_0", False, (1.0, 0.0)),
            AblationCase("boundary_0_interior_1", False, (0.0, 1.0)),
        ]
    raise ValueError(f"Unsupported mode: {mode}")


def _metric_value(metrics: Any, key: str) -> float | None:
    results_dict = getattr(metrics, "results_dict", {}) or {}
    if key in results_dict:
        return float(results_dict[key])

    box = getattr(metrics, "box", None)
    if box is None:
        return None
    fallbacks = {
        "metrics/precision(B)": "mp",
        "metrics/recall(B)": "mr",
        "metrics/mAP50(B)": "map50",
        "metrics/mAP50-95(B)": "map",
    }
    value = getattr(box, fallbacks[key], None)
    return float(value) if value is not None else None


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def run_case(
    spec: CheckpointSpec,
    case: AblationCase,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = spec.config
    algorithms = list(config.get("algorithm_names") or [])
    algorithm_kwargs = list(config.get("algorithm_kwargs") or [{} for _ in algorithms])
    mode = str(config["channel_mode"])
    run_name = _safe_name(
        f"{args.name}_{spec.run_dir.name}_{spec.weight_kind}_{case.name}"
    )

    row: dict[str, Any] = {
        "run_dir": str(spec.run_dir),
        "checkpoint": str(spec.weights),
        "weight_kind": spec.weight_kind,
        "channel_mode": mode,
        "case": case.name,
        "extra_channels_enabled": case.extra_channels_enabled,
        "disabled_values": json.dumps(case.disabled_values),
        "algorithm_names": json.dumps(algorithms),
        "val_run": str(args.project / run_name),
    }
    if args.dry_run:
        row["status"] = "dry_run"
        return row

    start = time.perf_counter()
    model = SuperpixelYOLO(
        str(spec.weights),
        algorithm=algorithms,
        algorithm_kwargs=algorithm_kwargs,
        channel_mode=mode,
        edge_representation=str(config.get("edge_representation", "zero_centered")),
        soft_max_distance=int(config.get("soft_max_distance", 32)),
        soft_boundary_width=float(config.get("soft_boundary_width", 6.0)),
        soft_affinity_threshold=float(config.get("soft_affinity_threshold", 0.5)),
        extra_channels_enabled=case.extra_channels_enabled,
        disabled_values=case.disabled_values,
    )
    val_kwargs: dict[str, Any] = {
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(args.project),
        "name": run_name,
        "split": args.split,
        "plots": False,
        "save_json": False,
        "exist_ok": True,
    }
    if args.device is not None:
        val_kwargs["device"] = args.device
    metrics = model.val(
        extra_channels_enabled=case.extra_channels_enabled,
        **val_kwargs,
    )
    row["seconds"] = round(time.perf_counter() - start, 3)
    for key in METRIC_KEYS:
        row[key] = _metric_value(metrics, key)

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    row["status"] = "ok"
    return row


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    fieldnames = [
        "status",
        "run_dir",
        "checkpoint",
        "weight_kind",
        "channel_mode",
        "case",
        "extra_channels_enabled",
        "disabled_values",
        "algorithm_names",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "seconds",
        "val_run",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "summary.md"
    lines = [
        "# Soft-channel constant ablation",
        "",
        "| checkpoint | mode | case | P | R | mAP50 | mAP50-95 | seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        label = Path(str(row.get("run_dir", ""))).name
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(row.get("channel_mode", "")),
                    str(row.get("case", "")),
                    _format_float(row.get("metrics/precision(B)")),
                    _format_float(row.get("metrics/recall(B)")),
                    _format_float(row.get("metrics/mAP50(B)")),
                    _format_float(row.get("metrics/mAP50-95(B)")),
                    _format_float(row.get("seconds"), digits=1),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_float(value: Any, digits: int = 5) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    args = parse_args()
    specs = discover_checkpoints(args.experiment_root, args.weights)
    if args.limit is not None:
        specs = specs[: args.limit]
    if not specs:
        roots = ", ".join(str(root) for root in args.experiment_root)
        raise SystemExit(f"No soft-channel checkpoints found under: {roots}")

    output_dir = args.project / args.name
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for case in cases_for_mode(str(spec.config["channel_mode"])):
            print(f"[ablation] {spec.run_dir.name} {spec.weight_kind} {case.name}", flush=True)
            row = run_case(spec, case, args)
            rows.append(row)
            write_summary(rows, output_dir)

    print(f"Wrote {output_dir / 'summary.csv'}")
    print(f"Wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
