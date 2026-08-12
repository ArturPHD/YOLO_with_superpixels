"""Validate trained superpixel checkpoints with swapped superpixel methods."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import time
from typing import Any

from yolo_superpixels import SuperpixelYOLO


DEFAULT_ALGORITHMS = [
    "color_affinity",
    "gradient_affinity",
    "color_gradient_affinity",
    "torch_slic",
    "spit_dense_adjacency",
]
EDGE_REPRESENTATION_CHOICES = ["same", "boundary", "zero_centered", "boundary_zero_centered"]
CHANNEL_MODE_CHOICES = ["adjacency", "soft_interior", "soft_boundary", "soft_boundary_interior"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="LABEL=PATH. Can be repeated, e.g. --checkpoint best=runs/.../best.pt",
    )
    parser.add_argument("--algorithm", nargs="*", default=DEFAULT_ALGORITHMS)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/superpixel_swap_matrix")
    parser.add_argument("--output", type=Path, default=Path("runs/superpixel_swap_matrix/swap_matrix.csv"))
    parser.add_argument("--include-spit", action="store_true", help="Also run raw SPiT dense edge channels.")
    parser.add_argument(
        "--channel-mode",
        choices=CHANNEL_MODE_CHOICES,
        default="adjacency",
        help="Extra-channel mode used by the trained input-channel checkpoint.",
    )
    parser.add_argument(
        "--edge-representation",
        choices=EDGE_REPRESENTATION_CHOICES,
        default="zero_centered",
        help="Representation used by the trained input-channel checkpoint.",
    )
    parser.add_argument("--n-segments", type=int, default=300)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--torch-slic-compactness", type=float, default=1.0)
    parser.add_argument("--torch-slic-iters", type=int, default=7)
    parser.add_argument("--affinity-temperature", type=float, default=0.1)
    parser.add_argument("--gradient-temperature", type=float, default=0.1)
    parser.add_argument("--voronoi-jitter", type=float, default=0.35)
    parser.add_argument("--voronoi-seed", type=int, default=0)
    parser.add_argument("--spit-voronoi-cells", type=int, default=256)
    parser.add_argument("--soft-max-distance", type=int, default=32)
    parser.add_argument("--soft-boundary-width", type=float, default=6.0)
    parser.add_argument("--soft-affinity-threshold", type=float, default=0.5)
    return parser.parse_args()


def parse_checkpoint(value: str) -> tuple[str, str]:
    if "=" not in value:
        path = Path(value)
        return path.stem, str(path)
    label, path = value.split("=", 1)
    return label.strip(), path.strip()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def kwargs_for_algorithm(name: str, args: argparse.Namespace) -> dict[str, Any]:
    if name in {"color_affinity", "gpu_affinity"}:
        return {"temperature": args.affinity_temperature, "reduction": "l2"}
    if name in {"gradient_affinity", "texture_affinity"}:
        return {"temperature": args.gradient_temperature, "reduction": "l2"}
    if name == "color_gradient_affinity":
        return {
            "temperature": args.gradient_temperature,
            "reduction": "l2",
            "color_weight": 1.0,
            "gradient_weight": 1.0,
        }
    if name in {"torch_slic", "gpu_slic"}:
        return {
            "n_segments": args.n_segments,
            "compactness": args.torch_slic_compactness,
            "num_iters": args.torch_slic_iters,
        }
    if name in {"torch_grid", "gpu_grid"}:
        return {"n_segments": args.n_segments}
    if name in {"torch_voronoi", "gpu_voronoi"}:
        return {
            "n_segments": args.n_segments,
            "jitter": args.voronoi_jitter,
            "seed": args.voronoi_seed,
        }
    if name in {"spit_voronoi", "pseudo_voronoi"}:
        return {"num_cells": args.spit_voronoi_cells}
    if name in {"spit", "spit_dense_adjacency", "spit_adjacency"}:
        return {"device": "auto"}
    return {}


def extract_metrics(result) -> dict[str, float]:
    box = result.box
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }


def write_outputs(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checkpoint",
        "algorithm",
        "channel_mode",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "seconds",
        "status",
        "error",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    markdown = output.with_suffix(".md")
    lines = [
        "| Checkpoint | Algorithm | Channel mode | P | R | mAP50 | mAP50-95 | Time | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        if row["status"] == "ok":
            lines.append(
                f"| {row['checkpoint']} | `{row['algorithm']}` | "
                f"`{row['channel_mode']}` | "
                f"{row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['map50']:.4f} | {row['map50_95']:.4f} | "
                f"{row['seconds']:.1f}s | ok |"
            )
        else:
            lines.append(
                f"| {row['checkpoint']} | `{row['algorithm']}` | `{row['channel_mode']}` |  |  |  |  | "
                f"{row['seconds']:.1f}s | {row['error']} |"
            )
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    checkpoints = [parse_checkpoint(value) for value in args.checkpoint]
    algorithms = list(args.algorithm)
    if args.include_spit and "spit" not in algorithms:
        algorithms.append("spit")

    rows: list[dict[str, Any]] = []
    for checkpoint_label, checkpoint_path in checkpoints:
        for algorithm in algorithms:
            print(f"==> {checkpoint_label} with {algorithm}")
            start = time.perf_counter()
            row: dict[str, Any] = {
                "checkpoint": checkpoint_label,
                "algorithm": algorithm,
                "channel_mode": args.channel_mode,
                "precision": "",
                "recall": "",
                "map50": "",
                "map50_95": "",
                "seconds": 0.0,
                "status": "error",
                "error": "",
            }
            try:
                model = SuperpixelYOLO(
                    checkpoint_path,
                    algorithm=[algorithm],
                    algorithm_kwargs=[kwargs_for_algorithm(algorithm, args)],
                    channel_mode=args.channel_mode,
                    extra_channels_enabled=True,
                    edge_representation=args.edge_representation,
                    soft_max_distance=args.soft_max_distance,
                    soft_boundary_width=args.soft_boundary_width,
                    soft_affinity_threshold=args.soft_affinity_threshold,
                )
                result = model.val(
                    data=args.data,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    workers=args.workers,
                    device=args.device,
                    project=args.project,
                    name=f"{safe_name(checkpoint_label)}_{safe_name(algorithm)}",
                    plots=False,
                    save_json=False,
                    verbose=False,
                )
                row.update(extract_metrics(result))
                row["status"] = "ok"
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(row["error"])
            finally:
                row["seconds"] = time.perf_counter() - start
                rows.append(row)
                write_outputs(rows, args.output)

    write_outputs(rows, args.output)
    print(f"saved: {args.output}")
    print(f"saved: {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
