"""Validate a superpixel-aware Ultralytics YOLO checkpoint.

Examples:
    python examples/val_ultralytics.py --model runs/.../weights/best.pt --data data.yaml --algorithm color_affinity --extra-channels both
"""

from __future__ import annotations

import argparse

from yolo_superpixels import SuperpixelYOLO


ALGORITHM_CHOICES = [
    "slic",
    "torch_slic",
    "gpu_slic",
    "spit",
    "spit_dense_adjacency",
    "spit_adjacency",
    "spit_voronoi",
    "pseudo_voronoi",
    "torch_grid",
    "gpu_grid",
    "torch_voronoi",
    "gpu_voronoi",
    "color_affinity",
    "gpu_affinity",
    "gradient_affinity",
    "texture_affinity",
    "color_gradient_affinity",
    "opencv_slic",
    "opencv_slico",
    "opencv_mslic",
    "opencv_lsc",
    "opencv_seeds",
]
EDGE_REPRESENTATION_CHOICES = ["same", "boundary", "zero_centered", "boundary_zero_centered"]
CHANNEL_MODE_CHOICES = ["adjacency", "soft_interior", "soft_boundary", "soft_boundary_interior"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="YOLO model path/name, usually a trained best.pt.")
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/superpixel_val")
    parser.add_argument("--name", default="val")
    parser.add_argument("--extra-channels", choices=["enabled", "disabled", "both"], default="enabled")
    parser.add_argument(
        "--channel-mode",
        choices=CHANNEL_MODE_CHOICES,
        default="adjacency",
        help="Must match the extra-channel mode used to fine-tune the checkpoint.",
    )
    parser.add_argument(
        "--edge-representation",
        choices=EDGE_REPRESENTATION_CHOICES,
        default="zero_centered",
        help="Must match the representation used to fine-tune the checkpoint.",
    )

    parser.add_argument(
        "--algorithm",
        choices=ALGORITHM_CHOICES,
        nargs="*",
        default=[],
        help="Zero, one, or many superpixel methods. Omit for RGB-only.",
    )
    parser.add_argument(
        "--extra-algorithm",
        choices=ALGORITHM_CHOICES,
        action="append",
        default=[],
        help="Append one more method; can be repeated.",
    )
    parser.add_argument("--n-segments", type=int, default=300)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--torch-slic-compactness", type=float, default=1.0)
    parser.add_argument("--torch-slic-iters", type=int, default=7)
    parser.add_argument("--soft-max-distance", type=int, default=32)
    parser.add_argument("--soft-boundary-width", type=float, default=6.0)
    parser.add_argument("--soft-affinity-threshold", type=float, default=0.5)
    parser.add_argument(
        "--disabled-value",
        type=float,
        default=0.0,
        help="Scalar used when --extra-channels disabled and --disabled-values is omitted.",
    )
    parser.add_argument(
        "--disabled-values",
        type=float,
        nargs="*",
        default=None,
        help="Per-extra-channel constants used when --extra-channels disabled, e.g. 1 0.",
    )
    parser.add_argument("--spit-module", default="spit.tokenizer.densesp")
    parser.add_argument("--spit-class", default="DenseSPEdgeEmbedder")
    parser.add_argument("--spit-device", default="auto")
    parser.add_argument("--spit-voronoi-cells", type=int, default=256)
    parser.add_argument("--affinity-temperature", type=float, default=0.1)
    parser.add_argument("--affinity-reduction", choices=["l1", "l2"], default="l2")
    parser.add_argument("--gradient-temperature", type=float, default=0.1)
    parser.add_argument("--gradient-color-weight", type=float, default=0.0)
    parser.add_argument("--gradient-weight", type=float, default=1.0)
    parser.add_argument("--voronoi-jitter", type=float, default=0.35)
    parser.add_argument("--voronoi-seed", type=int, default=0)
    parser.add_argument("--voronoi-chunk-size", type=int, default=65536)
    parser.add_argument("--opencv-region-size", type=int, default=None)
    parser.add_argument("--opencv-iterations", type=int, default=10)
    parser.add_argument("--opencv-ruler", type=float, default=10.0)
    parser.add_argument("--opencv-ratio", type=float, default=0.075)
    parser.add_argument("--opencv-num-levels", type=int, default=4)
    parser.add_argument("--opencv-prior", type=int, default=2)
    parser.add_argument("--opencv-histogram-bins", type=int, default=5)
    parser.add_argument("--opencv-double-step", action="store_true")
    parser.add_argument("--opencv-no-connectivity", action="store_true")
    parser.add_argument("--opencv-min-element-size", type=int, default=25)
    parser.add_argument("--opencv-lab", action="store_true")
    parser.add_argument("--opencv-blur-kernel", type=int, default=0)
    return parser.parse_args()


def selected_algorithms(args: argparse.Namespace) -> list[str]:
    return list(args.algorithm or []) + list(args.extra_algorithm or [])


def algorithm_kwargs_for_name(name: str, args: argparse.Namespace) -> dict:
    if name == "slic":
        return {"n_segments": args.n_segments, "compactness": args.compactness, "sigma": args.sigma}
    if name in {"torch_slic", "gpu_slic"}:
        return {
            "n_segments": args.n_segments,
            "compactness": args.torch_slic_compactness,
            "num_iters": args.torch_slic_iters,
        }
    if name in {"spit", "spit_dense_adjacency", "spit_adjacency"}:
        return {
            "module_path": args.spit_module,
            "class_name": args.spit_class,
            "device": args.spit_device,
        }
    if name in {"spit_voronoi", "pseudo_voronoi"}:
        return {"num_cells": args.spit_voronoi_cells}
    if name in {"torch_grid", "gpu_grid"}:
        return {"n_segments": args.n_segments}
    if name in {"torch_voronoi", "gpu_voronoi"}:
        return {
            "n_segments": args.n_segments,
            "jitter": args.voronoi_jitter,
            "seed": args.voronoi_seed,
            "chunk_size": args.voronoi_chunk_size,
        }
    if name in {"gradient_affinity", "texture_affinity"}:
        return {
            "temperature": args.gradient_temperature,
            "reduction": args.affinity_reduction,
            "color_weight": args.gradient_color_weight,
            "gradient_weight": args.gradient_weight,
        }
    if name == "color_gradient_affinity":
        return {
            "temperature": args.gradient_temperature,
            "reduction": args.affinity_reduction,
            "color_weight": max(args.gradient_color_weight, 1.0),
            "gradient_weight": args.gradient_weight,
        }
    if name.startswith("opencv_"):
        return {
            "n_segments": args.n_segments,
            "region_size": args.opencv_region_size,
            "iterations": args.opencv_iterations,
            "ruler": args.opencv_ruler,
            "ratio": args.opencv_ratio,
            "num_levels": args.opencv_num_levels,
            "prior": args.opencv_prior,
            "histogram_bins": args.opencv_histogram_bins,
            "double_step": args.opencv_double_step,
            "enforce_connectivity": not args.opencv_no_connectivity,
            "min_element_size": args.opencv_min_element_size,
            "convert_to_lab": args.opencv_lab,
            "blur_kernel": args.opencv_blur_kernel,
        }
    return {"temperature": args.affinity_temperature, "reduction": args.affinity_reduction}


def algorithm_kwargs_from_args(args: argparse.Namespace) -> list[dict]:
    return [algorithm_kwargs_for_name(name, args) for name in selected_algorithms(args)]


def run_val(args: argparse.Namespace, enabled: bool):
    suffix = "extra_on" if enabled else "extra_off"
    model = SuperpixelYOLO(
        args.model,
        algorithm=selected_algorithms(args),
        algorithm_kwargs=algorithm_kwargs_from_args(args),
        channel_mode=args.channel_mode,
        extra_channels_enabled=enabled,
        edge_representation=args.edge_representation,
        soft_max_distance=args.soft_max_distance,
        soft_boundary_width=args.soft_boundary_width,
        soft_affinity_threshold=args.soft_affinity_threshold,
        disabled_value=args.disabled_value,
        disabled_values=args.disabled_values,
    )
    val_kwargs = {
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": f"{args.name}_{suffix}",
    }
    if args.device is not None:
        val_kwargs["device"] = args.device
    return model.val(extra_channels_enabled=enabled, **val_kwargs)


def main() -> None:
    args = parse_args()
    if args.extra_channels in {"enabled", "both"}:
        run_val(args, enabled=True)
    if args.extra_channels in {"disabled", "both"}:
        run_val(args, enabled=False)


if __name__ == "__main__":
    main()
