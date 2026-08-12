"""Train an Ultralytics YOLO detector with superpixel adjacency channels.

Example:
    python examples/train_ultralytics.py --model yolo11n.pt --data data.yaml --epochs 50 --imgsz 640
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
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model path/name, e.g. yolo11n.pt.")
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="Ultralytics device string, e.g. 0 or 0,1.")
    parser.add_argument("--project", default="runs/superpixel")
    parser.add_argument("--name", default="slic-adjacency")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate. Ignored by optimizer=auto.")
    parser.add_argument("--lrf", type=float, default=None, help="Final LR fraction for Ultralytics scheduler.")
    parser.add_argument("--optimizer", default=None, help="Ultralytics optimizer, e.g. auto, SGD, AdamW.")
    parser.add_argument("--freeze", type=int, default=None, help="Freeze the first N layers for fine-tuning.")
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine LR schedule.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience.")
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=float, default=None)
    parser.add_argument("--warmup-momentum", type=float, default=None)
    parser.add_argument("--warmup-bias-lr", type=float, default=None)
    parser.add_argument("--exist-ok", action="store_true", help="Allow writing into an existing run name.")

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
    parser.add_argument("--init-extra", choices=["zero", "tiny", "mean", "kaiming"], default="tiny")
    parser.add_argument(
        "--channel-mode",
        choices=CHANNEL_MODE_CHOICES,
        default="adjacency",
        help=(
            "Extra-channel family. adjacency gives RGB+right/down channels. "
            "soft_interior gives RGB+one normalized distance-to-boundary channel. "
            "soft_boundary gives RGB+one boundary-band channel. "
            "soft_boundary_interior gives RGB+both soft channels."
        ),
    )
    parser.add_argument(
        "--edge-representation",
        choices=EDGE_REPRESENTATION_CHOICES,
        default="zero_centered",
        help=(
            "How to encode right/down channels: same=old 1-same/0-boundary, "
            "boundary=1-boundary/0-same, zero_centered=+1-same/-1-boundary, "
            "boundary_zero_centered=+1-boundary/-1-same."
        ),
    )

    parser.add_argument("--n-segments", type=int, default=300, help="SLIC only.")
    parser.add_argument("--compactness", type=float, default=10.0, help="CPU SLIC only.")
    parser.add_argument("--sigma", type=float, default=1.0, help="SLIC only.")
    parser.add_argument("--torch-slic-compactness", type=float, default=1.0, help="torch_slic/gpu_slic only.")
    parser.add_argument("--torch-slic-iters", type=int, default=7, help="torch_slic/gpu_slic only.")
    parser.add_argument(
        "--soft-max-distance",
        type=int,
        default=32,
        help="soft_* modes: maximum erosion steps used for distance-to-boundary.",
    )
    parser.add_argument(
        "--soft-boundary-width",
        type=float,
        default=6.0,
        help="soft_boundary modes: pixels over which the boundary signal fades to zero.",
    )
    parser.add_argument(
        "--soft-affinity-threshold",
        type=float,
        default=0.5,
        help="soft_* modes with affinity algorithms: affinity <= threshold is treated as boundary.",
    )

    parser.add_argument(
        "--spit-module",
        default="spit.tokenizer.densesp",
        help="SPiT only: module containing DenseSPEdgeEmbedder.",
    )
    parser.add_argument("--spit-class", default="DenseSPEdgeEmbedder", help="SPiT only.")
    parser.add_argument(
        "--spit-device",
        default="auto",
        help="SPiT only. Use 'auto' to follow the YOLO batch device, or e.g. cuda:0/cpu.",
    )
    parser.add_argument("--spit-voronoi-cells", type=int, default=256, help="spit_voronoi only.")
    parser.add_argument("--affinity-temperature", type=float, default=0.1, help="color_affinity only.")
    parser.add_argument("--affinity-reduction", choices=["l1", "l2"], default="l2", help="color_affinity only.")
    parser.add_argument("--gradient-temperature", type=float, default=0.1, help="gradient_affinity only.")
    parser.add_argument("--gradient-color-weight", type=float, default=0.0, help="gradient_affinity only.")
    parser.add_argument("--gradient-weight", type=float, default=1.0, help="gradient_affinity only.")
    parser.add_argument("--voronoi-jitter", type=float, default=0.35, help="torch_voronoi only.")
    parser.add_argument("--voronoi-seed", type=int, default=0, help="torch_voronoi only.")
    parser.add_argument("--voronoi-chunk-size", type=int, default=65536, help="torch_voronoi only.")
    parser.add_argument("--opencv-region-size", type=int, default=None, help="opencv_* only.")
    parser.add_argument("--opencv-iterations", type=int, default=10, help="opencv_* only.")
    parser.add_argument("--opencv-ruler", type=float, default=10.0, help="opencv_slic/slico/mslic only.")
    parser.add_argument("--opencv-ratio", type=float, default=0.075, help="opencv_lsc only.")
    parser.add_argument("--opencv-num-levels", type=int, default=4, help="opencv_seeds only.")
    parser.add_argument("--opencv-prior", type=int, default=2, help="opencv_seeds only.")
    parser.add_argument("--opencv-histogram-bins", type=int, default=5, help="opencv_seeds only.")
    parser.add_argument("--opencv-double-step", action="store_true", help="opencv_seeds only.")
    parser.add_argument("--opencv-no-connectivity", action="store_true", help="Disable OpenCV connectivity cleanup.")
    parser.add_argument("--opencv-min-element-size", type=int, default=25, help="OpenCV connectivity cleanup.")
    parser.add_argument("--opencv-lab", action="store_true", help="Convert RGB to CieLAB before OpenCV superpixels.")
    parser.add_argument("--opencv-blur-kernel", type=int, default=0, help="Optional Gaussian blur kernel for OpenCV.")
    return parser.parse_args()


def selected_algorithms(args: argparse.Namespace) -> list[str]:
    return list(args.algorithm or []) + list(args.extra_algorithm or [])


def algorithm_kwargs_for_name(name: str, args: argparse.Namespace) -> dict:
    if name == "slic":
        return {
            "n_segments": args.n_segments,
            "compactness": args.compactness,
            "sigma": args.sigma,
        }
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
    return {
        "temperature": args.affinity_temperature,
        "reduction": args.affinity_reduction,
    }


def main() -> None:
    args = parse_args()
    algorithms = selected_algorithms(args)
    algorithm_kwargs = [algorithm_kwargs_for_name(name, args) for name in algorithms]

    model = SuperpixelYOLO(
        args.model,
        algorithm=algorithms,
        algorithm_kwargs=algorithm_kwargs,
        channel_mode=args.channel_mode,
        init_extra=args.init_extra,
        edge_representation=args.edge_representation,
        soft_max_distance=args.soft_max_distance,
        soft_boundary_width=args.soft_boundary_width,
        soft_affinity_threshold=args.soft_affinity_threshold,
    )

    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    optional_train_kwargs = {
        "lr0": args.lr0,
        "lrf": args.lrf,
        "optimizer": args.optimizer,
        "freeze": args.freeze,
        "cos_lr": args.cos_lr if args.cos_lr else None,
        "patience": args.patience,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "warmup_momentum": args.warmup_momentum,
        "warmup_bias_lr": args.warmup_bias_lr,
        "exist_ok": args.exist_ok if args.exist_ok else None,
    }
    train_kwargs.update({key: value for key, value in optional_train_kwargs.items() if value is not None})

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
