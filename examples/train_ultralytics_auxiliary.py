"""Train an Ultralytics YOLO detector with a training-only superpixel auxiliary loss.

Example:
    python examples/train_ultralytics_auxiliary.py --model yolo12n.pt --data data.yaml --algorithm torch_slic
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from yolo_superpixels import AuxiliaryLossConfig, SuperpixelAuxiliaryYOLO
from yolo_superpixels.integrations.ultralytics import SUPPORTED_ALGORITHMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo12n.pt", help="YOLO model path/name.")
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/superpixel_auxiliary")
    parser.add_argument("--name", default="auxiliary_superpixels")
    parser.add_argument("--exist-ok", action="store_true")

    parser.add_argument("--algorithm", choices=SUPPORTED_ALGORITHMS, nargs="+", default=["torch_slic"])
    parser.add_argument(
        "--algorithm-kwargs",
        default=None,
        help="JSON dict for one algorithm or JSON list of dicts for several algorithms.",
    )
    parser.add_argument("--target-layer", type=int, default=None)
    parser.add_argument("--loss-weight", type=float, default=0.05)
    parser.add_argument("--target-mode", choices=["auto", "labels", "edges"], default="auto")
    parser.add_argument("--target-resize", choices=["nearest", "area", "bilinear"], default="nearest")
    parser.add_argument("--loss-type", choices=["balanced_bce", "bce"], default="balanced_bce")
    parser.add_argument("--disable-auxiliary", action="store_true")
    parser.add_argument("--fail-open", action="store_true")
    parser.add_argument("--fail-without-image", action="store_true")
    parser.add_argument("--head-hidden-channels", type=int, default=None)
    parser.add_argument("--head-init", choices=["zero", "tiny", "default"], default="tiny")
    parser.add_argument("--no-scale-by-batch", action="store_true")

    parser.add_argument("--n-segments", type=int, default=300)
    parser.add_argument("--compactness", type=float, default=1.0)
    parser.add_argument("--num-iters", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--reduction", choices=["l1", "l2"], default="l2")

    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--lrf", type=float, default=None)
    parser.add_argument("--optimizer", default=None)
    parser.add_argument("--freeze", type=int, default=None)
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=float, default=None)
    parser.add_argument("--warmup-momentum", type=float, default=None)
    parser.add_argument("--warmup-bias-lr", type=float, default=None)
    return parser.parse_args()


def default_algorithm_kwargs(name: str, args: argparse.Namespace) -> dict[str, Any]:
    name = name.lower()
    if name in {"torch_slic", "gpu_slic"}:
        return {
            "n_segments": args.n_segments,
            "compactness": args.compactness,
            "num_iters": args.num_iters,
        }
    if name == "slic":
        return {"n_segments": args.n_segments, "compactness": args.compactness}
    if name in {"opencv_slic", "opencv_slico", "opencv_mslic", "opencv_lsc", "opencv_seeds"}:
        return {
            "n_segments": args.n_segments,
            "iterations": args.num_iters,
            "ruler": args.compactness,
        }
    if name in {"color_affinity", "gpu_affinity"}:
        return {"temperature": args.temperature, "reduction": args.reduction}
    if name in {"gradient_affinity", "texture_affinity", "color_gradient_affinity"}:
        return {"temperature": args.temperature, "reduction": args.reduction}
    if name in {"torch_grid", "gpu_grid", "torch_voronoi", "gpu_voronoi"}:
        return {"n_segments": args.n_segments}
    if name in {"spit_voronoi", "pseudo_voronoi"}:
        return {"num_cells": args.n_segments}
    return {}


def build_algorithm_kwargs(args: argparse.Namespace) -> dict[str, Any] | list[dict[str, Any]]:
    if args.algorithm_kwargs:
        parsed = json.loads(args.algorithm_kwargs)
        if not isinstance(parsed, (dict, list)):
            raise ValueError("--algorithm-kwargs must decode to a JSON dict or list.")
        return parsed
    values = [default_algorithm_kwargs(name, args) for name in args.algorithm]
    return values[0] if len(values) == 1 else values


def build_config(args: argparse.Namespace) -> AuxiliaryLossConfig:
    return AuxiliaryLossConfig(
        algorithm_names=list(args.algorithm),
        algorithm_kwargs=build_algorithm_kwargs(args),
        target_layer=args.target_layer,
        enabled=not args.disable_auxiliary,
        fail_without_image=args.fail_without_image,
        fail_open=args.fail_open,
        loss_weight=args.loss_weight,
        target_mode=args.target_mode,
        target_resize=args.target_resize,
        loss_type=args.loss_type,
        head_hidden_channels=args.head_hidden_channels,
        head_init=args.head_init,
        scale_by_batch=not args.no_scale_by_batch,
    )


def main() -> None:
    args = parse_args()
    model = SuperpixelAuxiliaryYOLO(args.model, config=build_config(args))
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
        "exist_ok": args.exist_ok if args.exist_ok else None,
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
    }
    train_kwargs.update({key: value for key, value in optional_train_kwargs.items() if value is not None})
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
