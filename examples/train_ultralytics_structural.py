"""Train an SP-11 foreground structural-supervision experiment.

Example:
    python examples/train_ultralytics_structural.py --model yolo12n.pt --data data.yaml --arm C
"""

from __future__ import annotations

import argparse

from yolo_superpixels import StructuralSupervisionConfig, SuperpixelStructuralYOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo12n.pt")
    parser.add_argument("--data", required=True)
    parser.add_argument("--arm", choices=["B", "C", "D"], default="C")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/sp11")
    parser.add_argument("--name", default="sp11")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--lambda-fg", type=float, default=0.05)
    parser.add_argument("--lambda-sp", type=float, default=0.05)
    parser.add_argument("--n-segments", type=int, default=300)
    parser.add_argument("--compactness", type=float, default=1.0)
    parser.add_argument("--num-iters", type=int, default=7)
    parser.add_argument("--purity-threshold", type=float, default=0.9)
    parser.add_argument("--min-region-size", type=int, default=4)
    parser.add_argument("--auxiliary-init-seed", type=int, default=0)
    parser.add_argument("--base-model-sha256", default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> StructuralSupervisionConfig:
    if args.arm == "B":
        lambda_sp, algorithm, kwargs = 0.0, "torch_slic", {"n_segments": args.n_segments, "compactness": args.compactness, "num_iters": args.num_iters}
    elif args.arm == "C":
        lambda_sp, algorithm, kwargs = args.lambda_sp, "torch_slic", {"n_segments": args.n_segments, "compactness": args.compactness, "num_iters": args.num_iters}
    else:
        lambda_sp, algorithm, kwargs = args.lambda_sp, "torch_grid", {"n_segments": args.n_segments}
    return StructuralSupervisionConfig(
        lambda_fg=args.lambda_fg,
        lambda_sp=lambda_sp,
        region_algorithm=algorithm,
        region_kwargs=kwargs,
        purity_threshold=args.purity_threshold,
        min_region_size=args.min_region_size,
        auxiliary_init_seed=args.auxiliary_init_seed,
        base_model_path=args.model,
        base_model_sha256=args.base_model_sha256,
    )


def main() -> None:
    args = parse_args()
    model = SuperpixelStructuralYOLO(args.model, config=build_config(args))
    kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
    }
    if args.device is not None:
        kwargs["device"] = args.device
    model.train(**kwargs)


if __name__ == "__main__":
    main()
