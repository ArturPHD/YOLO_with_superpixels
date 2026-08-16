"""Train an Ultralytics YOLO detector with SPiT token neck injection.

Example:
    python examples/train_ultralytics_neck.py --model yolo12n.pt --data data.yaml --fusion add
"""

from __future__ import annotations

import argparse

from yolo_superpixels import NeckInjectionConfig, SuperpixelNeckYOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo12n.pt", help="YOLO model path/name.")
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/superpixel_neck")
    parser.add_argument("--name", default="spit_neck")
    parser.add_argument("--exist-ok", action="store_true")

    parser.add_argument("--fusion", choices=["add", "concat", "cross_attention"], default="add")
    parser.add_argument("--target-layer", type=int, default=None)
    parser.add_argument("--gate-init", choices=["zero", "tiny", "one"], default="tiny")
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--disable-injection", action="store_true")

    parser.add_argument(
        "--token-provider",
        choices=["region", "spit_labels", "spit"],
        default="region",
        help="region uses torch-native region tokens; spit_labels uses SPiT labels; spit uses full SPiT tokens.",
    )
    parser.add_argument("--spit-tokenizer", choices=["superpixel", "default"], default="superpixel")
    parser.add_argument("--spit-emb-dim", type=int, default=64)
    parser.add_argument("--spit-num-bins", type=int, default=5)
    parser.add_argument("--spit-heads", type=int, default=4)
    parser.add_argument("--spit-depth", type=int, default=1)
    parser.add_argument("--spit-keep-k", type=int, default=0)
    parser.add_argument(
        "--spit-image-size",
        type=int,
        default=160,
        help="Resize RGB before SPiT tokenization. Use 0 to keep the training image size.",
    )
    parser.add_argument("--spit-weights", default=None)
    parser.add_argument(
        "--allow-random-spit",
        action="store_true",
        help="Allow full token_provider=spit without pretrained SPiT weights. Use only for a deliberate ablation.",
    )
    parser.add_argument(
        "--spit-train-mode",
        choices=["joint", "frozen", "delayed", "scratch"],
        default="joint",
        help=(
            "SPiT optimization policy: joint trains normally; frozen keeps pretrained SPiT "
            "fixed; delayed freezes it for --spit-freeze-epochs then fine-tunes it; scratch "
            "trains a random SPiT branch from detection loss."
        ),
    )
    parser.add_argument(
        "--spit-lr-scale",
        type=float,
        default=1.0,
        help="SPiT optimizer LR as a multiplier of the matching YOLO optimizer group LR.",
    )
    parser.add_argument(
        "--spit-freeze-epochs",
        type=int,
        default=0,
        help="Complete epochs to keep SPiT frozen in delayed mode; 21 unfreezes at epoch 22.",
    )
    parser.add_argument(
        "--freeze-spit",
        action="store_true",
        help="Legacy static freeze flag. Prefer --spit-train-mode frozen for training runs.",
    )
    parser.add_argument("--drop-delta", action="store_true")
    parser.add_argument("--no-bbox-reg", action="store_true")
    parser.add_argument("--region-algorithm", choices=["torch_slic", "torch_grid", "torch_voronoi"], default="torch_slic")
    parser.add_argument("--region-n-segments", type=int, default=300)
    parser.add_argument("--region-compactness", type=float, default=1.0)
    parser.add_argument("--region-num-iters", type=int, default=7)
    parser.add_argument("--region-jitter", type=float, default=0.35)
    parser.add_argument("--region-seed", type=int, default=0)
    parser.add_argument("--spit-label-lambda-grad", type=float, default=27.8)
    parser.add_argument("--spit-label-lambda-col", type=float, default=10.0)
    parser.add_argument("--spit-label-maxlvl", type=int, default=4)

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


def build_config(args: argparse.Namespace) -> NeckInjectionConfig:
    return NeckInjectionConfig(
        fusion=args.fusion,
        target_layer=args.target_layer,
        enabled=not args.disable_injection,
        gate_init=args.gate_init,
        attention_heads=args.attention_heads,
        attention_dropout=args.attention_dropout,
        token_provider=args.token_provider,
        tokenizer=args.spit_tokenizer,
        emb_dim=args.spit_emb_dim,
        num_bins=args.spit_num_bins,
        heads=args.spit_heads,
        depth=args.spit_depth,
        keep_k=args.spit_keep_k,
        image_size=None if args.spit_image_size <= 0 else args.spit_image_size,
        spit_weights=args.spit_weights,
        freeze_spit=args.freeze_spit,
        spit_train_mode=args.spit_train_mode,
        spit_lr_scale=args.spit_lr_scale,
        spit_freeze_epochs=args.spit_freeze_epochs,
        drop_delta=args.drop_delta,
        bbox_reg=not args.no_bbox_reg,
        region_algorithm=args.region_algorithm,
        region_n_segments=args.region_n_segments,
        region_compactness=args.region_compactness,
        region_num_iters=args.region_num_iters,
        region_jitter=args.region_jitter,
        region_seed=args.region_seed,
        spit_label_lambda_grad=args.spit_label_lambda_grad,
        spit_label_lambda_col=args.spit_label_lambda_col,
        spit_label_maxlvl=args.spit_label_maxlvl,
        extra_kwargs={"allow_random_spit": True} if args.allow_random_spit else {},
    )


def main() -> None:
    args = parse_args()
    model = SuperpixelNeckYOLO(args.model, config=build_config(args))

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
