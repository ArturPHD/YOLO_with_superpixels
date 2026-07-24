"""Validate a YOLO checkpoint trained with SPiT token neck injection."""

from __future__ import annotations

import argparse

from yolo_superpixels import NeckInjectionConfig, SuperpixelNeckYOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/superpixel_neck_val")
    parser.add_argument("--name", default="val")
    parser.add_argument("--injection", choices=["enabled", "disabled", "both"], default="enabled")

    parser.add_argument("--fusion", choices=["add", "concat", "cross_attention"], default="add")
    parser.add_argument("--target-layer", type=int, default=None)
    parser.add_argument("--gate-init", choices=["zero", "tiny", "one"], default="tiny")
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--token-provider", choices=["region", "spit_labels", "spit"], default="region")
    parser.add_argument("--spit-tokenizer", choices=["superpixel", "default"], default="superpixel")
    parser.add_argument("--spit-emb-dim", type=int, default=64)
    parser.add_argument("--spit-num-bins", type=int, default=5)
    parser.add_argument("--spit-heads", type=int, default=4)
    parser.add_argument("--spit-depth", type=int, default=1)
    parser.add_argument("--spit-keep-k", type=int, default=0)
    parser.add_argument("--spit-image-size", type=int, default=160)
    parser.add_argument("--spit-weights", default=None)
    parser.add_argument(
        "--allow-random-spit",
        action="store_true",
        help="Allow full token_provider=spit without pretrained SPiT weights. Use only for a deliberate ablation.",
    )
    parser.add_argument("--freeze-spit", action="store_true")
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
    return parser.parse_args()


def build_config(args: argparse.Namespace, enabled: bool) -> NeckInjectionConfig:
    return NeckInjectionConfig(
        fusion=args.fusion,
        target_layer=args.target_layer,
        enabled=enabled,
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


def run_val(args: argparse.Namespace, enabled: bool):
    suffix = "inject_on" if enabled else "inject_off"
    model = SuperpixelNeckYOLO(args.model, config=build_config(args, enabled=enabled))
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
    return model.val(injection_enabled=enabled, **val_kwargs)


def main() -> None:
    args = parse_args()
    if args.injection in {"enabled", "both"}:
        run_val(args, enabled=True)
    if args.injection in {"disabled", "both"}:
        run_val(args, enabled=False)


if __name__ == "__main__":
    main()
