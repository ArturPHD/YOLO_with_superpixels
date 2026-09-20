"""Train YOLO with local superpixel-to-P3 correspondence inside its graph."""

from __future__ import annotations

import argparse

from yolo_superpixels import BackboneCorrespondenceConfig, SuperpixelBackboneYOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo12n.pt")
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="runs/backbone_correspondence")
    parser.add_argument("--name", default="spit_p3_native")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--placement", choices=["native_backbone", "isolated_p3"], default="native_backbone")
    parser.add_argument("--fusion", choices=["overlap_pool", "overlap_attention"], default="overlap_attention")
    parser.add_argument("--p3-layer", type=int, default=None)
    parser.add_argument("--gate-init", choices=["zero", "tiny", "one"], default="tiny")
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--overlap-logit-bias", type=float, default=1.0)
    parser.add_argument("--disable-injection", action="store_true")

    parser.add_argument("--token-provider", choices=["region", "spit_labels", "spit"], default="spit")
    parser.add_argument("--spit-weights", default=None)
    parser.add_argument("--spit-train-mode", choices=["joint", "frozen", "delayed", "scratch"], default="frozen")
    parser.add_argument("--spit-lr-scale", type=float, default=1.0)
    parser.add_argument("--spit-freeze-epochs", type=int, default=0)
    parser.add_argument("--allow-random-spit", action="store_true")
    parser.add_argument("--spit-emb-dim", type=int, default=384)
    parser.add_argument("--spit-num-bins", type=int, default=16)
    parser.add_argument("--spit-heads", type=int, default=6)
    parser.add_argument("--spit-depth", type=int, default=12)
    parser.add_argument("--spit-keep-k", type=int, default=0)
    parser.add_argument("--spit-image-size", type=int, default=160)
    parser.add_argument("--spit-tokenizer", choices=["superpixel", "default"], default="superpixel")
    parser.add_argument("--drop-delta", action="store_true")
    parser.add_argument("--no-bbox-reg", action="store_true")
    parser.add_argument("--region-algorithm", choices=["torch_slic", "torch_grid", "torch_voronoi"], default="torch_slic")
    parser.add_argument("--region-n-segments", type=int, default=300)
    parser.add_argument("--region-compactness", type=float, default=1.0)
    parser.add_argument("--region-num-iters", type=int, default=7)
    parser.add_argument("--region-jitter", type=float, default=0.35)
    parser.add_argument("--region-seed", type=int, default=0)

    parser.add_argument("--optimizer", default=None)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--lrf", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--cos-lr", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BackboneCorrespondenceConfig:
    return BackboneCorrespondenceConfig(
        placement=args.placement,
        fusion=args.fusion,
        p3_layer=args.p3_layer,
        enabled=not args.disable_injection,
        gate_init=args.gate_init,
        attention_heads=args.attention_heads,
        overlap_logit_bias=args.overlap_logit_bias,
        token_provider=args.token_provider,
        emb_dim=args.spit_emb_dim,
        num_bins=args.spit_num_bins,
        heads=args.spit_heads,
        depth=args.spit_depth,
        keep_k=args.spit_keep_k,
        image_size=None if args.spit_image_size <= 0 else args.spit_image_size,
        tokenizer=args.spit_tokenizer,
        spit_weights=args.spit_weights,
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
        extra_kwargs={"allow_random_spit": True} if args.allow_random_spit else {},
    )


def main() -> None:
    args = parse_args()
    model = SuperpixelBackboneYOLO(args.model, config=build_config(args))
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "seed": args.seed,
        "project": args.project,
        "name": args.name,
    }
    optional = {
        "device": args.device,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
        "exist_ok": args.exist_ok if args.exist_ok else None,
        "cos_lr": args.cos_lr if args.cos_lr else None,
    }
    train_kwargs.update({key: value for key, value in optional.items() if value is not None})
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
