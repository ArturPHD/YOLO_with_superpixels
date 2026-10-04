"""One-image smoke test for SPiT token neck injection.

By default this uses the lightweight GPU-native ``region`` token provider with
``torch_slic`` labels. Use ``--token-provider spit --spit-tokenizer superpixel
--allow-random-spit`` to deliberately test the full random SPiT token path.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from yolo_superpixels import NeckInjectionConfig, SuperpixelNeckYOLO

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo12n.pt")
    parser.add_argument(
        "--fusion",
        choices=["add", "concat", "cross_attention"],
        nargs="*",
        default=["add", "concat", "cross_attention"],
    )
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--token-provider", choices=["region", "spit_labels", "qstar", "spit"], default="region")
    parser.add_argument("--region-algorithm", choices=["torch_slic", "torch_grid", "torch_voronoi"], default="torch_slic")
    parser.add_argument("--spit-tokenizer", choices=["default", "superpixel"], default="default")
    parser.add_argument("--spit-image-size", type=int, default=64)
    parser.add_argument("--qstar-image-size", type=int, default=160)
    parser.add_argument("--qstar-target-k", type=int, default=128)
    parser.add_argument(
        "--allow-random-spit",
        action="store_true",
        help="Allow full token_provider=spit without pretrained SPiT weights. Use only for a deliberate ablation.",
    )
    parser.add_argument(
        "--scratch-dir",
        default=None,
        help="Optional directory for generated smoke data/runs. Defaults to the project root.",
    )
    return parser.parse_args()


def make_dataset(dataset_dir: Path) -> Path:
    image_dir = dataset_dir / "images" / "train"
    label_dir = dataset_dir / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(123)
    image = rng.integers(0, 255, size=(96, 96, 3), dtype=np.uint8)
    image[22:74, 28:68] = np.array([230, 70, 90], dtype=np.uint8)

    Image.fromarray(image).save(image_dir / "random_box.jpg")
    (label_dir / "random_box.txt").write_text("0 0.5 0.5 0.42 0.54\n", encoding="utf-8")

    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_dir),
                "train": "images/train",
                "val": "images/train",
                "nc": 1,
                "names": ["object"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml


def main() -> None:
    args = parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))
    scratch_root = Path(args.scratch_dir) if args.scratch_dir else ROOT
    data_yaml = make_dataset(scratch_root / "smoke_dataset_neck")
    runs_dir = scratch_root / "smoke_runs"

    for fusion in args.fusion:
        config = NeckInjectionConfig(
            fusion=fusion,
            token_provider=args.token_provider,
            region_algorithm=args.region_algorithm,
            region_n_segments=24,
            region_num_iters=2,
            tokenizer=args.spit_tokenizer,
            image_size=args.qstar_image_size if args.token_provider == "qstar" else args.spit_image_size,
            qstar_target_k=args.qstar_target_k,
            emb_dim=32,
            heads=4,
            depth=1,
            num_bins=4,
            gate_init="tiny",
            extra_kwargs={"allow_random_spit": True} if args.allow_random_spit else {},
        )
        model = SuperpixelNeckYOLO(args.model, config=config)
        model.train(
            data=str(data_yaml),
            epochs=1,
            imgsz=args.imgsz,
            batch=1,
            workers=0,
            device=args.device,
            project=str(runs_dir),
            name=f"neck_{Path(args.model).stem}_{fusion}_one_image",
            exist_ok=True,
            plots=False,
            val=True,
            verbose=False,
        )


if __name__ == "__main__":
    main()
