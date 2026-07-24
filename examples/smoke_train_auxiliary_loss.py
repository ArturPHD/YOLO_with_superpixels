"""One-image smoke test for the training-only auxiliary superpixel loss."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from yolo_superpixels import AuxiliaryLossConfig, SuperpixelAuxiliaryYOLO

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo12n.pt")
    parser.add_argument("--algorithm", default="torch_slic")
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--loss-weight", type=float, default=0.05)
    parser.add_argument("--target-resize", choices=["nearest", "area", "bilinear"], default="nearest")
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

    rng = np.random.default_rng(456)
    image = rng.integers(0, 255, size=(96, 96, 3), dtype=np.uint8)
    image[18:78, 24:72] = np.array([50, 170, 225], dtype=np.uint8)

    Image.fromarray(image).save(image_dir / "random_box.jpg")
    (label_dir / "random_box.txt").write_text("0 0.5 0.5 0.5 0.62\n", encoding="utf-8")

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
    data_yaml = make_dataset(scratch_root / "smoke_dataset_auxiliary")
    runs_dir = scratch_root / "smoke_runs"

    config = AuxiliaryLossConfig(
        algorithm_names=[args.algorithm],
        algorithm_kwargs={"n_segments": 24, "num_iters": 2, "compactness": 1.0}
        if args.algorithm in {"torch_slic", "gpu_slic"}
        else {},
        loss_weight=args.loss_weight,
        target_mode="auto",
        target_resize=args.target_resize,
        loss_type="balanced_bce",
        head_init="tiny",
    )
    model = SuperpixelAuxiliaryYOLO(args.model, config=config)
    model.train(
        data=str(data_yaml),
        epochs=1,
        imgsz=args.imgsz,
        batch=1,
        workers=0,
        device=args.device,
        project=str(runs_dir),
        name=f"aux_{Path(args.model).stem}_{args.algorithm}_one_image",
        exist_ok=True,
        plots=False,
        val=True,
        verbose=False,
    )


if __name__ == "__main__":
    main()
