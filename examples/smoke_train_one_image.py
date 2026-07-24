"""One-image smoke test for the input-channel superpixel strategy.

The script creates a tiny YOLO detection dataset, downloads/loads a YOLO12 model,
patches its first convolution to use RGB plus two adjacency channels, and trains
for one epoch. It is intentionally small: the goal is only to catch channel
mismatches in train/val/warmup.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

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
ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo12n.pt")
    parser.add_argument("--algorithm", choices=ALGORITHM_CHOICES, default="slic")
    parser.add_argument("--edge-representation", choices=EDGE_REPRESENTATION_CHOICES, default="zero_centered")
    parser.add_argument("--imgsz", type=int, default=96)
    parser.add_argument("--device", default="cpu")
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

    rng = np.random.default_rng(42)
    image = rng.integers(0, 255, size=(96, 96, 3), dtype=np.uint8)
    image[24:72, 30:66] = np.array([220, 50, 80], dtype=np.uint8)

    Image.fromarray(image).save(image_dir / "random_box.jpg")
    (label_dir / "random_box.txt").write_text("0 0.5 0.5 0.38 0.50\n", encoding="utf-8")

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
    data_yaml = make_dataset(scratch_root / "smoke_dataset")
    runs_dir = scratch_root / "smoke_runs"
    if args.algorithm in {"spit", "spit_dense_adjacency", "spit_adjacency"}:
        algorithm_kwargs = {"device": "auto"}
    elif args.algorithm == "slic":
        algorithm_kwargs = {"n_segments": 24, "compactness": 8.0, "sigma": 0.0}
    elif args.algorithm in {"torch_slic", "gpu_slic"}:
        algorithm_kwargs = {"n_segments": 24, "compactness": 1.0, "num_iters": 5}
    elif args.algorithm in {"spit_voronoi", "pseudo_voronoi"}:
        algorithm_kwargs = {"num_cells": 24}
    elif args.algorithm in {"torch_grid", "gpu_grid"}:
        algorithm_kwargs = {"n_segments": 24}
    elif args.algorithm in {"torch_voronoi", "gpu_voronoi"}:
        algorithm_kwargs = {"n_segments": 24, "jitter": 0.35}
    elif args.algorithm in {"gradient_affinity", "texture_affinity"}:
        algorithm_kwargs = {"temperature": 0.1, "reduction": "l2"}
    elif args.algorithm == "color_gradient_affinity":
        algorithm_kwargs = {"temperature": 0.1, "reduction": "l2", "color_weight": 1.0}
    elif args.algorithm.startswith("opencv_"):
        algorithm_kwargs = {"n_segments": 24, "iterations": 2}
    else:
        algorithm_kwargs = {"temperature": 0.1, "reduction": "l2"}

    model = SuperpixelYOLO(
        args.model,
        algorithm=args.algorithm,
        algorithm_kwargs=algorithm_kwargs,
        init_extra="tiny",
        edge_representation=args.edge_representation,
    )
    model.train(
        data=str(data_yaml),
        epochs=1,
        imgsz=args.imgsz,
        batch=1,
        workers=0,
        device=args.device,
        project=str(runs_dir),
        name=f"input_channels_{Path(args.model).stem}_{args.algorithm}_one_image",
        exist_ok=True,
        plots=False,
        val=True,
        verbose=False,
    )


if __name__ == "__main__":
    main()
