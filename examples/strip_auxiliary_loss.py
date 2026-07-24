"""Remove auxiliary-loss modules from a checkpoint and save a plain YOLO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from yolo_superpixels import strip_auxiliary_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Checkpoint with auxiliary-loss modules.")
    parser.add_argument("--output", required=True, help="Where to save the stripped checkpoint.")
    parser.add_argument(
        "--keep-criterion-wrapper",
        action="store_true",
        help="Only remove feature wrappers; keep the criterion wrapper if it is present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    removed = strip_auxiliary_loss(model.model, remove_criterion_wrapper=not args.keep_criterion_wrapper)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    print(f"Removed {removed} auxiliary wrapper(s). Saved {output}.")


if __name__ == "__main__":
    main()
