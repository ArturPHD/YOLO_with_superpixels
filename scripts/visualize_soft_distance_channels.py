"""Visualize soft superpixel distance channels for one image.

The output shows the original image, then one row per generated channel with a
heatmap overlay and a raw heatmap panel. This exercises the same
DistanceChannelBuilder used by input-channel training.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from yolo_superpixels.integrations.ultralytics import _build_algorithm_from_name
from yolo_superpixels.superpixels.adjacency import DistanceChannelBuilder


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COCO = Path("datasets/coco")
DEFAULT_OUTPUT = ROOT / "visualizations" / "soft_distance_channels"
CHANNEL_MODES = ("soft_interior", "soft_boundary", "soft_boundary_interior")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--coco-root", type=Path, default=DEFAULT_COCO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--algorithm", nargs="+", default=["torch_slic"])
    parser.add_argument("--channel-mode", nargs="+", choices=CHANNEL_MODES, default=list(CHANNEL_MODES))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel-width", type=int, default=430)
    parser.add_argument("--n-segments", type=int, default=300)
    parser.add_argument("--torch-slic-compactness", type=float, default=1.0)
    parser.add_argument("--torch-slic-iters", type=int, default=7)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--voronoi-jitter", type=float, default=0.35)
    parser.add_argument("--voronoi-seed", type=int, default=0)
    parser.add_argument("--spit-voronoi-cells", type=int, default=256)
    parser.add_argument("--soft-max-distance", type=int, default=32)
    parser.add_argument("--soft-boundary-width", type=float, default=6.0)
    parser.add_argument("--soft-affinity-threshold", type=float, default=0.5)
    return parser.parse_args()


def choose_image(coco_root: Path, seed: int) -> Path:
    image_dir = coco_root / "images" / "val2017"
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No COCO val images found in {image_dir}.")
    rng = random.Random(seed)
    return rng.choice(images)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def algorithm_kwargs(name: str, args: argparse.Namespace) -> dict[str, Any]:
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
    if name in {"torch_grid", "gpu_grid"}:
        return {"n_segments": args.n_segments}
    if name in {"torch_voronoi", "gpu_voronoi"}:
        return {
            "n_segments": args.n_segments,
            "jitter": args.voronoi_jitter,
            "seed": args.voronoi_seed,
        }
    if name in {"spit_voronoi", "pseudo_voronoi"}:
        return {"num_cells": args.spit_voronoi_cells}
    if name in {"spit_dense_adjacency", "spit_adjacency"}:
        return {"device": "auto"}
    if name.startswith("opencv_"):
        return {"n_segments": args.n_segments, "iterations": 10, "convert_to_lab": True}
    return {}


def tensor_from_image(image: np.ndarray, device: str):
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(image).copy()).permute(2, 0, 1).float()
    return (tensor.unsqueeze(0) / 255.0).to(device)


def channel_names(mode: str) -> list[str]:
    if mode == "soft_boundary_interior":
        return ["soft_boundary", "soft_interior"]
    return [mode]


def heatmap(channel: np.ndarray) -> np.ndarray:
    value = np.asarray(channel, dtype=np.float32)
    value = np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    red = value * 255.0
    green = (1.0 - np.abs(value - 0.5) * 2.0) * 190.0
    blue = (1.0 - value) * 255.0
    return np.stack((red, green, blue), axis=-1).clip(0, 255).astype(np.uint8)


def overlay_heatmap(image: np.ndarray, channel: np.ndarray) -> np.ndarray:
    color = heatmap(channel).astype(np.float32)
    base = image.astype(np.float32)
    return (base * 0.55 + color * 0.45).clip(0, 255).astype(np.uint8)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["arialbd.ttf" if bold else "arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"]
    font_dirs = []
    if os.environ.get("WINDIR"):
        font_dirs.append(Path(os.environ["WINDIR"]) / "Fonts")
    font_dirs.append(Path("/usr/share/fonts/truetype/dejavu"))
    for directory in font_dirs:
        for name in names:
            path = directory / name
            if path.exists():
                return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def resize_to_width(image: Image.Image, width: int) -> Image.Image:
    scale = width / image.width
    return image.resize((width, max(1, round(image.height * scale))), Image.Resampling.LANCZOS)


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def placeholder_panel(width: int, height: int, text: str) -> Image.Image:
    panel = Image.new("RGB", (width, height), (238, 238, 238))
    draw = ImageDraw.Draw(panel)
    font = load_font(17)
    lines = wrap_text(text, font, width - 28)
    y = max(16, height // 2 - 11 * len(lines))
    for line in lines:
        draw.text((14, y), line, fill=(80, 80, 80), font=font)
        y += 23
    return panel


def make_grid(
    original: Image.Image,
    image_name: str,
    rows: list[dict[str, Any]],
    output_path: Path,
    panel_width: int,
) -> None:
    gap = 18
    margin = 28
    label_h = 50
    title_h = 64
    original_panel = resize_to_width(original, panel_width * 2 + gap)
    panel_h = resize_to_width(original, panel_width).height
    row_h = label_h + panel_h + gap
    canvas_w = panel_width * 2 + gap + margin * 2
    canvas_h = title_h + original_panel.height + gap + row_h * len(rows) + margin

    canvas = Image.new("RGB", (canvas_w, canvas_h), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(26, bold=True)
    label_font = load_font(18, bold=True)
    small_font = load_font(14)

    y = margin
    draw.text(
        (margin, y),
        f"Soft superpixel distance channels: {image_name}",
        fill=(20, 20, 20),
        font=title_font,
    )
    y += title_h - margin
    canvas.paste(original_panel, (margin, y))
    y += original_panel.height + gap

    for row in rows:
        label = (
            f"{row['algorithm']} | {row['mode']} / {row['channel']} | "
            f"used: {row['used_device']} | {row['status']}"
        )
        draw.text((margin, y), label, fill=(20, 20, 20), font=label_font)
        if row.get("extra"):
            draw.text((margin, y + 25), row["extra"], fill=(90, 90, 90), font=small_font)
        panel_y = y + label_h

        if row["ok"]:
            overlay = resize_to_width(Image.fromarray(row["overlay"]), panel_width)
            raw = resize_to_width(Image.fromarray(row["heatmap"]), panel_width)
        else:
            overlay = placeholder_panel(panel_width, panel_h, row.get("error", "unavailable"))
            raw = placeholder_panel(panel_width, panel_h, row.get("error", "unavailable"))
        canvas.paste(overlay, (margin, panel_y))
        canvas.paste(raw, (margin + panel_width + gap, panel_y))
        y += row_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    image_path = args.image or choose_image(args.coco_root, args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    original = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    image = np.asarray(original)
    device = resolve_device(args.device)
    tensor = tensor_from_image(image, device)

    rows: list[dict[str, Any]] = []
    for algorithm_name in args.algorithm:
        for mode in args.channel_mode:
            start = time.perf_counter()
            row_base = {
                "algorithm": algorithm_name,
                "mode": mode,
                "used_device": device,
                "status": "ok",
                "ok": True,
            }
            try:
                algorithm = _build_algorithm_from_name(algorithm_name, algorithm_kwargs(algorithm_name, args))
                builder = DistanceChannelBuilder(
                    algorithm=algorithm,
                    mode=mode,
                    max_distance=args.soft_max_distance,
                    boundary_width=args.soft_boundary_width,
                    affinity_threshold=args.soft_affinity_threshold,
                )
                channels = builder.channels_for_batch(tensor)[0].detach().float().cpu().numpy()
                elapsed = time.perf_counter() - start
                for channel_index, channel_name in enumerate(channel_names(mode)):
                    channel = channels[channel_index]
                    rows.append(
                        {
                            **row_base,
                            "channel": channel_name,
                            "elapsed_s": round(elapsed, 3),
                            "extra": (
                                f"min={channel.min():.3f}, max={channel.max():.3f}, "
                                f"mean={channel.mean():.3f}, {elapsed:.3f}s"
                            ),
                            "overlay": overlay_heatmap(image, channel),
                            "heatmap": heatmap(channel),
                        }
                    )
            except Exception as exc:
                rows.append(
                    {
                        **row_base,
                        "channel": "-",
                        "status": "unavailable",
                        "ok": False,
                        "elapsed_s": round(time.perf_counter() - start, 3),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            print(f"{algorithm_name}/{mode}: {rows[-1]['status']} ({rows[-1].get('extra', '')})")

    original_copy = output_dir / f"selected_original_{image_path.stem}.jpg"
    original.save(original_copy)
    output_path = output_dir / f"soft_distance_channels_{image_path.stem}.png"
    make_grid(original, image_path.name, rows, output_path, args.panel_width)

    metadata = []
    for row in rows:
        clean = dict(row)
        clean.pop("overlay", None)
        clean.pop("heatmap", None)
        metadata.append(clean)
    metadata_path = output_dir / f"soft_distance_channels_{image_path.stem}.json"
    metadata_path.write_text(
        json.dumps(
            {
                "image": str(image_path),
                "original_copy": str(original_copy),
                "grid": str(output_path),
                "device": device,
                "methods": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"saved: {output_path}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
