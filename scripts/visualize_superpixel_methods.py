"""Visualize available superpixel-channel algorithms on one image.

The output is a single grid image: original COCO photo on top, then one row per
method with a boundary overlay and a pseudo-colored segmentation map.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COCO = Path("datasets/coco")
DEFAULT_OUTPUT = ROOT / "visualizations" / "superpixel_methods"


@dataclass(frozen=True)
class MethodSpec:
    key: str
    title: str
    backend: str
    kwargs: dict[str, Any]
    timeout_s: int = 45
    gpu_candidate: bool = False
    strict_gpu: bool = False
    recommended: bool = True
    default_visualization: bool = True
    note: str = ""


METHODS = [
    MethodSpec(
        "color_affinity",
        "Color affinity",
        "GPU/CPU PyTorch",
        {"temperature": 0.1},
        gpu_candidate=True,
        strict_gpu=True,
    ),
    MethodSpec(
        "gradient_affinity",
        "Gradient affinity",
        "GPU/CPU PyTorch",
        {"temperature": 0.1},
        gpu_candidate=True,
        strict_gpu=True,
    ),
    MethodSpec(
        "color_gradient_affinity",
        "Color + gradient affinity",
        "GPU/CPU PyTorch",
        {"temperature": 0.1, "color_weight": 1.0},
        gpu_candidate=True,
        strict_gpu=True,
    ),
    MethodSpec(
        "torch_slic",
        "Torch SLIC-like",
        "GPU/CPU PyTorch",
        {"n_segments": 300, "compactness": 1.0, "num_iters": 7},
        gpu_candidate=True,
        strict_gpu=True,
    ),
    MethodSpec(
        "torch_grid",
        "Torch grid",
        "GPU/CPU PyTorch",
        {"n_segments": 300},
        gpu_candidate=True,
        strict_gpu=True,
        recommended=False,
        note="control only: regular grid, not content-aware",
    ),
    MethodSpec(
        "torch_voronoi",
        "Torch Voronoi",
        "GPU/CPU PyTorch",
        {"n_segments": 300},
        gpu_candidate=True,
        strict_gpu=True,
        recommended=False,
        note="control only: spatial Voronoi, not content-aware",
    ),
    MethodSpec(
        "spit_voronoi",
        "SPiT pseudo-Voronoi",
        "GPU tensor; CPU centroid init",
        {"num_cells": 256},
        gpu_candidate=True,
        recommended=False,
        note="control only: pseudo-Voronoi partition, not image-adaptive superpixels",
    ),
    MethodSpec(
        "spit_dense_adjacency",
        "SPiT dense adjacency",
        "GPU/CuPy experimental",
        {},
        timeout_s=25,
        gpu_candidate=True,
        default_visualization=False,
        note="experimental: can timeout on full-size Windows/CuPy visualization",
    ),
    MethodSpec(
        "spit_dense",
        "SPiT dense raw",
        "GPU/CuPy raw channels",
        {},
        timeout_s=25,
        gpu_candidate=True,
        default_visualization=False,
        note="experimental: can timeout on full-size Windows/CuPy visualization",
    ),
    MethodSpec("slic", "scikit-image SLIC", "CPU", {"n_segments": 300, "compactness": 10.0, "sigma": 1.0}),
    MethodSpec("opencv_slic", "OpenCV SLIC", "CPU", {"n_segments": 300, "iterations": 10, "convert_to_lab": True}),
    MethodSpec("opencv_slico", "OpenCV SLICO", "CPU", {"n_segments": 300, "iterations": 10, "convert_to_lab": True}),
    MethodSpec("opencv_mslic", "OpenCV MSLIC", "CPU", {"n_segments": 300, "iterations": 10, "convert_to_lab": True}),
    MethodSpec("opencv_lsc", "OpenCV LSC", "CPU", {"n_segments": 300, "iterations": 10, "convert_to_lab": True}),
    MethodSpec("opencv_seeds", "OpenCV SEEDS", "CPU", {"n_segments": 300, "iterations": 10}),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", type=Path, default=DEFAULT_COCO)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel-width", type=int, default=430)
    parser.add_argument("--timeout", type=int, default=None, help="Override per-method timeout.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ... for tensor methods.")
    parser.add_argument("--only-gpu", action="store_true", help="Show only methods that can run on the GPU path.")
    parser.add_argument(
        "--include-controls",
        action="store_true",
        help="Include grid/Voronoi/pseudo-Voronoi control baselines. They are hidden by default.",
    )
    parser.add_argument(
        "--include-experimental",
        action="store_true",
        help="Include experimental visualizations hidden by default, such as full-size SPiT dense.",
    )
    parser.add_argument(
        "--layout",
        choices=("overlay-left", "segments-left"),
        default="overlay-left",
        help="Panel order in each method row.",
    )
    parser.add_argument(
        "--run-spit-dense",
        action="store_true",
        help="Try SPiT DenseSPEdgeEmbedder. On this Windows/CuPy stack it may hang, so it is skipped by default.",
    )
    parser.add_argument(
        "--spit-dense-mode",
        choices=("timeout", "direct"),
        default="timeout",
        help="Run SPiT dense in a guarded subprocess or directly in the main process.",
    )
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


def edges_to_labels(edges: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Build connected components from right/down channels."""

    edges = np.asarray(edges)
    if edges.shape[0] != 2:
        raise ValueError(f"Expected 2xHxW edges, got {edges.shape}.")
    _, height, width = edges.shape
    parent = np.arange(height * width, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    right = edges[0] > threshold
    down = edges[1] > threshold
    for y in range(height):
        row = y * width
        for x in range(width - 1):
            if right[y, x]:
                union(row + x, row + x + 1)
    for y in range(height - 1):
        row = y * width
        next_row = (y + 1) * width
        for x in range(width):
            if down[y, x]:
                union(row + x, next_row + x)

    roots = np.array([find(i) for i in range(height * width)], dtype=np.int32)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.reshape(height, width).astype(np.int32)


def labels_to_boundaries(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    boundaries = np.zeros(labels.shape, dtype=bool)
    boundaries[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundaries[1:, :] |= labels[1:, :] != labels[:-1, :]
    return boundaries


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    hashed = labels * 1103515245 + 12345
    red = (hashed & 255).astype(np.uint8)
    green = ((hashed >> 8) & 255).astype(np.uint8)
    blue = ((hashed >> 16) & 255).astype(np.uint8)
    colors = np.stack([red, green, blue], axis=-1)
    return ((colors.astype(np.uint16) + 70).clip(0, 255)).astype(np.uint8)


def overlay_boundaries(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    base = image.astype(np.float32).copy()
    boundaries = labels_to_boundaries(labels)
    color = np.array([255.0, 40.0, 40.0], dtype=np.float32)
    base[boundaries] = base[boundaries] * 0.25 + color * 0.75

    # Thicken by one neighbor for readability.
    thick = boundaries.copy()
    thick[:, 1:] |= boundaries[:, :-1]
    thick[:, :-1] |= boundaries[:, 1:]
    thick[1:, :] |= boundaries[:-1, :]
    thick[:-1, :] |= boundaries[1:, :]
    base[thick & ~boundaries] = base[thick & ~boundaries] * 0.55 + color * 0.45
    return base.clip(0, 255).astype(np.uint8)


def _tensor_from_image(image: np.ndarray, device: str):
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(image).copy()).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)


def run_method_worker(
    key: str,
    kwargs: dict[str, Any],
    image: np.ndarray,
    device: str,
    queue: mp.Queue,
) -> None:
    try:
        labels = run_method(key, kwargs, image, device)
        queue.put({"ok": True, "labels": labels.astype(np.int32)})
    except Exception as exc:
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def run_method(key: str, kwargs: dict[str, Any], image: np.ndarray, device: str) -> np.ndarray:
    if key == "color_affinity":
        from yolo_superpixels import TorchColorAffinityAlgorithm

        alg = TorchColorAffinityAlgorithm(**kwargs)
        edges = alg.edges_for_batch(_tensor_from_image(image, device))[0].detach().cpu().numpy()
        return edges_to_labels(edges, threshold=0.55)

    if key in {"gradient_affinity", "color_gradient_affinity"}:
        from yolo_superpixels import TorchGradientAffinityAlgorithm

        alg = TorchGradientAffinityAlgorithm(**kwargs)
        edges = alg.edges_for_batch(_tensor_from_image(image, device))[0].detach().cpu().numpy()
        return edges_to_labels(edges, threshold=0.55)

    if key == "torch_slic":
        from yolo_superpixels import TorchSLICSuperpixelAlgorithm

        alg = TorchSLICSuperpixelAlgorithm(**kwargs)
        tensor = _tensor_from_image(image, device)
        with __import__("torch").no_grad():
            labels = alg._segment_batch(tensor[:, :3])[0].detach().cpu().numpy()
        return labels

    if key == "torch_grid":
        from yolo_superpixels import TorchGridSuperpixelAlgorithm

        alg = TorchGridSuperpixelAlgorithm(**kwargs)
        tensor = _tensor_from_image(image, device)
        labels = alg._segment_batch(tensor[:, :3])[0].detach().cpu().numpy()
        return labels

    if key == "torch_voronoi":
        from yolo_superpixels import TorchVoronoiSuperpixelAlgorithm

        alg = TorchVoronoiSuperpixelAlgorithm(**kwargs)
        tensor = _tensor_from_image(image, device)
        labels = alg._segment_batch(tensor[:, :3])[0].detach().cpu().numpy()
        return labels

    if key == "spit_voronoi":
        from yolo_superpixels import SPiTPseudoVoronoiAlgorithm
        from spit.tokenizer.voronoi import fast_pseudo_voronoi

        SPiTPseudoVoronoiAlgorithm(**kwargs)
        tensor = _tensor_from_image(image, device)
        with __import__("torch").no_grad():
            labels = fast_pseudo_voronoi(tensor[:, :3], int(kwargs["num_cells"]))[0].detach().cpu().numpy()
        return labels

    if key in {"spit_dense", "spit_dense_adjacency", "spit_adjacency"}:
        import torch
        from yolo_superpixels.superpixels.algorithms import SPiTDenseEdgeAlgorithm

        SPiTDenseEdgeAlgorithm._ensure_local_spit_on_path()
        from spit.tokenizer.densesp import DenseSPEdgeEmbedder

        tensor = _tensor_from_image(image, device)
        embedder = DenseSPEdgeEmbedder(return_seg=True).to(device).eval()
        with torch.no_grad():
            _, seg = embedder(tensor)
        return seg[0].detach().cpu().numpy()

    if key == "slic":
        from yolo_superpixels import SLICSuperpixelAlgorithm

        return SLICSuperpixelAlgorithm(**kwargs)(image)

    if key.startswith("opencv_"):
        from yolo_superpixels import OpenCVSuperpixelAlgorithm

        method = key.removeprefix("opencv_")
        return OpenCVSuperpixelAlgorithm(method=method, **kwargs)(image)

    raise ValueError(f"Unknown method: {key}")


def run_with_timeout(spec: MethodSpec, image: np.ndarray, device: str, timeout_override: int | None) -> dict[str, Any]:
    timeout_s = timeout_override or spec.timeout_s
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=run_method_worker, args=(spec.key, spec.kwargs, image, device, queue))
    start = time.perf_counter()
    proc.start()
    proc.join(timeout_s)
    elapsed = time.perf_counter() - start

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return {"ok": False, "error": f"timeout after {timeout_s}s", "elapsed_s": elapsed}
    if queue.empty():
        return {"ok": False, "error": f"process exited with code {proc.exitcode}", "elapsed_s": elapsed}
    result = queue.get()
    result["elapsed_s"] = elapsed
    return result


def run_without_timeout(spec: MethodSpec, image: np.ndarray, device: str) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        labels = run_method(spec.key, spec.kwargs, image, device)
        return {"ok": True, "labels": labels.astype(np.int32), "elapsed_s": time.perf_counter() - start}
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": time.perf_counter() - start,
        }


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


def placeholder_panel(width: int, height: int, text: str) -> Image.Image:
    panel = Image.new("RGB", (width, height), (238, 238, 238))
    draw = ImageDraw.Draw(panel)
    font = load_font(18)
    lines = wrap_text(text, font, width - 32)
    y = max(16, height // 2 - 12 * len(lines))
    for line in lines:
        draw.text((16, y), line, fill=(80, 80, 80), font=font)
        y += 24
    return panel


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
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


def make_grid(
    original: Image.Image,
    image_name: str,
    rows: list[dict[str, Any]],
    output_path: Path,
    panel_width: int,
    layout: str,
) -> None:
    gap = 18
    margin = 28
    label_h = 46
    title_h = 64
    original_panel = resize_to_width(original, panel_width * 2 + gap)
    panel_h = resize_to_width(original, panel_width).height
    row_h = label_h + panel_h + gap
    canvas_w = panel_width * 2 + gap + margin * 2
    canvas_h = title_h + original_panel.height + gap + row_h * len(rows) + margin

    canvas = Image.new("RGB", (canvas_w, canvas_h), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(26, bold=True)
    label_font = load_font(19, bold=True)
    small_font = load_font(15)

    y = margin
    if layout == "overlay-left":
        left_caption = "left: overlay on RGB"
        right_caption = "right: superpixel map"
    else:
        left_caption = "left: superpixel map"
        right_caption = "right: overlay on RGB"
    draw.text(
        (margin, y),
        f"COCO {image_name} ({left_caption}, {right_caption})",
        fill=(20, 20, 20),
        font=title_font,
    )
    y += title_h - margin
    canvas.paste(original_panel, (margin, y))
    y += original_panel.height + gap

    for row in rows:
        status = row["status"]
        label = f"{row['title']} | {row['backend']} | used: {row['used_device']} | {status}"
        draw.text((margin, y), label, fill=(20, 20, 20), font=label_font)
        if row.get("extra"):
            draw.text((margin, y + 24), row["extra"], fill=(90, 90, 90), font=small_font)
        panel_y = y + label_h

        if row["ok"]:
            seg_img = Image.fromarray(colorize_labels(row["labels"]))
            overlay_img = Image.fromarray(overlay_boundaries(np.asarray(original), row["labels"]))
            seg_panel = resize_to_width(seg_img, panel_width)
            overlay_panel = resize_to_width(overlay_img, panel_width)
        else:
            message = row.get("error", "unavailable")
            seg_panel = placeholder_panel(panel_width, panel_h, message)
            overlay_panel = placeholder_panel(panel_width, panel_h, message)

        if layout == "overlay-left":
            left_panel = overlay_panel
            right_panel = seg_panel
        else:
            left_panel = seg_panel
            right_panel = overlay_panel

        canvas.paste(left_panel, (margin, panel_y))
        canvas.paste(right_panel, (margin + panel_width + gap, panel_y))
        draw.rectangle((margin, panel_y, margin + panel_width - 1, panel_y + panel_h - 1), outline=(210, 210, 210))
        draw.rectangle(
            (
                margin + panel_width + gap,
                panel_y,
                margin + panel_width * 2 + gap - 1,
                panel_y + panel_h - 1,
            ),
            outline=(210, 210, 210),
        )
        y += row_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    image_path = args.image or choose_image(args.coco_root, args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    original = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    image = np.array(original)
    device = resolve_device(args.device)

    rows = []
    methods = [
        spec
        for spec in METHODS
        if (not args.only_gpu or spec.gpu_candidate)
        and (args.include_controls or spec.recommended)
        and (args.include_experimental or spec.default_visualization)
    ]
    for spec in methods:
        if spec.key in {"spit_dense", "spit_dense_adjacency"} and not args.run_spit_dense:
            result = {
                "ok": False,
                "error": "skipped by default on Windows; use --run-spit-dense to try it",
                "elapsed_s": 0.0,
            }
        elif spec.key in {"spit_dense", "spit_dense_adjacency"} and args.spit_dense_mode == "direct":
            result = run_without_timeout(spec, image, device)
        elif spec.key in {"spit_dense", "spit_dense_adjacency"}:
            result = run_with_timeout(spec, image, device, args.timeout)
        else:
            result = run_without_timeout(spec, image, device)
        row = {
            "key": spec.key,
            "title": spec.title,
            "backend": spec.backend,
            "strict_gpu": spec.strict_gpu,
            "recommended": spec.recommended,
            "default_visualization": spec.default_visualization,
            "used_device": device
            if spec.key
            in {
                "color_affinity",
                "gradient_affinity",
                "color_gradient_affinity",
                "torch_slic",
                "torch_grid",
                "torch_voronoi",
                "spit_voronoi",
                "spit_dense",
                "spit_dense_adjacency",
            }
            else "cpu",
            "ok": bool(result["ok"]),
            "elapsed_s": round(float(result.get("elapsed_s", 0.0)), 3),
            "status": "ok",
        }
        if result["ok"]:
            row["labels"] = result["labels"]
            extra = f"{int(np.unique(result['labels']).size)} components, {row['elapsed_s']}s"
            row["extra"] = f"{extra}; {spec.note}" if spec.note else extra
        else:
            row["error"] = result.get("error", "unknown error")
            row["status"] = "unavailable"
            extra = f"{row['error']}, {row['elapsed_s']}s"
            row["extra"] = f"{extra}; {spec.note}" if spec.note else extra
        rows.append(row)
        print(f"{spec.key}: {row['status']} ({row['extra']})")

    original_copy = output_dir / f"selected_original_{image_path.stem}.jpg"
    original.save(original_copy)
    output_path = output_dir / f"superpixel_methods_{image_path.stem}.png"
    make_grid(original, image_path.name, rows, output_path, args.panel_width, args.layout)

    metadata = []
    for row in rows:
        clean = dict(row)
        clean.pop("labels", None)
        metadata.append(clean)
    metadata_path = output_dir / f"superpixel_methods_{image_path.stem}.json"
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
    mp.freeze_support()
    main()
