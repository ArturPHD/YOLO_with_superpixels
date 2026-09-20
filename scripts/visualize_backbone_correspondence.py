"""Visualize P3-cell to superpixel overlap and the two correspondence routes."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from yolo_superpixels.strategies.backbone_correspondence import build_overlap_edges
from yolo_superpixels.superpixels.algorithms import TorchSLICSuperpixelAlgorithm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Input RGB image path.")
    parser.add_argument("--output", default="visualizations/backbone_correspondence.png")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--p3-stride", type=int, default=8)
    parser.add_argument("--segments", type=int, default=300)
    parser.add_argument("--compactness", type=float, default=1.0)
    parser.add_argument("--iters", type=int, default=7)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _boundaries(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(labels, dtype=bool)
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    return boundary


def _overlay_boundaries(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    output = image.copy()
    output[_boundaries(labels)] = np.array([255, 55, 55], dtype=np.uint8)
    return output


def _draw_route(axis, placement: str) -> None:
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 7)
    axis.axis("off")
    axis.set_title("Native backbone" if placement == "native_backbone" else "Isolated P3", fontsize=13, weight="bold")
    boxes = {
        "rgb": (0.3, 3.0, 1.6, 1.0, "RGB"),
        "spit": (2.6, 5.0, 2.0, 1.0, "SPiT tokens"),
        "p3": (2.6, 2.5, 2.0, 1.0, "P3 CNN cells"),
        "fusion": (5.3, 3.8, 2.3, 1.0, "Local overlap\nattention"),
        "p4": (8.0, 5.3, 1.5, 0.8, "P4"),
        "p5": (8.0, 4.0, 1.5, 0.8, "P5"),
        "neck": (8.0, 1.9, 1.5, 0.8, "Neck"),
    }
    for key, (x, y, width, height, label) in boxes.items():
        color = "#e8f0fe" if key in {"rgb", "p3"} else "#e8f7ed" if key == "spit" else "#fff1d6"
        axis.add_patch(Rectangle((x, y), width, height, facecolor=color, edgecolor="#3d4b5c", linewidth=1.4))
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9)

    def arrow(start, end):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, linewidth=1.4, color="#384b5f"))

    arrow((1.9, 3.5), (2.6, 3.0))
    arrow((1.9, 3.5), (2.6, 5.0))
    arrow((4.6, 5.5), (5.3, 4.8))
    arrow((4.6, 3.0), (5.3, 4.0))
    if placement == "native_backbone":
        arrow((7.6, 4.5), (8.0, 5.7))
        arrow((7.6, 4.3), (8.0, 4.4))
        arrow((7.6, 4.0), (8.0, 2.3))
        axis.text(5.2, 1.0, "Enriched P3 continues into P4, P5 and the P3 neck skip.", ha="left", fontsize=9)
    else:
        arrow((4.6, 3.0), (8.0, 5.7))
        arrow((4.6, 3.0), (8.0, 4.4))
        arrow((7.6, 4.1), (8.0, 2.3))
        axis.text(5.2, 1.0, "Original P3 makes P4/P5; only the P3 neck skip is enriched.", ha="left", fontsize=9)


def main() -> None:
    args = parse_args()
    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(args.image)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
    device = torch.device(args.device)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
    algorithm = TorchSLICSuperpixelAlgorithm(
        n_segments=args.segments,
        compactness=args.compactness,
        num_iters=args.iters,
    )
    with torch.no_grad():
        labels = algorithm._segment_batch(tensor).long() + 1
        target_size = args.imgsz // args.p3_stride
        edges = build_overlap_edges(labels, (target_size, target_size), int(labels.max()) + 1)
        edge_counts = torch.bincount(edges.cell_index, minlength=target_size * target_size)
        focus_cell = int(edge_counts.argmax().cpu())

    label_map = labels[0].cpu().numpy()
    row, col = divmod(focus_cell, target_size)
    cell_size = args.p3_stride
    top, left = row * cell_size, col * cell_size
    selected_edges = edges.cell_index == focus_cell
    selected_regions = edges.region_index[selected_edges].cpu().numpy()
    selected_areas = edges.area[selected_edges].cpu().numpy()
    overlay = _overlay_boundaries(image, label_map)
    overlay = (0.72 * overlay + 0.28 * image).astype(np.uint8)
    for region in selected_regions:
        overlay[label_map == region] = (0.55 * overlay[label_map == region] + 0.45 * np.array([60, 190, 90])).astype(np.uint8)
    cv2.rectangle(overlay, (left, top), (left + cell_size - 1, top + cell_size - 1), (30, 80, 255), 2)

    crop_radius = 64
    y0, y1 = max(0, top - crop_radius), min(args.imgsz, top + cell_size + crop_radius)
    x0, x1 = max(0, left - crop_radius), min(args.imgsz, left + cell_size + crop_radius)
    figure, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original image", fontsize=13, weight="bold")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title(
        f"P3 cell ({row}, {col}) and intersecting SLIC regions", fontsize=13, weight="bold"
    )
    axes[0, 1].axis("off")
    axes[1, 0].imshow(overlay[y0:y1, x0:x1])
    axes[1, 0].set_title("Zoomed correspondence support", fontsize=13, weight="bold")
    axes[1, 0].axis("off")
    area_text = "\n".join(f"region {region}: {100 * area:.1f}%" for region, area in zip(selected_regions, selected_areas))
    axes[1, 0].text(0.02, 0.02, area_text or "one region", transform=axes[1, 0].transAxes, fontsize=10, color="white", bbox={"facecolor": "black", "alpha": 0.65, "pad": 4})
    _draw_route(axes[1, 1], "native_backbone")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)

    isolated_output = output.with_name(f"{output.stem}_isolated{output.suffix}")
    figure, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    _draw_route(axis, "isolated_p3")
    figure.savefig(isolated_output, dpi=180)
    plt.close(figure)
    print(f"device={device}; selected_p3_cell={row},{col}; edges={len(selected_regions)}")
    print(f"wrote={output}")
    print(f"wrote={isolated_output}")


if __name__ == "__main__":
    main()
