"""Pluggable superpixel algorithms.

Classic algorithms receive one RGB image as a ``numpy.ndarray`` with shape
``H x W x 3`` and return either:

- an ``H x W`` integer label map; or
- two edge/adjacency channels with shape ``2 x H x W`` or ``H x W x 2``; or
- a five-channel tensor/array where channels 3 and 4 are already the extra channels.

Torch-native algorithms may additionally implement ``edges_for_batch``. That path
receives a ``B x 3 x H x W`` tensor and returns ``B x 2 x H x W`` edge channels
without leaving the tensor device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
import math
from pathlib import Path
import sys
from typing import Any, Callable, Protocol, Sequence

import numpy as np


class SuperpixelAlgorithm(Protocol):
    """Protocol for custom superpixel or edge-channel algorithms."""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return labels or edge channels for one RGB image."""


class TensorSuperpixelAlgorithm(Protocol):
    """Protocol for algorithms that can build edge channels directly from tensors."""

    def edges_for_batch(
        self,
        batch_tensor: Any,
        channel_order: Sequence[str] = ("right", "down"),
    ) -> Any:
        """Return ``B x 2 x H x W`` edge channels for a ``B x 3 x H x W`` tensor."""


def _ensure_hwc_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an HxWx3 RGB image, got shape {image.shape}.")
    return image


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = _ensure_hwc_rgb(image)
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)

    image = image.astype(np.float32, copy=False)
    if image.max() <= 1.5 and image.min() >= -0.5:
        image = image.clip(0.0, 1.0) * 255.0
    else:
        image = image.clip(0.0, 255.0)
    return np.ascontiguousarray(image.astype(np.uint8))


def _normalize_torch_image_batch(batch_tensor):
    """Return a detached float RGB batch suitable for non-trainable preprocessing."""

    import torch

    if not isinstance(batch_tensor, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(batch_tensor)!r}.")
    if batch_tensor.ndim != 4 or batch_tensor.shape[1] < 3:
        raise ValueError(f"Expected a BxCxHxW tensor with at least 3 channels, got {batch_tensor.shape}.")
    return batch_tensor[:, :3].detach().float().clamp(0.0, 1.0)


def _select_ordered_edges(edges, channel_order: Sequence[str]):
    if tuple(channel_order) == ("right", "down"):
        return edges
    if tuple(channel_order) == ("down", "right"):
        return edges[:, [1, 0]]
    raise ValueError("channel_order must be ('right', 'down') or ('down', 'right').")


def _torch_output_to_batched_edges(output, height: int, width: int, channel_order: Sequence[str]):
    """Normalize torch output to ``B x 2 x H x W`` edge channels."""

    import torch

    if isinstance(output, (list, tuple)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor output, got {type(output)!r}.")

    if output.ndim == 3:
        if output.shape[-2:] == (height, width) and (output.shape[0] == 2 or output.shape[0] >= 5):
            output = output.unsqueeze(0)
        elif output.shape[:2] == (height, width) and (output.shape[-1] == 2 or output.shape[-1] >= 5):
            output = output.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported unbatched tensor output shape {tuple(output.shape)}.")

    if output.ndim != 4:
        raise ValueError(f"Expected 3D or 4D tensor output, got shape {tuple(output.shape)}.")

    if output.shape[-2:] == (height, width):
        if output.shape[1] == 2:
            return _select_ordered_edges(output, channel_order)
        if output.shape[1] >= 5:
            return _select_ordered_edges(output[:, 3:5], channel_order)

    if output.shape[1:3] == (height, width):
        if output.shape[-1] == 2:
            return _select_ordered_edges(output.permute(0, 3, 1, 2).contiguous(), channel_order)
        if output.shape[-1] >= 5:
            edges = output[..., 3:5].permute(0, 3, 1, 2).contiguous()
            return _select_ordered_edges(edges, channel_order)

    raise ValueError(
        "Torch superpixel output must be Bx2xHxW, Bx5xHxW, BxHxWx2, BxHxWx5, "
        f"or an unbatched 2/5-channel tensor. Got shape {tuple(output.shape)}; "
        f"expected H={height}, W={width}."
    )


@dataclass
class CallableSuperpixelAlgorithm:
    """Wrap any callable so it follows the superpixel algorithm protocol."""

    fn: Callable[[np.ndarray], np.ndarray]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return np.asarray(self.fn(_ensure_hwc_rgb(image)))


@dataclass
class SLICSuperpixelAlgorithm:
    """SLIC adapter backed by ``skimage.segmentation.slic``.

    ``scikit-image`` is imported lazily, so the package can still be used with other
    algorithms when the SLIC extra is not installed.
    """

    n_segments: int = 300
    compactness: float = 10.0
    sigma: float = 1.0
    start_label: int = 0
    slic_zero: bool = False
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = _ensure_hwc_rgb(image)
        try:
            from skimage.segmentation import slic
        except ImportError as exc:
            raise ImportError(
                "SLICSuperpixelAlgorithm requires scikit-image. Install with "
                "`pip install yolo-superpixels[slic]` or `pip install scikit-image`."
            ) from exc

        return np.asarray(
            slic(
                image,
                n_segments=self.n_segments,
                compactness=self.compactness,
                sigma=self.sigma,
                start_label=self.start_label,
                slic_zero=self.slic_zero,
                channel_axis=-1,
                **self.extra_kwargs,
            )
        )


@dataclass
class OpenCVSuperpixelAlgorithm:
    """OpenCV ximgproc superpixel adapter.

    OpenCV's Python ximgproc superpixel implementations are CPU algorithms.
    They are still useful baselines next to GPU-native methods, but selecting one
    during YOLO training will move the image to CPU for this preprocessing step.
    Install ``opencv-contrib-python`` to get ``cv2.ximgproc``.
    """

    method: str = "slico"
    n_segments: int = 300
    region_size: int | None = None
    iterations: int = 10
    ruler: float = 10.0
    ratio: float = 0.075
    num_levels: int = 4
    prior: int = 2
    histogram_bins: int = 5
    double_step: bool = False
    enforce_connectivity: bool = True
    min_element_size: int = 25
    convert_to_lab: bool = False
    blur_kernel: int = 0

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = self._prepare_image(image)
        height, width = image.shape[:2]
        method = self.method.lower()
        region_size = self._resolve_region_size(height, width)

        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "OpenCVSuperpixelAlgorithm requires opencv-contrib-python. "
                "Install with `pip install opencv-contrib-python`."
            ) from exc
        if not hasattr(cv2, "ximgproc"):
            raise ImportError(
                "Your cv2 package does not expose cv2.ximgproc. Install "
                "`opencv-contrib-python`, not only `opencv-python`."
            )

        if method in {"slic", "slico", "mslic"}:
            algorithm = {
                "slic": cv2.ximgproc.SLIC,
                "slico": cv2.ximgproc.SLICO,
                "mslic": cv2.ximgproc.MSLIC,
            }[method]
            superpixels = cv2.ximgproc.createSuperpixelSLIC(
                image,
                algorithm=algorithm,
                region_size=region_size,
                ruler=float(self.ruler),
            )
            superpixels.iterate(int(self.iterations))
            if self.enforce_connectivity:
                superpixels.enforceLabelConnectivity(int(self.min_element_size))
            return np.asarray(superpixels.getLabels(), dtype=np.int32)

        if method == "lsc":
            superpixels = cv2.ximgproc.createSuperpixelLSC(
                image,
                region_size=region_size,
                ratio=float(self.ratio),
            )
            superpixels.iterate(int(self.iterations))
            if self.enforce_connectivity:
                superpixels.enforceLabelConnectivity(int(self.min_element_size))
            return np.asarray(superpixels.getLabels(), dtype=np.int32)

        if method == "seeds":
            superpixels = cv2.ximgproc.createSuperpixelSEEDS(
                width,
                height,
                image.shape[-1],
                int(self.n_segments),
                int(self.num_levels),
                int(self.prior),
                int(self.histogram_bins),
                bool(self.double_step),
            )
            superpixels.iterate(image, int(self.iterations))
            return np.asarray(superpixels.getLabels(), dtype=np.int32)

        raise ValueError(
            "OpenCV superpixel method must be one of 'slic', 'slico', 'mslic', "
            f"'lsc', or 'seeds'. Got {self.method!r}."
        )

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        image = _to_uint8_rgb(image)
        try:
            import cv2
        except ImportError:
            return image

        if self.blur_kernel and self.blur_kernel > 1:
            kernel = int(self.blur_kernel)
            if kernel % 2 == 0:
                kernel += 1
            image = cv2.GaussianBlur(image, (kernel, kernel), 0)
        if self.convert_to_lab:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        return image

    def _resolve_region_size(self, height: int, width: int) -> int:
        if self.region_size is not None:
            return max(1, int(self.region_size))
        if self.n_segments <= 0:
            raise ValueError("n_segments must be positive.")
        return max(1, int(round(math.sqrt((height * width) / float(self.n_segments)))))


@dataclass
class TorchColorAffinityAlgorithm:
    """Fast GPU-native local color affinity baseline.

    This is not SLIC. It builds the same two right/down channels directly from
    neighboring RGB similarity and is useful as a cheap fully tensor-native
    baseline.
    """

    temperature: float = 0.1
    reduction: str = "l2"

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = _ensure_hwc_rgb(image).astype(np.float32, copy=False)
        if image.max() > 1.5:
            image = image / 255.0

        right = np.zeros(image.shape[:2], dtype=np.float32)
        down = np.zeros(image.shape[:2], dtype=np.float32)
        if image.shape[1] > 1:
            diff = image[:, :-1] - image[:, 1:]
            right[:, :-1] = np.exp(-np.linalg.norm(diff, axis=-1) / self.temperature)
        if image.shape[0] > 1:
            diff = image[:-1, :] - image[1:, :]
            down[:-1, :] = np.exp(-np.linalg.norm(diff, axis=-1) / self.temperature)
        return np.stack((right, down), axis=0)

    def edges_for_batch(
        self,
        batch_tensor,
        channel_order: Sequence[str] = ("right", "down"),
    ):
        import torch

        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")

        rgb = _normalize_torch_image_batch(batch_tensor)
        batch, _, height, width = rgb.shape
        right = torch.zeros((batch, height, width), dtype=rgb.dtype, device=rgb.device)
        down = torch.zeros((batch, height, width), dtype=rgb.dtype, device=rgb.device)

        if width > 1:
            diff = rgb[:, :, :, :-1] - rgb[:, :, :, 1:]
            if self.reduction == "l1":
                distance = diff.abs().mean(dim=1)
            elif self.reduction == "l2":
                distance = torch.sqrt(torch.sum(diff * diff, dim=1).clamp_min(1e-12))
            else:
                raise ValueError("reduction must be 'l1' or 'l2'.")
            right[:, :, :-1] = torch.exp(-distance / self.temperature)

        if height > 1:
            diff = rgb[:, :, :-1, :] - rgb[:, :, 1:, :]
            if self.reduction == "l1":
                distance = diff.abs().mean(dim=1)
            else:
                distance = torch.sqrt(torch.sum(diff * diff, dim=1).clamp_min(1e-12))
            down[:, :-1, :] = torch.exp(-distance / self.temperature)

        return _select_ordered_edges(torch.stack((right, down), dim=1), channel_order)


@dataclass
class TorchGradientAffinityAlgorithm:
    """GPU-native affinity from local gradient/texture features.

    Like ``TorchColorAffinityAlgorithm``, this returns right/down similarity
    channels directly instead of a hard superpixel label map. It is useful as a
    texture/edge-sensitive companion to color affinity.
    """

    temperature: float = 0.1
    reduction: str = "l2"
    color_weight: float = 0.0
    gradient_weight: float = 1.0

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = _ensure_hwc_rgb(image).astype(np.float32, copy=False)
        if image.max() > 1.5:
            image = image / 255.0

        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
        edges = self.edges_for_batch(tensor)
        return edges[0].detach().float().cpu().numpy()

    def edges_for_batch(
        self,
        batch_tensor,
        channel_order: Sequence[str] = ("right", "down"),
    ):
        import torch
        import torch.nn.functional as F

        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if self.reduction not in {"l1", "l2"}:
            raise ValueError("reduction must be 'l1' or 'l2'.")

        rgb = _normalize_torch_image_batch(batch_tensor)
        batch, _, height, width = rgb.shape
        luminance = (
            rgb[:, 0:1] * 0.299
            + rgb[:, 1:2] * 0.587
            + rgb[:, 2:3] * 0.114
        )
        kernel_x = luminance.new_tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
        ).unsqueeze(0) / 4.0
        kernel_y = kernel_x.transpose(-1, -2)
        padded = F.pad(luminance, (1, 1, 1, 1), mode="replicate")
        grad_x = F.conv2d(padded, kernel_x)
        grad_y = F.conv2d(padded, kernel_y)
        grad_mag = torch.sqrt((grad_x * grad_x + grad_y * grad_y).clamp_min(1e-12))

        gradient_features = torch.cat([grad_x, grad_y, grad_mag], dim=1) * float(self.gradient_weight)
        features = [gradient_features]
        if self.color_weight:
            features.insert(0, rgb * float(self.color_weight))
        feature_tensor = torch.cat(features, dim=1)

        right = torch.zeros((batch, height, width), dtype=rgb.dtype, device=rgb.device)
        down = torch.zeros((batch, height, width), dtype=rgb.dtype, device=rgb.device)

        if width > 1:
            diff = feature_tensor[:, :, :, :-1] - feature_tensor[:, :, :, 1:]
            if self.reduction == "l1":
                distance = diff.abs().mean(dim=1)
            else:
                distance = torch.sqrt(torch.sum(diff * diff, dim=1).clamp_min(1e-12))
            right[:, :, :-1] = torch.exp(-distance / self.temperature)

        if height > 1:
            diff = feature_tensor[:, :, :-1, :] - feature_tensor[:, :, 1:, :]
            if self.reduction == "l1":
                distance = diff.abs().mean(dim=1)
            else:
                distance = torch.sqrt(torch.sum(diff * diff, dim=1).clamp_min(1e-12))
            down[:, :-1, :] = torch.exp(-distance / self.temperature)

        return _select_ordered_edges(torch.stack((right, down), dim=1), channel_order)


@dataclass
class TorchSLICSuperpixelAlgorithm:
    """SLIC-like superpixels implemented with batched PyTorch tensor ops.

    This is designed for experiments where CPU ``skimage.slic`` is too expensive
    inside the YOLO forward pass. It follows the usual SLIC idea of local
    color/spatial clustering around a regular grid of centers, but it is not a
    bit-for-bit replacement for scikit-image SLIC.
    """

    n_segments: int = 300
    compactness: float = 1.0
    num_iters: int = 7

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = _ensure_hwc_rgb(image).astype(np.float32, copy=False)
        if image.max() > 1.5:
            image = image / 255.0

        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
        edges = self.edges_for_batch(tensor)
        return edges[0].detach().float().cpu().numpy()

    def edges_for_batch(
        self,
        batch_tensor,
        channel_order: Sequence[str] = ("right", "down"),
    ):
        import torch

        if self.n_segments <= 0:
            raise ValueError("n_segments must be positive.")
        if self.num_iters <= 0:
            raise ValueError("num_iters must be positive.")

        with torch.no_grad():
            rgb = _normalize_torch_image_batch(batch_tensor)
            labels = self._segment_batch(rgb)
            return self._labels_to_edges(labels, batch_tensor.dtype, channel_order)

    def labels_for_batch(self, batch_tensor):
        """Return ``B x H x W`` SLIC-like labels for tensor-native target construction."""

        import torch

        rgb = _normalize_torch_image_batch(batch_tensor)
        with torch.no_grad():
            return self._segment_batch(rgb).long()

    def _segment_batch(self, rgb):
        import torch

        batch, channels, height, width = rgb.shape
        num_pixels = height * width
        step = (num_pixels / float(self.n_segments)) ** 0.5
        grid_h = max(1, int(round(height / step)))
        grid_w = max(1, int(round(width / step)))
        num_centers = grid_h * grid_w

        device = rgb.device
        dtype = rgb.dtype
        y_grid = torch.arange(height, dtype=dtype, device=device)
        x_grid = torch.arange(width, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
        y_flat = yy.reshape(-1)
        x_flat = xx.reshape(-1)

        center_y_1d = (torch.arange(grid_h, dtype=dtype, device=device) + 0.5) * height / grid_h - 0.5
        center_x_1d = (torch.arange(grid_w, dtype=dtype, device=device) + 0.5) * width / grid_w - 0.5
        center_yy, center_xx = torch.meshgrid(center_y_1d, center_x_1d, indexing="ij")
        center_y = center_yy.reshape(1, -1).expand(batch, -1).contiguous()
        center_x = center_xx.reshape(1, -1).expand(batch, -1).contiguous()

        sample_y = center_y[0].round().long().clamp(0, height - 1)
        sample_x = center_x[0].round().long().clamp(0, width - 1)
        sample_index = sample_y * width + sample_x
        rgb_flat = rgb.reshape(batch, channels, -1)
        center_color = torch.gather(
            rgb_flat,
            2,
            sample_index.view(1, 1, -1).expand(batch, channels, num_centers),
        ).contiguous()

        base_cell_y = torch.div(
            torch.arange(height, device=device) * grid_h,
            height,
            rounding_mode="floor",
        ).clamp_max(grid_h - 1)
        base_cell_x = torch.div(
            torch.arange(width, device=device) * grid_w,
            width,
            rounding_mode="floor",
        ).clamp_max(grid_w - 1)
        base_y_map = base_cell_y[:, None].expand(height, width)
        base_x_map = base_cell_x[None, :].expand(height, width)

        labels = torch.zeros((batch, num_pixels), dtype=torch.long, device=device)
        spatial_scale = max(step, 1.0)
        spatial_weight = (self.compactness / 10.0) ** 2

        for _ in range(self.num_iters):
            best_distance = torch.full((batch, num_pixels), float("inf"), dtype=dtype, device=device)
            best_labels = labels

            for offset_y in (-1, 0, 1):
                cand_y = (base_y_map + offset_y).clamp(0, grid_h - 1)
                for offset_x in (-1, 0, 1):
                    cand_x = (base_x_map + offset_x).clamp(0, grid_w - 1)
                    cand_index = (cand_y * grid_w + cand_x).reshape(-1).long()
                    gather_index_color = cand_index.view(1, 1, -1).expand(batch, channels, num_pixels)
                    gather_index = cand_index.view(1, -1).expand(batch, num_pixels)

                    cand_color = torch.gather(center_color, 2, gather_index_color)
                    color_distance = ((rgb_flat - cand_color) ** 2).mean(dim=1)

                    cand_y_pos = torch.gather(center_y, 1, gather_index)
                    cand_x_pos = torch.gather(center_x, 1, gather_index)
                    spatial_distance = ((y_flat - cand_y_pos) ** 2 + (x_flat - cand_x_pos) ** 2) / (
                        spatial_scale * spatial_scale
                    )
                    distance = color_distance + spatial_weight * spatial_distance
                    better = distance < best_distance
                    best_distance = torch.where(better, distance, best_distance)
                    best_labels = torch.where(better, gather_index, best_labels)

            labels = best_labels
            counts = torch.zeros((batch, num_centers), dtype=dtype, device=device)
            ones = torch.ones((batch, num_pixels), dtype=dtype, device=device)
            counts.scatter_add_(1, labels, ones)
            safe_counts = counts.clamp_min(1.0)

            new_color = torch.zeros((batch, channels, num_centers), dtype=dtype, device=device)
            new_color.scatter_add_(2, labels.unsqueeze(1).expand(batch, channels, num_pixels), rgb_flat)
            new_color = new_color / safe_counts.unsqueeze(1)

            new_y = torch.zeros((batch, num_centers), dtype=dtype, device=device)
            new_x = torch.zeros((batch, num_centers), dtype=dtype, device=device)
            new_y.scatter_add_(1, labels, y_flat.view(1, -1).expand(batch, num_pixels))
            new_x.scatter_add_(1, labels, x_flat.view(1, -1).expand(batch, num_pixels))
            new_y = new_y / safe_counts
            new_x = new_x / safe_counts

            valid = counts > 0
            center_color = torch.where(valid.unsqueeze(1), new_color, center_color)
            center_y = torch.where(valid, new_y, center_y)
            center_x = torch.where(valid, new_x, center_x)

        return labels.view(batch, height, width)

    @staticmethod
    def _labels_to_edges(labels, dtype, channel_order: Sequence[str]):
        import torch

        batch, height, width = labels.shape
        right = torch.zeros((batch, height, width), dtype=dtype, device=labels.device)
        down = torch.zeros((batch, height, width), dtype=dtype, device=labels.device)
        if width > 1:
            right[:, :, :-1] = labels[:, :, :-1] == labels[:, :, 1:]
        if height > 1:
            down[:, :-1, :] = labels[:, :-1, :] == labels[:, 1:, :]
        return _select_ordered_edges(torch.stack((right, down), dim=1), channel_order)


@dataclass
class TorchGridSuperpixelAlgorithm:
    """Strict GPU-native regular grid superpixel baseline."""

    n_segments: int = 300

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = _ensure_hwc_rgb(image).astype(np.float32, copy=False)
        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
        edges = self.edges_for_batch(tensor)
        return edges[0].detach().float().cpu().numpy()

    def edges_for_batch(
        self,
        batch_tensor,
        channel_order: Sequence[str] = ("right", "down"),
    ):
        if self.n_segments <= 0:
            raise ValueError("n_segments must be positive.")

        rgb = _normalize_torch_image_batch(batch_tensor)
        labels = self._segment_batch(rgb)
        return TorchSLICSuperpixelAlgorithm._labels_to_edges(labels, batch_tensor.dtype, channel_order)

    def labels_for_batch(self, batch_tensor):
        """Return ``B x H x W`` grid labels for tensor-native target construction."""

        rgb = _normalize_torch_image_batch(batch_tensor)
        return self._segment_batch(rgb).long()

    def _segment_batch(self, rgb):
        import torch

        batch, _, height, width = rgb.shape
        device = rgb.device
        num_pixels = height * width
        step = (num_pixels / float(self.n_segments)) ** 0.5
        grid_h = max(1, int(round(height / step)))
        grid_w = max(1, int(round(width / step)))

        y = torch.div(torch.arange(height, device=device) * grid_h, height, rounding_mode="floor")
        x = torch.div(torch.arange(width, device=device) * grid_w, width, rounding_mode="floor")
        labels = y[:, None] * grid_w + x[None, :]
        return labels.long().unsqueeze(0).expand(batch, -1, -1)


@dataclass
class TorchVoronoiSuperpixelAlgorithm:
    """Strict GPU-native deterministic spatial Voronoi baseline.

    This is a spatial partition, not an image-adaptive superpixel method. It is
    useful as a no-CPU-transfer irregular baseline and as a companion to
    content-aware methods.
    """

    n_segments: int = 300
    jitter: float = 0.35
    seed: int = 0
    chunk_size: int = 65536

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = _ensure_hwc_rgb(image).astype(np.float32, copy=False)
        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
        edges = self.edges_for_batch(tensor)
        return edges[0].detach().float().cpu().numpy()

    def edges_for_batch(
        self,
        batch_tensor,
        channel_order: Sequence[str] = ("right", "down"),
    ):
        if self.n_segments <= 0:
            raise ValueError("n_segments must be positive.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        rgb = _normalize_torch_image_batch(batch_tensor)
        labels = self._segment_batch(rgb)
        return TorchSLICSuperpixelAlgorithm._labels_to_edges(labels, batch_tensor.dtype, channel_order)

    def labels_for_batch(self, batch_tensor):
        """Return ``B x H x W`` Voronoi labels for tensor-native target construction."""

        rgb = _normalize_torch_image_batch(batch_tensor)
        return self._segment_batch(rgb).long()

    def _segment_batch(self, rgb):
        import torch

        batch, _, height, width = rgb.shape
        device = rgb.device
        dtype = rgb.dtype
        num_pixels = height * width
        step = (num_pixels / float(self.n_segments)) ** 0.5
        grid_h = max(1, int(round(height / step)))
        grid_w = max(1, int(round(width / step)))
        num_centers = grid_h * grid_w

        center_y_1d = (torch.arange(grid_h, dtype=dtype, device=device) + 0.5) * height / grid_h - 0.5
        center_x_1d = (torch.arange(grid_w, dtype=dtype, device=device) + 0.5) * width / grid_w - 0.5
        center_yy, center_xx = torch.meshgrid(center_y_1d, center_x_1d, indexing="ij")
        center_y = center_yy.reshape(-1)
        center_x = center_xx.reshape(-1)

        if self.jitter:
            idx = torch.arange(num_centers, dtype=dtype, device=device)
            jitter_y = torch.remainder(
                torch.sin(idx * 12.9898 + float(self.seed)) * 43758.5453,
                1.0,
            ) - 0.5
            jitter_x = torch.remainder(
                torch.sin(idx * 78.233 + float(self.seed) * 1.37) * 24634.6345,
                1.0,
            ) - 0.5
            center_y = (center_y + jitter_y * float(self.jitter) * height / grid_h).clamp(0, height - 1)
            center_x = (center_x + jitter_x * float(self.jitter) * width / grid_w).clamp(0, width - 1)

        y_grid = torch.arange(height, dtype=dtype, device=device)
        x_grid = torch.arange(width, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
        coords_y = yy.reshape(-1)
        coords_x = xx.reshape(-1)
        labels_flat = torch.empty(num_pixels, dtype=torch.long, device=device)

        for start in range(0, num_pixels, int(self.chunk_size)):
            end = min(start + int(self.chunk_size), num_pixels)
            dy = coords_y[start:end, None] - center_y[None, :]
            dx = coords_x[start:end, None] - center_x[None, :]
            labels_flat[start:end] = (dy * dy + dx * dx).argmin(dim=1)

        labels = labels_flat.view(height, width)
        return labels.unsqueeze(0).expand(batch, -1, -1)


@dataclass
class SPiTPseudoVoronoiAlgorithm:
    """SPiT/RViT pseudo-Voronoi baseline built from PCA-tree partitions.

    This is not a content-aware superpixel method. It mirrors the online
    irregular partition baseline in SPiT/RViT and stays tensor-native when SPiT is
    importable. It is useful to test whether irregular partitions alone help.
    """

    num_cells: int = 256

    def __post_init__(self) -> None:
        SPiTDenseEdgeAlgorithm._ensure_local_spit_on_path()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = _ensure_hwc_rgb(image).astype(np.float32, copy=False)
        if image.max() > 1.5:
            image = image / 255.0

        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
        edges = self.edges_for_batch(tensor)
        return edges[0].detach().float().cpu().numpy()

    def edges_for_batch(
        self,
        batch_tensor,
        channel_order: Sequence[str] = ("right", "down"),
    ):
        import torch
        from spit.tokenizer.voronoi import fast_pseudo_voronoi

        if self.num_cells <= 0:
            raise ValueError("num_cells must be positive.")

        rgb = _normalize_torch_image_batch(batch_tensor)
        with torch.no_grad():
            labels = fast_pseudo_voronoi(rgb, int(self.num_cells))
        return TorchSLICSuperpixelAlgorithm._labels_to_edges(labels, batch_tensor.dtype, channel_order)

    def labels_for_batch(self, batch_tensor):
        """Return ``B x H x W`` pseudo-Voronoi labels for tensor-native target construction."""

        import torch
        from spit.tokenizer.voronoi import fast_pseudo_voronoi

        rgb = _normalize_torch_image_batch(batch_tensor)
        with torch.no_grad():
            return fast_pseudo_voronoi(rgb, int(self.num_cells)).long()


@dataclass
class SPiTDenseEdgeAlgorithm:
    """Adapter for SPiT-style dense edge embedders.

    The old repository used ``DenseSPEdgeEmbedder`` and received a five-channel
    tensor back. This class keeps that path optional: provide an already created
    embedder, or let the adapter import it from ``module_path``.
    """

    embedder: Any | None = None
    module_path: str = "spit.tokenizer.densesp"
    class_name: str = "DenseSPEdgeEmbedder"
    device: str = "auto"
    _embedder_device: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.embedder is not None:
            self._prepare_embedder_for_device(None)
            return
        self._ensure_local_spit_on_path()
        module = import_module(self.module_path)
        embedder_cls = getattr(module, self.class_name)
        self.embedder = embedder_cls()
        self._prepare_embedder_for_device(None)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = _ensure_hwc_rgb(image)
        try:
            import torch
        except ImportError as exc:
            raise ImportError("SPiTDenseEdgeAlgorithm requires torch.") from exc

        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
        if tensor.max() > 1.5:
            tensor = tensor / 255.0
        edges = self.edges_for_batch(tensor.unsqueeze(0))
        return edges[0].detach().float().cpu().numpy()

    def edges_for_batch(
        self,
        batch_tensor,
        channel_order: Sequence[str] = ("right", "down"),
    ):
        """Run SPiT on a tensor batch and keep the result on the input device."""

        import torch

        rgb = _normalize_torch_image_batch(batch_tensor)
        input_device = rgb.device
        target_device = self._resolve_device(input_device)
        if target_device is not None:
            rgb = rgb.to(target_device, non_blocking=True)
        self._prepare_embedder_for_device(target_device)

        with torch.no_grad():
            output = self.embedder(rgb)

        edges = _torch_output_to_batched_edges(
            output,
            height=int(batch_tensor.shape[-2]),
            width=int(batch_tensor.shape[-1]),
            channel_order=channel_order,
        )
        return edges.to(device=input_device, dtype=batch_tensor.dtype, non_blocking=True)

    def _resolve_device(self, input_device):
        import torch

        if self.device in {"auto", "same", "input"}:
            return input_device
        return torch.device(self.device)

    def _prepare_embedder_for_device(self, input_device) -> None:
        if self.embedder is None:
            return

        target_device = self._resolve_device(input_device)
        if target_device is not None and hasattr(self.embedder, "to") and self._embedder_device != str(target_device):
            self.embedder = self.embedder.to(target_device)
            self._embedder_device = str(target_device)
        if hasattr(self.embedder, "eval"):
            self.embedder.eval()

    @staticmethod
    def _ensure_local_spit_on_path() -> None:
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "third_party" / "SPiT"
            if (candidate / "spit" / "tokenizer" / "densesp.py").exists():
                candidate_str = str(candidate)
                if candidate_str not in sys.path:
                    sys.path.insert(0, candidate_str)
                return


@dataclass
class SPiTDenseAdjacencyAlgorithm(SPiTDenseEdgeAlgorithm):
    """Convert SPiT dense segment labels into right/down adjacency channels.

    ``SPiTDenseEdgeAlgorithm`` exposes the raw ``DenseSPEdgeEmbedder`` channels,
    whose scale and meaning differ from the binary adjacency channels used by
    SLIC-like methods. This variant requests SPiT's segmentation labels and
    returns the same two-channel representation as label-based superpixels:
    channel 0 is right-neighbor sameness, channel 1 is down-neighbor sameness.
    """

    def __post_init__(self) -> None:
        if self.embedder is not None:
            self._prepare_embedder_for_device(None)
            return
        self._ensure_local_spit_on_path()
        module = import_module(self.module_path)
        embedder_cls = getattr(module, self.class_name)
        try:
            self.embedder = embedder_cls(return_seg=True)
        except TypeError:
            self.embedder = embedder_cls()
        self._prepare_embedder_for_device(None)

    def edges_for_batch(
        self,
        batch_tensor,
        channel_order: Sequence[str] = ("right", "down"),
    ):
        """Run SPiT and convert returned segmentation labels to adjacency."""

        import torch

        rgb = _normalize_torch_image_batch(batch_tensor)
        input_device = rgb.device
        target_device = self._resolve_device(input_device)
        if target_device is not None:
            rgb = rgb.to(target_device, non_blocking=True)
        self._prepare_embedder_for_device(target_device)

        with torch.no_grad():
            output = self.embedder(rgb)

        labels = self._extract_labels(
            output,
            height=int(batch_tensor.shape[-2]),
            width=int(batch_tensor.shape[-1]),
        )
        labels = labels.to(device=input_device, non_blocking=True).long()
        edges = TorchSLICSuperpixelAlgorithm._labels_to_edges(labels, batch_tensor.dtype, channel_order)
        return edges.to(device=input_device, dtype=batch_tensor.dtype, non_blocking=True)

    def labels_for_batch(self, batch_tensor):
        """Run SPiT and return segmentation labels without converting them to edges."""

        import torch

        rgb = _normalize_torch_image_batch(batch_tensor)
        input_device = rgb.device
        target_device = self._resolve_device(input_device)
        if target_device is not None:
            rgb = rgb.to(target_device, non_blocking=True)
        self._prepare_embedder_for_device(target_device)

        with torch.no_grad():
            output = self.embedder(rgb)

        labels = self._extract_labels(
            output,
            height=int(batch_tensor.shape[-2]),
            width=int(batch_tensor.shape[-1]),
        )
        return labels.to(device=input_device, non_blocking=True).long()

    @staticmethod
    def _extract_labels(output, height: int, width: int):
        import torch

        candidates: list[Any]
        if isinstance(output, dict):
            candidates = [
                output.get(key)
                for key in ("seg", "segments", "labels", "label", "superpixels")
            ]
        elif isinstance(output, (list, tuple)):
            candidates = list(output[1:]) + list(output[:1])
        else:
            candidates = [output]

        for candidate in candidates:
            if candidate is None or not isinstance(candidate, torch.Tensor):
                continue
            labels = candidate
            if labels.ndim == 4 and labels.shape[1] == 1:
                labels = labels[:, 0]
            elif labels.ndim == 4 and labels.shape[-1] == 1:
                labels = labels[..., 0]
            if labels.ndim == 3 and labels.shape[-2:] == (height, width):
                return labels
            if labels.ndim == 2 and labels.shape == (height, width):
                return labels.unsqueeze(0)

        shapes = [
            tuple(candidate.shape)
            for candidate in candidates
            if isinstance(candidate, torch.Tensor)
        ]
        raise ValueError(
            "SPiT adjacency mode needs a segmentation label tensor shaped "
            f"BxHxW or HxW. Got tensor shapes: {shapes}."
        )
