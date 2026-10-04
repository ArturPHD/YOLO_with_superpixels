"""Ultralytics YOLO integration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from ..strategies.input_channels.patch import (
    PatchResult,
    infer_first_conv_in_channels,
    patch_first_conv2d,
)
from ..superpixels.adjacency import (
    DISTANCE_CHANNEL_MODES,
    AdjacencyChannelBuilder,
    DistanceChannelBuilder,
    distance_mode_extra_channels,
)
from ..superpixels.algorithms import (
    OpenCVSuperpixelAlgorithm,
    QStarSuperpixelAlgorithm,
    SLICSuperpixelAlgorithm,
    SPiTDenseAdjacencyAlgorithm,
    SPiTDenseEdgeAlgorithm,
    SPiTPseudoVoronoiAlgorithm,
    SuperpixelAlgorithm,
    TorchColorAffinityAlgorithm,
    TorchGradientAffinityAlgorithm,
    TorchGridSuperpixelAlgorithm,
    TorchSLICSuperpixelAlgorithm,
    TorchVoronoiSuperpixelAlgorithm,
)
from .run_config import write_superpixel_run_config

ENV_PREFIX = "YOLO_SUPERPIXELS_"
_ACTIVE_ADAPTER: "SuperpixelYOLOAdapter | None" = None
SUPPORTED_ALGORITHMS = (
    "slic",
    "qstar",
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
)
CHANNEL_MODES = ("adjacency", *DISTANCE_CHANNEL_MODES)


def _env_key(name: str) -> str:
    return f"{ENV_PREFIX}{name}"


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _build_algorithm_from_name(name: str, kwargs: dict[str, Any]) -> SuperpixelAlgorithm:
    name = name.lower()
    kwargs = dict(kwargs)
    if name == "slic":
        return SLICSuperpixelAlgorithm(**kwargs)
    if name == "qstar":
        return QStarSuperpixelAlgorithm(**kwargs)
    if name in {"torch_slic", "gpu_slic"}:
        return TorchSLICSuperpixelAlgorithm(**kwargs)
    if name == "spit":
        return SPiTDenseEdgeAlgorithm(**kwargs)
    if name in {"spit_dense_adjacency", "spit_adjacency"}:
        return SPiTDenseAdjacencyAlgorithm(**kwargs)
    if name in {"spit_voronoi", "pseudo_voronoi"}:
        return SPiTPseudoVoronoiAlgorithm(**kwargs)
    if name in {"torch_grid", "gpu_grid"}:
        return TorchGridSuperpixelAlgorithm(**kwargs)
    if name in {"torch_voronoi", "gpu_voronoi"}:
        return TorchVoronoiSuperpixelAlgorithm(**kwargs)
    if name in {"color_affinity", "gpu_affinity"}:
        return TorchColorAffinityAlgorithm(**kwargs)
    if name in {"gradient_affinity", "texture_affinity"}:
        return TorchGradientAffinityAlgorithm(**kwargs)
    if name == "color_gradient_affinity":
        kwargs.setdefault("color_weight", 1.0)
        kwargs.setdefault("gradient_weight", 1.0)
        return TorchGradientAffinityAlgorithm(**kwargs)
    if name.startswith("opencv_"):
        method = name.removeprefix("opencv_")
        return OpenCVSuperpixelAlgorithm(method=method, **kwargs)
    raise ValueError(
        "Unknown superpixel algorithm "
        f"'{name}'. Expected one of: {', '.join(SUPPORTED_ALGORITHMS)}."
    )


def set_active_adapter(adapter: "SuperpixelYOLOAdapter | None") -> None:
    """Set the in-process adapter used by ``SuperpixelDetectionTrainer``."""

    global _ACTIVE_ADAPTER
    _ACTIVE_ADAPTER = adapter


def adapter_from_environment() -> "SuperpixelYOLOAdapter":
    """Build an adapter from environment variables for Ultralytics DDP workers."""

    if os.environ.get(_env_key("ALGORITHMS")) is not None:
        algorithm_names = json.loads(os.environ.get(_env_key("ALGORITHMS"), "[]"))
        algorithm_kwargs = json.loads(os.environ.get(_env_key("ALGORITHM_KWARGS_LIST"), "[]"))
    else:
        algorithm_name = os.environ.get(_env_key("ALGORITHM"), "")
        algorithm_names = [] if algorithm_name.lower() in {"", "none", "rgb"} else [algorithm_name]
        algorithm_kwargs = json.loads(os.environ.get(_env_key("ALGORITHM_KWARGS"), "{}"))
    channel_order = tuple(json.loads(os.environ.get(_env_key("CHANNEL_ORDER"), '["right", "down"]')))
    target_channels_env = os.environ.get(_env_key("TARGET_CHANNELS"))
    disabled_values_raw = os.environ.get(_env_key("DISABLED_VALUES"))
    disabled_values = json.loads(disabled_values_raw) if disabled_values_raw else None

    return SuperpixelYOLOAdapter(
        algorithm_names=algorithm_names,
        algorithm_kwargs=algorithm_kwargs,
        channel_mode=os.environ.get(_env_key("CHANNEL_MODE"), "adjacency"),
        channel_order=channel_order,  # type: ignore[arg-type]
        edge_representation=os.environ.get(_env_key("EDGE_REPRESENTATION"), "zero_centered"),
        edge_value_scale=float(os.environ.get(_env_key("EDGE_VALUE_SCALE"), "1.0")),
        soft_max_distance=int(os.environ.get(_env_key("SOFT_MAX_DISTANCE"), "32")),
        soft_boundary_width=float(os.environ.get(_env_key("SOFT_BOUNDARY_WIDTH"), "6.0")),
        soft_affinity_threshold=float(os.environ.get(_env_key("SOFT_AFFINITY_THRESHOLD"), "0.5")),
        fail_open=_parse_bool(os.environ.get(_env_key("FAIL_OPEN")), False),
        extra_channels_enabled=_parse_bool(os.environ.get(_env_key("EXTRA_CHANNELS_ENABLED")), True),
        disabled_value=float(os.environ.get(_env_key("DISABLED_VALUE"), "0.0")),
        disabled_values=disabled_values,
        init_extra=os.environ.get(_env_key("INIT_EXTRA"), "tiny"),
        target_channels=int(target_channels_env) if target_channels_env else None,
        base_channels=int(os.environ.get(_env_key("BASE_CHANNELS"), "3")),
    )


@dataclass
class SuperpixelYOLOAdapter:
    """Patch YOLO models and trainers with superpixel adjacency input."""

    algorithm: SuperpixelAlgorithm | str | Sequence[SuperpixelAlgorithm | str] | None = None
    algorithm_name: str | None = None
    algorithm_names: list[str] | None = None
    algorithm_kwargs: dict[str, Any] | list[dict[str, Any]] = field(default_factory=dict)
    channel_mode: str = "adjacency"
    channel_order: tuple[str, str] = ("right", "down")
    edge_representation: str = "zero_centered"
    edge_value_scale: float = 1.0
    soft_max_distance: int = 32
    soft_boundary_width: float = 6.0
    soft_affinity_threshold: float = 0.5
    fail_open: bool = False
    extra_channels_enabled: bool = True
    disabled_value: float = 0.0
    disabled_values: Sequence[float] | None = None
    init_extra: str = "tiny"
    target_channels: int | None = None
    base_channels: int = 3
    _algorithms: tuple[SuperpixelAlgorithm, ...] = field(default=(), init=False, repr=False)
    _algorithm_kwargs_list: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.channel_mode not in CHANNEL_MODES:
            allowed = ", ".join(CHANNEL_MODES)
            raise ValueError(f"channel_mode must be one of: {allowed}.")
        algorithms, names = self._normalize_algorithms()
        if algorithms and names:
            raise ValueError(
                "Do not mix algorithm objects and algorithm names in one adapter. "
                "Use names for Ultralytics/DDP experiments."
            )
        kwargs_list = self._normalize_algorithm_kwargs(max(len(algorithms), len(names)))
        if not algorithms and names:
            algorithms = tuple(
                _build_algorithm_from_name(name, kwargs)
                for name, kwargs in zip(names, kwargs_list)
            )
        self._algorithms = tuple(algorithms)
        self._algorithm_kwargs_list = kwargs_list
        self.algorithm_names = names
        self.algorithm_name = names[0] if len(names) == 1 else None
        if len(self._algorithms) == 0:
            self.algorithm = None
        elif len(self._algorithms) == 1:
            self.algorithm = self._algorithms[0]
        else:
            self.algorithm = self._algorithms
        if self.target_channels is None:
            per_algorithm = 2
            if self.channel_mode != "adjacency":
                per_algorithm = distance_mode_extra_channels(self.channel_mode)  # type: ignore[arg-type]
            self.target_channels = self.base_channels + per_algorithm * len(self._algorithms)

    @property
    def algorithms(self) -> tuple[SuperpixelAlgorithm, ...]:
        return self._algorithms

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "algorithm_names": self.algorithm_names or [],
            "algorithm_kwargs": self._algorithm_kwargs_list,
            "channel_mode": self.channel_mode,
            "channel_order": list(self.channel_order),
            "edge_representation": self.edge_representation,
            "edge_value_scale": self.edge_value_scale,
            "soft_max_distance": self.soft_max_distance,
            "soft_boundary_width": self.soft_boundary_width,
            "soft_affinity_threshold": self.soft_affinity_threshold,
            "fail_open": self.fail_open,
            "extra_channels_enabled": self.extra_channels_enabled,
            "disabled_value": self.disabled_value,
            "disabled_values": list(self.disabled_values) if self.disabled_values is not None else None,
            "init_extra": self.init_extra,
            "target_channels": self.target_channels,
            "base_channels": self.base_channels,
        }

    def _normalize_algorithms(self) -> tuple[tuple[SuperpixelAlgorithm, ...], list[str]]:
        names: list[str] = []
        algorithms: list[SuperpixelAlgorithm] = []

        if isinstance(self.algorithm, str):
            names = [] if self.algorithm.lower() in {"none", "rgb"} else [self.algorithm]
        elif isinstance(self.algorithm, (list, tuple)):
            for item in self.algorithm:
                if isinstance(item, str):
                    if item.lower() not in {"none", "rgb"}:
                        names.append(item)
                else:
                    algorithms.append(item)
        elif self.algorithm is not None:
            algorithms.append(self.algorithm)

        if self.algorithm_names:
            names = [name for name in self.algorithm_names if name.lower() not in {"none", "rgb"}]
        elif self.algorithm_name:
            names = [] if self.algorithm_name.lower() in {"none", "rgb"} else [self.algorithm_name]

        return tuple(algorithms), names

    def _normalize_algorithm_kwargs(self, count: int) -> list[dict[str, Any]]:
        if count == 0:
            return []
        if isinstance(self.algorithm_kwargs, list):
            values = [dict(value) for value in self.algorithm_kwargs]
            if len(values) != count:
                raise ValueError(
                    f"Expected {count} algorithm kwargs entries, got {len(values)}."
                )
            return values
        value = dict(self.algorithm_kwargs)
        if count == 1:
            return [value]
        if value:
            raise ValueError(
                "Pass algorithm_kwargs as a list when using multiple algorithms."
            )
        return [{} for _ in range(count)]

    def channel_builder(self) -> AdjacencyChannelBuilder:
        if self.channel_mode != "adjacency":
            return DistanceChannelBuilder(
                algorithm=self.algorithms,
                mode=self.channel_mode,  # type: ignore[arg-type]
                max_distance=self.soft_max_distance,
                boundary_width=self.soft_boundary_width,
                affinity_threshold=self.soft_affinity_threshold,
                fail_open=self.fail_open,
                enabled=self.extra_channels_enabled,
                disabled_value=self.disabled_value,
                disabled_values=self.disabled_values,
            )
        return AdjacencyChannelBuilder(
            algorithm=self.algorithms,
            channel_order=self.channel_order,  # type: ignore[arg-type]
            edge_representation=self.edge_representation,  # type: ignore[arg-type]
            edge_value_scale=self.edge_value_scale,
            fail_open=self.fail_open,
            enabled=self.extra_channels_enabled,
            disabled_value=self.disabled_value,
            disabled_values=self.disabled_values,
        )

    def export_environment(self) -> None:
        """Export serializable adapter settings for Ultralytics DDP subprocesses."""

        names = self.algorithm_names or []
        os.environ[_env_key("ALGORITHMS")] = json.dumps(names)
        os.environ[_env_key("ALGORITHM_KWARGS_LIST")] = json.dumps(self._algorithm_kwargs_list)
        os.environ[_env_key("ALGORITHM")] = names[0] if len(names) == 1 else ("none" if not names else ",".join(names))
        os.environ[_env_key("ALGORITHM_KWARGS")] = json.dumps(
            self._algorithm_kwargs_list[0] if len(self._algorithm_kwargs_list) == 1 else {}
        )
        os.environ[_env_key("CHANNEL_MODE")] = self.channel_mode
        os.environ[_env_key("CHANNEL_ORDER")] = json.dumps(list(self.channel_order))
        os.environ[_env_key("EDGE_REPRESENTATION")] = self.edge_representation
        os.environ[_env_key("EDGE_VALUE_SCALE")] = str(self.edge_value_scale)
        os.environ[_env_key("SOFT_MAX_DISTANCE")] = str(self.soft_max_distance)
        os.environ[_env_key("SOFT_BOUNDARY_WIDTH")] = str(self.soft_boundary_width)
        os.environ[_env_key("SOFT_AFFINITY_THRESHOLD")] = str(self.soft_affinity_threshold)
        os.environ[_env_key("FAIL_OPEN")] = "1" if self.fail_open else "0"
        os.environ[_env_key("EXTRA_CHANNELS_ENABLED")] = "1" if self.extra_channels_enabled else "0"
        os.environ[_env_key("DISABLED_VALUE")] = str(self.disabled_value)
        if self.disabled_values is None:
            os.environ.pop(_env_key("DISABLED_VALUES"), None)
        else:
            os.environ[_env_key("DISABLED_VALUES")] = json.dumps(list(self.disabled_values))
        os.environ[_env_key("INIT_EXTRA")] = self.init_extra
        os.environ[_env_key("TARGET_CHANNELS")] = str(int(self.target_channels))
        os.environ[_env_key("BASE_CHANNELS")] = str(self.base_channels)

    def set_extra_channels_enabled(self, enabled: bool) -> None:
        """Enable or disable the two generated extra channels for future patches."""

        self.extra_channels_enabled = enabled

    def patch_model(self, model) -> PatchResult:
        return patch_first_conv2d(
            model,
            channel_builder=self.channel_builder(),
            target_channels=int(self.target_channels),
            base_channels=self.base_channels,
            init_extra=self.init_extra,
        )

    def prepare_training(self) -> None:
        set_active_adapter(self)
        self.export_environment()


def _load_detection_trainer_base():
    try:
        from ultralytics.models.yolo.detect import DetectionTrainer
    except ImportError as exc:
        raise ImportError(
            "SuperpixelDetectionTrainer requires ultralytics. Install with "
            "`pip install yolo-superpixels[ultralytics]` or `pip install ultralytics`."
        ) from exc
    return DetectionTrainer


try:
    DetectionTrainerBase = _load_detection_trainer_base()
    _ULTRALYTICS_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    _ULTRALYTICS_IMPORT_ERROR = exc

    class DetectionTrainerBase:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise _ULTRALYTICS_IMPORT_ERROR


class SuperpixelDetectionTrainer(DetectionTrainerBase):
    """Importable trainer class that patches the model before training.

    Keeping this class at module level makes it compatible with Ultralytics DDP,
    which imports the trainer class in child processes.
    """

    def __init__(self, *args, superpixel_adapter: SuperpixelYOLOAdapter | None = None, **kwargs) -> None:
        self.superpixel_adapter = superpixel_adapter or _ACTIVE_ADAPTER or adapter_from_environment()
        super().__init__(*args, **kwargs)
        write_superpixel_run_config(
            self.save_dir,
            strategy="input_channels",
            config=self.superpixel_adapter.to_jsonable(),
            args=self.args,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        weights_channels = infer_first_conv_in_channels(weights)
        load_after_patch = weights is not None and weights_channels == self.superpixel_adapter.target_channels

        if load_after_patch:
            model = super().get_model(cfg=cfg, weights=None, verbose=verbose)
            self.superpixel_patch = self.superpixel_adapter.patch_model(model)
            model.load(weights)
            return model

        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        self.superpixel_patch = self.superpixel_adapter.patch_model(model)
        return model


class SuperpixelYOLO:
    """Small wrapper around ``ultralytics.YOLO`` with superpixel training enabled."""

    def __init__(self, model: Any, adapter: SuperpixelYOLOAdapter | None = None, **adapter_kwargs) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "SuperpixelYOLO requires ultralytics. Install with "
                "`pip install yolo-superpixels[ultralytics]` or `pip install ultralytics`."
            ) from exc

        self.yolo = model if hasattr(model, "train") and hasattr(model, "model") else YOLO(model)
        self.adapter = adapter or SuperpixelYOLOAdapter(**adapter_kwargs)

    def train(self, **kwargs):
        self.adapter.prepare_training()
        return self.yolo.train(trainer=SuperpixelDetectionTrainer, **kwargs)

    def set_extra_channels_enabled(self, enabled: bool) -> None:
        """Switch generated superpixel channels on/off for validation or prediction."""

        self.adapter.set_extra_channels_enabled(enabled)
        if getattr(self.yolo, "model", None) is not None:
            self.adapter.patch_model(self.yolo.model)

    def val(self, extra_channels_enabled: Optional[bool] = None, **kwargs):
        old_enabled = self.adapter.extra_channels_enabled
        if extra_channels_enabled is not None:
            self.adapter.set_extra_channels_enabled(extra_channels_enabled)
        try:
            self.adapter.patch_model(self.yolo.model)
            return self.yolo.val(**kwargs)
        finally:
            if extra_channels_enabled is not None:
                self.adapter.set_extra_channels_enabled(old_enabled)
                self.adapter.patch_model(self.yolo.model)

    def predict(self, *args, extra_channels_enabled: Optional[bool] = None, **kwargs):
        old_enabled = self.adapter.extra_channels_enabled
        if extra_channels_enabled is not None:
            self.adapter.set_extra_channels_enabled(extra_channels_enabled)
        try:
            self.adapter.patch_model(self.yolo.model)
            return self.yolo.predict(*args, **kwargs)
        finally:
            if extra_channels_enabled is not None:
                self.adapter.set_extra_channels_enabled(old_enabled)
                self.adapter.patch_model(self.yolo.model)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self.yolo, name)
