"""Ultralytics integration for SP-11 structural supervision."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from ..strategies.structural_supervision import (
    StructuralSupervisionConfig,
    StructuralSupervisionPatchResult,
    StructuralSupervisionWrapper,
    install_structural_criterion,
    patch_structural_supervision,
    strip_structural_supervision,
)
from .run_config import write_superpixel_run_config

ENV_PREFIX = "YOLO_SUPERPIXELS_STRUCTURAL_"
_ACTIVE_STRUCTURAL_ADAPTER: "StructuralSupervisionYOLOAdapter | None" = None
_LOSS_NAMES = (
    "box_loss",
    "cls_loss",
    "dfl_loss",
    "fg_aux_loss_raw",
    "fg_aux_loss_weighted",
    "sp_consistency_loss_raw",
    "sp_consistency_loss_weighted",
    "aux_total_weighted",
    "sp_requested_regions",
    "sp_actual_regions",
    "sp_active_foreground_regions",
    "sp_active_background_regions",
    "sp_ignored_ambiguous_regions",
    "sp_ignored_small_regions",
    "sp_mean_region_area",
    "sp_median_region_area",
    "sp_foreground_region_variance_raw",
    "sp_background_region_variance_raw",
)


def _env_key(name: str) -> str:
    return f"{ENV_PREFIX}{name}"


def set_active_structural_adapter(adapter: "StructuralSupervisionYOLOAdapter | None") -> None:
    global _ACTIVE_STRUCTURAL_ADAPTER
    _ACTIVE_STRUCTURAL_ADAPTER = adapter


def structural_adapter_from_environment() -> "StructuralSupervisionYOLOAdapter":
    values = json.loads(os.environ.get(_env_key("CONFIG"), "{}"))
    return StructuralSupervisionYOLOAdapter(config=StructuralSupervisionConfig.from_jsonable(values))


def _weights_have_structural_supervision(weights: Any) -> bool:
    modules = getattr(weights, "modules", None)
    return bool(callable(modules) and any(isinstance(module, StructuralSupervisionWrapper) for module in modules()))


@dataclass
class StructuralSupervisionYOLOAdapter:
    """Install and manage the isolated SP-11 training-only strategy."""

    config: StructuralSupervisionConfig = field(default_factory=StructuralSupervisionConfig)

    def export_environment(self) -> None:
        os.environ[_env_key("CONFIG")] = json.dumps(self.config.to_jsonable())

    def prepare_training(self) -> None:
        set_active_structural_adapter(self)
        self.export_environment()

    def patch_model(self, model, install_criterion: bool = False) -> StructuralSupervisionPatchResult:
        return patch_structural_supervision(model, self.config, install_criterion=install_criterion)

    def install_criterion(self, model) -> None:
        install_structural_criterion(model, self.config)

    def strip(self, model, remove_criterion_wrapper: bool = True) -> int:
        return strip_structural_supervision(model, remove_criterion_wrapper=remove_criterion_wrapper)


def _load_detection_trainer_base():
    try:
        from ultralytics.models.yolo.detect import DetectionTrainer
    except ImportError as exc:
        raise ImportError("StructuralSupervisionDetectionTrainer requires ultralytics.") from exc
    return DetectionTrainer


try:
    DetectionTrainerBase = _load_detection_trainer_base()
    _ULTRALYTICS_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    _ULTRALYTICS_IMPORT_ERROR = exc

    class DetectionTrainerBase:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise _ULTRALYTICS_IMPORT_ERROR


class StructuralSupervisionDetectionTrainer(DetectionTrainerBase):
    """Trainer that keeps SP-11 weights in checkpoints but runs it only in train mode."""

    def __init__(self, *args, structural_adapter: StructuralSupervisionYOLOAdapter | None = None, **kwargs) -> None:
        self.structural_adapter = (
            structural_adapter or _ACTIVE_STRUCTURAL_ADAPTER or structural_adapter_from_environment()
        )
        super().__init__(*args, **kwargs)
        config = self.structural_adapter.config.to_jsonable()
        if not config.get("base_model_path") and getattr(self.args, "model", None):
            config["base_model_path"] = str(self.args.model)
        write_superpixel_run_config(
            self.save_dir,
            strategy="structural_supervision_sp11",
            config=config,
            args=self.args,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        if _weights_have_structural_supervision(weights):
            model = super().get_model(cfg=cfg, weights=None, verbose=verbose)
            self.structural_patch = self.structural_adapter.patch_model(model, install_criterion=False)
            model.load(weights)
            return model
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        self.structural_patch = self.structural_adapter.patch_model(model, install_criterion=False)
        return model

    def set_class_weights(self):
        super().set_class_weights()
        try:
            from ultralytics.utils.torch_utils import unwrap_model
        except ImportError:
            unwrap_model = lambda model: model  # noqa: E731
        self.structural_adapter.install_criterion(unwrap_model(self.model))
        if self.ema is not None:
            self.structural_adapter.install_criterion(unwrap_model(self.ema.ema))

    def get_validator(self):
        validator = super().get_validator()
        self.loss_names = _LOSS_NAMES
        return validator


class SuperpixelStructuralYOLO:
    """Ultralytics YOLO wrapper for SP-11 train-only structural supervision."""

    def __init__(
        self,
        model: Any,
        adapter: StructuralSupervisionYOLOAdapter | None = None,
        config: StructuralSupervisionConfig | None = None,
        **config_kwargs,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("SuperpixelStructuralYOLO requires ultralytics.") from exc
        self.yolo = model if hasattr(model, "train") and hasattr(model, "model") else YOLO(model)
        self.adapter = adapter or StructuralSupervisionYOLOAdapter(
            config=config or StructuralSupervisionConfig(**config_kwargs)
        )

    def train(self, **kwargs):
        self.adapter.prepare_training()
        return self.yolo.train(trainer=StructuralSupervisionDetectionTrainer, **kwargs)

    def strip_structural_supervision(self, remove_criterion_wrapper: bool = True) -> int:
        model = getattr(self.yolo, "model", None)
        return self.adapter.strip(model, remove_criterion_wrapper) if model is not None else 0

    def val(self, **kwargs):
        return self.yolo.val(**kwargs)

    def predict(self, *args, **kwargs):
        return self.yolo.predict(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self.yolo, name)
