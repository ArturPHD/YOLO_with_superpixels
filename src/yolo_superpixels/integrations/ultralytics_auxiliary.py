"""Ultralytics integration for training-only auxiliary superpixel losses."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from ..strategies.auxiliary_loss import (
    AuxiliaryLossConfig,
    AuxiliaryLossPatchResult,
    install_auxiliary_criterion,
    patch_auxiliary_loss,
    strip_auxiliary_loss,
)
from .run_config import write_superpixel_run_config

ENV_PREFIX = "YOLO_SUPERPIXELS_AUX_"
_ACTIVE_AUXILIARY_ADAPTER: "AuxiliaryLossYOLOAdapter | None" = None


def _env_key(name: str) -> str:
    return f"{ENV_PREFIX}{name}"


def set_active_auxiliary_adapter(adapter: "AuxiliaryLossYOLOAdapter | None") -> None:
    """Set the in-process adapter used by ``AuxiliaryLossDetectionTrainer``."""

    global _ACTIVE_AUXILIARY_ADAPTER
    _ACTIVE_AUXILIARY_ADAPTER = adapter


def auxiliary_adapter_from_environment() -> "AuxiliaryLossYOLOAdapter":
    values = json.loads(os.environ.get(_env_key("CONFIG"), "{}"))
    return AuxiliaryLossYOLOAdapter(config=AuxiliaryLossConfig.from_jsonable(values))


def _weights_have_auxiliary_loss(weights: Any) -> bool:
    if weights is None:
        return False
    modules = getattr(weights, "modules", None)
    if callable(modules):
        for module in weights.modules():
            if module.__class__.__name__ == "AuxiliarySuperpixelWrapper":
                return True
    state_dict = getattr(weights, "state_dict", None)
    if callable(state_dict):
        return any(".head.net." in key for key in state_dict().keys())
    return False


@dataclass
class AuxiliaryLossYOLOAdapter:
    """Patch YOLO models with a training-only superpixel auxiliary loss."""

    config: AuxiliaryLossConfig = field(default_factory=AuxiliaryLossConfig)

    def export_environment(self) -> None:
        os.environ[_env_key("CONFIG")] = json.dumps(self.config.to_jsonable())

    def prepare_training(self) -> None:
        set_active_auxiliary_adapter(self)
        self.export_environment()

    def patch_model(self, model, install_criterion: bool = False) -> AuxiliaryLossPatchResult:
        return patch_auxiliary_loss(model, self.config, install_criterion=install_criterion)

    def install_criterion(self, model) -> None:
        install_auxiliary_criterion(model, self.config)

    def set_auxiliary_enabled(self, enabled: bool) -> None:
        self.config.enabled = enabled

    def strip(self, model, remove_criterion_wrapper: bool = True) -> int:
        return strip_auxiliary_loss(model, remove_criterion_wrapper=remove_criterion_wrapper)


def _load_detection_trainer_base():
    try:
        from ultralytics.models.yolo.detect import DetectionTrainer
    except ImportError as exc:
        raise ImportError(
            "AuxiliaryLossDetectionTrainer requires ultralytics. Install with "
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


class AuxiliaryLossDetectionTrainer(DetectionTrainerBase):
    """Trainer that keeps auxiliary weights in checkpoints but uses them only for training."""

    def __init__(self, *args, auxiliary_adapter: AuxiliaryLossYOLOAdapter | None = None, **kwargs) -> None:
        self.auxiliary_adapter = auxiliary_adapter or _ACTIVE_AUXILIARY_ADAPTER or auxiliary_adapter_from_environment()
        super().__init__(*args, **kwargs)
        write_superpixel_run_config(
            self.save_dir,
            strategy="auxiliary_loss",
            config=self.auxiliary_adapter.config.to_jsonable(),
            args=self.args,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        if _weights_have_auxiliary_loss(weights):
            model = super().get_model(cfg=cfg, weights=None, verbose=verbose)
            self.auxiliary_patch = self.auxiliary_adapter.patch_model(model, install_criterion=False)
            model.load(weights)
            return model

        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        self.auxiliary_patch = self.auxiliary_adapter.patch_model(model, install_criterion=False)
        return model

    def set_class_weights(self):
        super().set_class_weights()
        try:
            from ultralytics.utils.torch_utils import unwrap_model
        except ImportError:
            unwrap_model = lambda model: model  # noqa: E731
        self.auxiliary_adapter.install_criterion(unwrap_model(self.model))
        if self.ema is not None:
            self.auxiliary_adapter.install_criterion(unwrap_model(self.ema.ema))

    def get_validator(self):
        validator = super().get_validator()
        if "sp_loss" not in self.loss_names:
            self.loss_names += ("sp_loss",)
        return validator


class SuperpixelAuxiliaryYOLO:
    """Small wrapper around ``ultralytics.YOLO`` with auxiliary superpixel loss enabled."""

    def __init__(
        self,
        model: Any,
        adapter: AuxiliaryLossYOLOAdapter | None = None,
        config: AuxiliaryLossConfig | None = None,
        **config_kwargs,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "SuperpixelAuxiliaryYOLO requires ultralytics. Install with "
                "`pip install yolo-superpixels[ultralytics]` or `pip install ultralytics`."
            ) from exc

        self.yolo = model if hasattr(model, "train") and hasattr(model, "model") else YOLO(model)
        if adapter is not None:
            self.adapter = adapter
        else:
            self.adapter = AuxiliaryLossYOLOAdapter(config=config or AuxiliaryLossConfig(**config_kwargs))

    def train(self, **kwargs):
        self.adapter.prepare_training()
        return self.yolo.train(trainer=AuxiliaryLossDetectionTrainer, **kwargs)

    def set_auxiliary_enabled(self, enabled: bool) -> None:
        self.adapter.set_auxiliary_enabled(enabled)
        if getattr(self.yolo, "model", None) is not None:
            self.adapter.patch_model(self.yolo.model, install_criterion=False)

    def strip_auxiliary_loss(self, remove_criterion_wrapper: bool = True) -> int:
        if getattr(self.yolo, "model", None) is None:
            return 0
        return self.adapter.strip(self.yolo.model, remove_criterion_wrapper=remove_criterion_wrapper)

    def val(self, **kwargs):
        return self.yolo.val(**kwargs)

    def predict(self, *args, **kwargs):
        return self.yolo.predict(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self.yolo, name)
