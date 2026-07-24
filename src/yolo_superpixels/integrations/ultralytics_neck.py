"""Ultralytics integration for SPiT token neck injection."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from ..strategies.neck_injection import (
    NeckInjectionConfig,
    NeckInjectionPatchResult,
    patch_neck_injection,
)
from .run_config import write_superpixel_run_config

ENV_PREFIX = "YOLO_SUPERPIXELS_NECK_"
_ACTIVE_NECK_ADAPTER: "NeckInjectionYOLOAdapter | None" = None


def _env_key(name: str) -> str:
    return f"{ENV_PREFIX}{name}"


def set_active_neck_adapter(adapter: "NeckInjectionYOLOAdapter | None") -> None:
    """Set the in-process adapter used by ``NeckInjectionDetectionTrainer``."""

    global _ACTIVE_NECK_ADAPTER
    _ACTIVE_NECK_ADAPTER = adapter


def neck_adapter_from_environment() -> "NeckInjectionYOLOAdapter":
    values = json.loads(os.environ.get(_env_key("CONFIG"), "{}"))
    return NeckInjectionYOLOAdapter(config=NeckInjectionConfig.from_jsonable(values))


@dataclass
class NeckInjectionYOLOAdapter:
    """Patch YOLO models with SPiT token neck injection."""

    config: NeckInjectionConfig = field(default_factory=NeckInjectionConfig)

    def export_environment(self) -> None:
        os.environ[_env_key("CONFIG")] = json.dumps(self.config.to_jsonable())

    def prepare_training(self) -> None:
        set_active_neck_adapter(self)
        self.export_environment()

    def patch_model(self, model) -> NeckInjectionPatchResult:
        return patch_neck_injection(model, self.config)

    def set_injection_enabled(self, enabled: bool) -> None:
        self.config.enabled = enabled


def _load_detection_trainer_base():
    try:
        from ultralytics.models.yolo.detect import DetectionTrainer
    except ImportError as exc:
        raise ImportError(
            "NeckInjectionDetectionTrainer requires ultralytics. Install with "
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


class NeckInjectionDetectionTrainer(DetectionTrainerBase):
    """Trainer that patches the YOLO neck after pretrained weights are loaded."""

    def __init__(self, *args, neck_adapter: NeckInjectionYOLOAdapter | None = None, **kwargs) -> None:
        self.neck_adapter = neck_adapter or _ACTIVE_NECK_ADAPTER or neck_adapter_from_environment()
        super().__init__(*args, **kwargs)
        write_superpixel_run_config(
            self.save_dir,
            strategy="neck_injection",
            config=self.neck_adapter.config.to_jsonable(),
            args=self.args,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        self.neck_patch = self.neck_adapter.patch_model(model)
        return model


class SuperpixelNeckYOLO:
    """Small wrapper around ``ultralytics.YOLO`` with neck injection enabled."""

    def __init__(
        self,
        model: Any,
        adapter: NeckInjectionYOLOAdapter | None = None,
        config: NeckInjectionConfig | None = None,
        **config_kwargs,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "SuperpixelNeckYOLO requires ultralytics. Install with "
                "`pip install yolo-superpixels[ultralytics]` or `pip install ultralytics`."
            ) from exc

        self.yolo = model if hasattr(model, "train") and hasattr(model, "model") else YOLO(model)
        if adapter is not None:
            self.adapter = adapter
        else:
            self.adapter = NeckInjectionYOLOAdapter(config=config or NeckInjectionConfig(**config_kwargs))

    def train(self, **kwargs):
        self.adapter.prepare_training()
        return self.yolo.train(trainer=NeckInjectionDetectionTrainer, **kwargs)

    def set_injection_enabled(self, enabled: bool) -> None:
        self.adapter.set_injection_enabled(enabled)
        if getattr(self.yolo, "model", None) is not None:
            self.adapter.patch_model(self.yolo.model)

    def val(self, injection_enabled: Optional[bool] = None, **kwargs):
        old_enabled = self.adapter.config.enabled
        if injection_enabled is not None:
            self.adapter.set_injection_enabled(injection_enabled)
        try:
            self.adapter.patch_model(self.yolo.model)
            return self.yolo.val(**kwargs)
        finally:
            if injection_enabled is not None:
                self.adapter.set_injection_enabled(old_enabled)
                self.adapter.patch_model(self.yolo.model)

    def predict(self, *args, injection_enabled: Optional[bool] = None, **kwargs):
        old_enabled = self.adapter.config.enabled
        if injection_enabled is not None:
            self.adapter.set_injection_enabled(injection_enabled)
        try:
            self.adapter.patch_model(self.yolo.model)
            return self.yolo.predict(*args, **kwargs)
        finally:
            if injection_enabled is not None:
                self.adapter.set_injection_enabled(old_enabled)
                self.adapter.patch_model(self.yolo.model)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self.yolo, name)
