"""Ultralytics integration for backbone superpixel correspondence."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from ..strategies.backbone_correspondence import (
    BackboneCorrespondenceConfig,
    BackboneCorrespondencePatchResult,
    patch_backbone_correspondence,
)
from .run_config import write_superpixel_run_config
from .ultralytics_neck import (
    DetectionTrainerBase,
    _find_spit_token_providers,
    _log_info,
    _spit_should_be_frozen,
    _split_spit_optimizer_groups,
)

ENV_PREFIX = "YOLO_SUPERPIXELS_BACKBONE_"
_ACTIVE_BACKBONE_ADAPTER: "BackboneCorrespondenceYOLOAdapter | None" = None


def _env_key(name: str) -> str:
    return f"{ENV_PREFIX}{name}"


def set_active_backbone_adapter(adapter: "BackboneCorrespondenceYOLOAdapter | None") -> None:
    global _ACTIVE_BACKBONE_ADAPTER
    _ACTIVE_BACKBONE_ADAPTER = adapter


def backbone_adapter_from_environment() -> "BackboneCorrespondenceYOLOAdapter":
    return BackboneCorrespondenceYOLOAdapter(
        config=BackboneCorrespondenceConfig.from_jsonable(json.loads(os.environ.get(_env_key("CONFIG"), "{}")))
    )


@dataclass
class BackboneCorrespondenceYOLOAdapter:
    """Patch an Ultralytics graph before its optimizer is constructed."""

    config: BackboneCorrespondenceConfig = field(default_factory=BackboneCorrespondenceConfig)

    def export_environment(self) -> None:
        os.environ[_env_key("CONFIG")] = json.dumps(self.config.to_jsonable())

    def prepare_training(self) -> None:
        set_active_backbone_adapter(self)
        self.export_environment()

    def patch_model(self, model) -> BackboneCorrespondencePatchResult:
        return patch_backbone_correspondence(model, self.config)

    def set_injection_enabled(self, enabled: bool) -> None:
        self.config.enabled = enabled


def _on_train_start(trainer: "BackboneCorrespondenceDetectionTrainer") -> None:
    trainer._apply_spit_training_policy(epoch_index=int(getattr(trainer, "start_epoch", 0)))
    trainer._log_spit_training_summary()


def _on_epoch_start(trainer: "BackboneCorrespondenceDetectionTrainer") -> None:
    trainer._apply_spit_training_policy(epoch_index=int(trainer.epoch))


class BackboneCorrespondenceDetectionTrainer(DetectionTrainerBase):
    """Detection trainer preserving Ultralytics optimizer and checkpoint lifecycle."""

    def __init__(self, *args, backbone_adapter: BackboneCorrespondenceYOLOAdapter | None = None, **kwargs) -> None:
        self.backbone_adapter = backbone_adapter or _ACTIVE_BACKBONE_ADAPTER or backbone_adapter_from_environment()
        self.spit_optimizer_group_indices: list[int] = []
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_start", _on_train_start)
        self.add_callback("on_train_epoch_start", _on_epoch_start)
        write_superpixel_run_config(
            self.save_dir,
            strategy="backbone_correspondence",
            config=self.backbone_adapter.config.to_jsonable(),
            args=self.args,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        self.backbone_patch = self.backbone_adapter.patch_model(model)
        return model

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        optimizer = super().build_optimizer(
            model=model,
            name=name,
            lr=lr,
            momentum=momentum,
            decay=decay,
            iterations=iterations,
        )
        config = self.backbone_adapter.config
        self.spit_optimizer_group_indices = _split_spit_optimizer_groups(
            optimizer,
            model,
            lr_scale=config.spit_lr_scale,
        )
        return optimizer

    def _apply_spit_training_policy(self, epoch_index: int) -> None:
        should_freeze = _spit_should_be_frozen(self.backbone_adapter.config, epoch_index)
        for provider in _find_spit_token_providers(self.model):
            provider.set_spit_frozen(should_freeze)

    def _log_spit_training_summary(self) -> None:
        config = self.backbone_adapter.config
        for index, provider in enumerate(_find_spit_token_providers(self.model)):
            trainable, total = provider.spit_parameter_counts()
            _log_info(
                f"Backbone SPiT provider {index}: placement={config.placement}, "
                f"mode={config.spit_train_mode}, frozen={provider.spit_frozen}, "
                f"trainable_params={trainable:,}/{total:,}, "
                f"loaded_keys={provider.loaded_spit_weight_keys}, "
                f"weights={provider.loaded_spit_weights or 'random'}"
            )


class SuperpixelBackboneYOLO:
    """Small YOLO wrapper for native-backbone and isolated-P3 correspondence."""

    def __init__(
        self,
        model: Any,
        adapter: BackboneCorrespondenceYOLOAdapter | None = None,
        config: BackboneCorrespondenceConfig | None = None,
        **config_kwargs,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("SuperpixelBackboneYOLO requires ultralytics.") from exc
        self.yolo = model if hasattr(model, "train") and hasattr(model, "model") else YOLO(model)
        self.adapter = adapter or BackboneCorrespondenceYOLOAdapter(
            config=config or BackboneCorrespondenceConfig(**config_kwargs)
        )

    def train(self, **kwargs):
        self.adapter.prepare_training()
        return self.yolo.train(trainer=BackboneCorrespondenceDetectionTrainer, **kwargs)

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
