"""Ultralytics integration for SPiT token neck injection."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from ..strategies.neck_injection import (
    NeckInjectionConfig,
    NeckInjectionPatchResult,
    SPiTTokenProvider,
    patch_neck_injection,
)
from .run_config import write_superpixel_run_config

ENV_PREFIX = "YOLO_SUPERPIXELS_NECK_"
_ACTIVE_NECK_ADAPTER: "NeckInjectionYOLOAdapter | None" = None


def _env_key(name: str) -> str:
    return f"{ENV_PREFIX}{name}"


def _log_info(message: str) -> None:
    """Log through Ultralytics when available without making it an import-time dependency."""

    try:
        from ultralytics.utils import LOGGER
    except ImportError:
        return
    LOGGER.info(message)


def _find_spit_token_providers(model) -> list[SPiTTokenProvider]:
    """Return full learned SPiT providers nested anywhere in a YOLO model."""

    if model is None or not hasattr(model, "modules"):
        return []
    return [module for module in model.modules() if isinstance(module, SPiTTokenProvider)]


def _split_spit_optimizer_groups(optimizer, model, lr_scale: float) -> list[int]:
    """Split SPiT parameters from existing optimizer groups and scale only their LR.

    The base Ultralytics optimizer is built first, so optimizer selection, weight decay,
    momentum, MuSGD-specific options, and future Ultralytics group metadata are preserved.
    This function only separates SPiT tensors from each matching group and scales its LR.
    """

    lr_scale = float(lr_scale)
    if lr_scale <= 0:
        raise ValueError("spit_lr_scale must be positive.")

    providers = _find_spit_token_providers(model)
    spit_parameters = [parameter for provider in providers for parameter in provider.spit.parameters()]
    if not spit_parameters:
        return []

    spit_ids = {id(parameter) for parameter in spit_parameters}
    tracked_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group.get("params", [])
        if id(parameter) in spit_ids
    }
    missing = spit_ids - tracked_ids
    if missing:
        raise RuntimeError(
            f"Optimizer is missing {len(missing)} SPiT parameter tensors. "
            "SPiT must be trainable while the optimizer is initially constructed."
        )

    if lr_scale == 1.0:
        return []

    spit_group_indices: list[int] = []
    original_groups = list(optimizer.param_groups)
    for group in original_groups:
        parameters = list(group.get("params", []))
        regular = [parameter for parameter in parameters if id(parameter) not in spit_ids]
        spit = [parameter for parameter in parameters if id(parameter) in spit_ids]
        if not spit:
            continue

        if regular:
            group["params"] = regular
            spit_group = {key: value for key, value in group.items() if key != "params"}
            spit_group["params"] = spit
            spit_group["spit_group"] = True
            spit_group["spit_lr_scale"] = lr_scale
            if "lr" in spit_group:
                spit_group["lr"] = float(spit_group["lr"]) * lr_scale
            if "initial_lr" in spit_group:
                spit_group["initial_lr"] = float(spit_group["initial_lr"]) * lr_scale
            optimizer.add_param_group(spit_group)
            spit_group_indices.append(len(optimizer.param_groups) - 1)
        else:
            group["spit_group"] = True
            group["spit_lr_scale"] = lr_scale
            if "lr" in group:
                group["lr"] = float(group["lr"]) * lr_scale
            if "initial_lr" in group:
                group["initial_lr"] = float(group["initial_lr"]) * lr_scale
            spit_group_indices.append(optimizer.param_groups.index(group))

    return spit_group_indices


def _spit_should_be_frozen(config: NeckInjectionConfig, epoch_index: int) -> bool:
    """Resolve the requested SPiT freeze state for a zero-based epoch index."""

    if config.freeze_spit:
        return True
    if config.spit_train_mode == "frozen":
        return True
    if config.spit_train_mode == "delayed":
        return int(epoch_index) < int(config.spit_freeze_epochs)
    return False


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

    def set_token_ablation(self, mode: str, seed: int = 0) -> None:
        self.config.token_ablation = mode
        self.config.token_ablation_seed = int(seed)

    def set_feature_diagnostics(
        self,
        enabled: bool,
        max_queries: int = 256,
        max_forwards: int = 0,
    ) -> None:
        self.config.feature_diagnostics = bool(enabled)
        self.config.feature_diagnostics_max_queries = int(max_queries)
        self.config.feature_diagnostics_max_forwards = int(max_forwards)


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


def _on_spit_train_start(trainer: "NeckInjectionDetectionTrainer") -> None:
    trainer._apply_spit_training_policy(epoch_index=int(getattr(trainer, "start_epoch", 0)))
    trainer._log_spit_training_summary()


def _on_spit_train_epoch_start(trainer: "NeckInjectionDetectionTrainer") -> None:
    trainer._apply_spit_training_policy(epoch_index=int(trainer.epoch))


class NeckInjectionDetectionTrainer(DetectionTrainerBase):
    """Trainer that patches the YOLO neck and controls learned SPiT fine-tuning."""

    def __init__(self, *args, neck_adapter: NeckInjectionYOLOAdapter | None = None, **kwargs) -> None:
        self.neck_adapter = neck_adapter or _ACTIVE_NECK_ADAPTER or neck_adapter_from_environment()
        self.spit_optimizer_group_indices: list[int] = []
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_start", _on_spit_train_start)
        self.add_callback("on_train_epoch_start", _on_spit_train_epoch_start)
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

    def build_optimizer(
        self,
        model,
        name="auto",
        lr=0.001,
        momentum=0.9,
        decay=1e-5,
        iterations=1e5,
    ):
        """Build the normal Ultralytics optimizer, then apply a lower LR to SPiT groups."""

        optimizer = super().build_optimizer(
            model=model,
            name=name,
            lr=lr,
            momentum=momentum,
            decay=decay,
            iterations=iterations,
        )
        config = self.neck_adapter.config
        self.spit_optimizer_group_indices = _split_spit_optimizer_groups(
            optimizer,
            model,
            lr_scale=config.spit_lr_scale,
        )
        if self.spit_optimizer_group_indices:
            lrs = [optimizer.param_groups[index].get("lr") for index in self.spit_optimizer_group_indices]
            _log_info(
                "SPiT optimizer groups: "
                f"indices={self.spit_optimizer_group_indices}, "
                f"lr_scale={config.spit_lr_scale:g}, lrs={lrs}"
            )
        elif _find_spit_token_providers(model):
            _log_info("SPiT optimizer LR scale=1; preserving native Ultralytics parameter groups.")
        return optimizer

    def _apply_spit_training_policy(self, epoch_index: int) -> None:
        """Apply frozen/trainable state for the requested zero-based training epoch."""

        providers = _find_spit_token_providers(self.model)
        if not providers:
            return

        config = self.neck_adapter.config
        should_freeze = _spit_should_be_frozen(config, epoch_index)
        for provider in providers:
            old_state = provider.spit_frozen
            provider.set_spit_frozen(should_freeze)
            if old_state != should_freeze:
                state = "FROZEN" if should_freeze else "UNFROZEN"
                _log_info(f"Epoch {epoch_index + 1}: SPiT {state}.")

    def _log_spit_training_summary(self) -> None:
        """Record checkpoint loading, freeze policy, parameter counts, and LR policy."""

        config = self.neck_adapter.config
        providers = _find_spit_token_providers(self.model)
        for index, provider in enumerate(providers):
            trainable, total = provider.spit_parameter_counts()
            _log_info(
                f"SPiT provider {index}: mode={config.spit_train_mode}, "
                f"frozen={provider.spit_frozen}, freeze_epochs={config.spit_freeze_epochs}, "
                f"lr_scale={config.spit_lr_scale:g}, trainable_params={trainable:,}/{total:,}, "
                f"loaded_keys={provider.loaded_spit_weight_keys}, "
                f"skipped_checkpoint_keys={len(provider.skipped_spit_weight_keys)}, "
                f"missing_model_keys={len(provider.missing_spit_weight_keys)}, "
                f"weights={provider.loaded_spit_weights or 'random'}"
            )
            if provider.skipped_spit_weight_keys:
                skipped_preview = list(provider.skipped_spit_weight_keys.items())[:12]
                _log_info(f"SPiT skipped checkpoint keys (first 12): {skipped_preview}")
            if provider.missing_spit_weight_keys:
                missing_preview = provider.missing_spit_weight_keys[:12]
                _log_info(f"SPiT missing model keys after checkpoint load (first 12): {missing_preview}")


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

    def set_token_ablation(self, mode: str, seed: int = 0) -> None:
        self.adapter.set_token_ablation(mode, seed)
        if getattr(self.yolo, "model", None) is not None:
            self.adapter.patch_model(self.yolo.model)

    def set_feature_diagnostics(
        self,
        enabled: bool,
        max_queries: int = 256,
        max_forwards: int = 0,
    ) -> None:
        self.adapter.set_feature_diagnostics(enabled, max_queries, max_forwards)
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
