"""Versioned, fail-closed protocol resolution for the SP-11 A/B/C/D pilot."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ARM_FILENAMES = {
    "A": "A_rgb_baseline.yaml",
    "B": "B_foreground_aux.yaml",
    "C": "C_foreground_aux_slic.yaml",
    "D": "D_foreground_aux_grid.yaml",
}
_ARM_TOP_LEVEL_KEYS = {"arm", "description", "experiment", "strategy", "structural_supervision"}
_STRUCTURAL_OVERRIDE_KEYS = {"lambda_sp", "region_algorithm", "region_kwargs"}


@dataclass(frozen=True)
class ResolvedSP11Arm:
    arm: str
    arm_path: Path
    arm_sha256: str
    protocol_path: Path
    protocol_sha256: str
    shared_protocol_sha256: str
    resolved_sha256: str
    resolved: dict[str, Any]
    arm_override: dict[str, Any]
    base_checkpoint: dict[str, Any]
    data_yaml: dict[str, Any]

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "strategy": self.resolved["strategy"],
            "arm_config_path": str(self.arm_path),
            "arm_config_sha256": self.arm_sha256,
            "resolved_config_sha256": self.resolved_sha256,
            "resolved_config": self.resolved,
            "base_checkpoint": self.base_checkpoint,
            "data_yaml": self.data_yaml,
            "intervention": self.arm_override.get("structural_supervision", {}),
        }


def load_sp11_pilot(protocol_path: str | Path, arms: list[str] | tuple[str, ...]) -> list[ResolvedSP11Arm]:
    """Load a shared protocol and arm-only overrides, resolving file identities."""

    protocol_file = Path(protocol_path).resolve()
    protocol = _load_yaml_mapping(protocol_file)
    _validate_protocol(protocol, protocol_file)
    requested = [str(arm).upper() for arm in arms]
    if not requested:
        raise ValueError("At least one SP-11 arm must be selected.")
    if len(set(requested)) != len(requested):
        raise ValueError("Each SP-11 arm may be selected only once.")
    unknown = set(requested).difference(ARM_FILENAMES)
    if unknown:
        raise ValueError(f"Unknown SP-11 arm(s): {sorted(unknown)}.")

    protocol_sha = _sha256_file(protocol_file)
    shared = _copy_mapping(protocol["shared"])
    shared_sha = _sha256_json(shared)
    resolved_arms = []
    for arm in requested:
        arm_path = protocol_file.parent / ARM_FILENAMES[arm]
        arm_payload = _load_yaml_mapping(arm_path)
        _validate_arm_override(arm, arm_payload, arm_path)
        resolved = _resolve_arm(shared, arm_payload)
        base_checkpoint = _resolve_file_identity(
            resolved["initialization"]["base_model_path"], protocol_file.parent, "base checkpoint"
        )
        expected_sha = resolved["initialization"].get("base_model_sha256")
        if expected_sha in {None, "", "record_at_launch"}:
            expected_sha = None
        if expected_sha is not None and expected_sha != base_checkpoint["sha256"]:
            raise ValueError(
                f"{arm}: base checkpoint SHA-256 mismatch; expected {expected_sha}, "
                f"got {base_checkpoint['sha256']}."
            )
        resolved["initialization"]["base_model_path"] = base_checkpoint["path"]
        resolved["initialization"]["base_model_sha256"] = base_checkpoint["sha256"]
        data_yaml = _resolve_file_identity(resolved["training"]["data"], protocol_file.parent, "data YAML")
        resolved["training"]["data"] = data_yaml["path"]
        resolved_arms.append(
            ResolvedSP11Arm(
                arm=arm,
                arm_path=arm_path.resolve(),
                arm_sha256=_sha256_file(arm_path),
                protocol_path=protocol_file,
                protocol_sha256=protocol_sha,
                shared_protocol_sha256=shared_sha,
                resolved_sha256=_sha256_json(resolved),
                resolved=resolved,
                arm_override=_copy_mapping(arm_payload),
                base_checkpoint=base_checkpoint,
                data_yaml=data_yaml,
            )
        )
    assert_matched_sp11_protocol(resolved_arms)
    return resolved_arms


def assert_matched_sp11_protocol(arms: list[ResolvedSP11Arm]) -> None:
    """Fail closed if supposedly paired arms differ outside their intervention."""

    if not arms:
        return
    reference = arms[0]
    common = (
        "shared_protocol_sha256",
        "protocol_sha256",
    )
    for arm in arms[1:]:
        for field in common:
            if getattr(arm, field) != getattr(reference, field):
                raise ValueError(f"SP-11 protocol drift: {arm.arm} differs in {field}.")
        if arm.base_checkpoint["sha256"] != reference.base_checkpoint["sha256"]:
            raise ValueError(f"SP-11 protocol drift: {arm.arm} has another base checkpoint.")
        if arm.data_yaml["sha256"] != reference.data_yaml["sha256"]:
            raise ValueError(f"SP-11 protocol drift: {arm.arm} has another data YAML.")


def pilot_manifest(resolved_arms: list[ResolvedSP11Arm]) -> dict[str, Any]:
    """Return machine-readable provenance before any training starts."""

    assert_matched_sp11_protocol(resolved_arms)
    if not resolved_arms:
        raise ValueError("Cannot create an SP-11 manifest without arms.")
    first = resolved_arms[0]
    return {
        "kind": "sp11_pilot_manifest",
        "protocol_path": str(first.protocol_path),
        "protocol_sha256": first.protocol_sha256,
        "shared_protocol_sha256": first.shared_protocol_sha256,
        "base_checkpoint": first.base_checkpoint,
        "data_yaml": first.data_yaml,
        "arms": [arm.manifest_entry() for arm in resolved_arms],
    }


def _resolve_arm(shared: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    resolved = _copy_mapping(shared)
    resolved["arm"] = override["arm"]
    resolved["strategy"] = override["strategy"]
    if override["strategy"] == "structural_supervision":
        structural = _copy_mapping(shared["structural_supervision"])
        structural.update(override.get("structural_supervision", {}))
        resolved["structural_supervision"] = structural
    else:
        resolved.pop("structural_supervision", None)
    return resolved


def _validate_protocol(protocol: dict[str, Any], path: Path) -> None:
    if protocol.get("protocol") != "sp11_pilot_v1":
        raise ValueError(f"{path}: expected protocol: sp11_pilot_v1.")
    shared = protocol.get("shared")
    if not isinstance(shared, dict):
        raise ValueError(f"{path}: shared mapping is required.")
    required = {"initialization", "training", "structural_supervision"}
    missing = required.difference(shared)
    if missing:
        raise ValueError(f"{path}: shared protocol misses {sorted(missing)}.")
    for key in required:
        if not isinstance(shared[key], dict):
            raise ValueError(f"{path}: shared.{key} must be a mapping.")
    if "base_model_path" not in shared["initialization"]:
        raise ValueError(f"{path}: shared.initialization.base_model_path is required.")
    if "data" not in shared["training"]:
        raise ValueError(f"{path}: shared.training.data is required.")


def _validate_arm_override(arm: str, override: dict[str, Any], path: Path) -> None:
    unexpected = set(override).difference(_ARM_TOP_LEVEL_KEYS)
    if unexpected:
        raise ValueError(f"{path}: shared fields are forbidden in arm overrides: {sorted(unexpected)}.")
    if override.get("arm") != arm:
        raise ValueError(f"{path}: arm must be {arm!r}.")
    strategy = override.get("strategy")
    if arm == "A":
        if strategy != "rgb_baseline" or "structural_supervision" in override:
            raise ValueError(f"{path}: A must be RGB-only and cannot define structural supervision.")
        return
    if strategy != "structural_supervision":
        raise ValueError(f"{path}: {arm} must use structural_supervision.")
    structural = override.get("structural_supervision")
    if not isinstance(structural, dict):
        raise ValueError(f"{path}: structural_supervision mapping is required.")
    unexpected_structural = set(structural).difference(_STRUCTURAL_OVERRIDE_KEYS)
    if unexpected_structural:
        raise ValueError(
            f"{path}: only intervention keys {_STRUCTURAL_OVERRIDE_KEYS} may override shared structural settings; "
            f"got {sorted(unexpected_structural)}."
        )


def _resolve_file_identity(value: Any, config_dir: Path, label: str) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"SP-11 {label} path must be a non-empty string.")
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise ValueError(f"SP-11 {label} path contains an unresolved environment variable: {value!r}.")
    path = Path(expanded).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, config_dir / path]
    existing = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if existing is None:
        raise FileNotFoundError(f"SP-11 {label} was not found. Tried: {', '.join(str(item) for item in candidates)}")
    return {"path": str(existing), "sha256": _sha256_file(existing), "bytes": existing.stat().st_size}


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping.")
    return value


def _copy_mapping(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
