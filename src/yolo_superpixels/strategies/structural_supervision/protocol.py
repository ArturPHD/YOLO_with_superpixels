"""Versioned, fail-closed protocol resolution for the SP-11 A/B/C/D pilot."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
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
_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_INIT_MODES = {"technical_pretrained_smoke", "scientific_paired_scratch"}


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
    dataset_identity: dict[str, Any]

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
            "dataset_identity_sha256": self.dataset_identity["sha256"],
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
    base_checkpoint = _resolve_initialization(shared["initialization"], protocol_file.parent)
    data_yaml = _resolve_file_identity(shared["training"]["data"], protocol_file.parent, "data YAML")
    dataset_identity = build_dataset_identity(data_yaml["path"])
    _validate_scientific_subset(shared, dataset_identity)
    resolved_arms = []
    for arm in requested:
        arm_path = protocol_file.parent / ARM_FILENAMES[arm]
        arm_payload = _load_yaml_mapping(arm_path)
        _validate_arm_override(arm, arm_payload, arm_path)
        resolved = _resolve_arm(shared, arm_payload)
        resolved["initialization"]["base_model_path"] = base_checkpoint["path"]
        resolved["initialization"]["base_model_sha256"] = base_checkpoint["sha256"]
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
                dataset_identity=dataset_identity,
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
        if arm.dataset_identity["sha256"] != reference.dataset_identity["sha256"]:
            raise ValueError(f"SP-11 protocol drift: {arm.arm} has another dataset identity.")


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
        "dataset_identity": _public_dataset_identity(first.dataset_identity),
        "software_source_identity": _software_source_identity(first.protocol_path.parents[2]),
        "shared_training_protocol": first.resolved["training"],
        "shared_structural_protocol": first.resolved.get("structural_supervision"),
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
    initialization = shared["initialization"]
    mode = initialization.get("mode", "technical_pretrained_smoke")
    if mode not in _INIT_MODES:
        raise ValueError(f"{path}: unsupported initialization mode {mode!r}.")
    if "base_model_path" not in initialization:
        raise ValueError(f"{path}: shared.initialization.base_model_path is required.")
    if mode == "scientific_paired_scratch":
        for key in ("source_model_yaml", "base_initialization_seed", "auxiliary_init_seed"):
            if key not in initialization:
                raise ValueError(f"{path}: scientific_paired_scratch requires initialization.{key}.")
    if "data" not in shared["training"]:
        raise ValueError(f"{path}: shared.training.data is required.")
    training = shared["training"]
    if training.get("patience") != 0:
        raise ValueError(f"{path}: fixed-horizon SP-11 protocol requires training.patience: 0.")
    if not isinstance(training.get("epochs"), int) or training["epochs"] < 1:
        raise ValueError(f"{path}: fixed positive training.epochs is required.")


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


def _resolve_initialization(initialization: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    """Resolve one physical base checkpoint shared by all selected arms."""

    mode = initialization.get("mode", "technical_pretrained_smoke")
    if mode == "technical_pretrained_smoke":
        identity = _resolve_file_identity(initialization["base_model_path"], config_dir, "base checkpoint")
    else:
        identity = _create_or_load_paired_scratch_checkpoint(initialization, config_dir)
    expected = initialization.get("base_model_sha256")
    if expected not in {None, "", "record_at_launch"} and expected != identity["sha256"]:
        raise ValueError(f"base checkpoint SHA-256 mismatch; expected {expected}, got {identity['sha256']}.")
    identity.update(
        {
            "initialization_mode": mode,
            "base_initialization_seed": initialization.get("base_initialization_seed"),
            "auxiliary_initialization_seed": initialization.get("auxiliary_init_seed"),
        }
    )
    return identity


def _create_or_load_paired_scratch_checkpoint(initialization: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    """Create the scratch state once, under isolated RNG, then only reuse it."""

    path = _resolve_path(initialization["base_model_path"], config_dir)
    source = _resolve_file_identity(initialization["source_model_yaml"], config_dir, "source model YAML")
    seed = initialization["base_initialization_seed"]
    if not isinstance(seed, int):
        raise ValueError("base_initialization_seed must be an integer.")
    if not path.exists():
        if not initialization.get("create_if_missing", False):
            raise FileNotFoundError(f"Paired scratch checkpoint is missing: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        import torch
        from ultralytics import YOLO

        # The isolated generator avoids shifting dataloader/augmentation RNG state.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            model = YOLO(source["path"])
            model.save(str(path))
        del model
    identity = _resolve_file_identity(str(path), config_dir, "paired scratch checkpoint")
    identity.update(
        {
            "source_model_yaml": source,
            "parameter_count": _checkpoint_parameter_count(path),
            "initialization_metadata_path": str(path.with_suffix(".json")),
        }
    )
    metadata_path = path.with_suffix(".json")
    payload = {
        "kind": "sp11_paired_scratch_initialization",
        "checkpoint": identity,
        "base_initialization_seed": seed,
        "software_source_identity": _software_source_identity(config_dir.parents[1]),
    }
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("checkpoint", {}).get("sha256") != identity["sha256"]:
            raise ValueError(f"Scratch checkpoint metadata does not match {path}.")
    else:
        metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return identity


def _checkpoint_parameter_count(path: Path) -> int:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    return int(sum(parameter.numel() for parameter in model.parameters())) if model is not None else 0


def build_dataset_identity(data_yaml: str | Path) -> dict[str, Any]:
    """Fingerprint sorted actual image paths and label file contents for both splits."""

    yaml_path = Path(data_yaml).resolve()
    data = _load_yaml_mapping(yaml_path)
    root = _resolve_dataset_source(data.get("path", str(yaml_path.parent)), yaml_path.parent, yaml_path.parent)
    records = {split: _split_identity(split, data.get(split), root, yaml_path.parent) for split in ("train", "val")}
    names = data.get("names", {})
    class_names = list(names.values()) if isinstance(names, dict) else list(names) if isinstance(names, list) else names
    identity = {
        "data_yaml_path": str(yaml_path), "data_yaml_sha256": _sha256_file(yaml_path), "dataset_root": str(root),
        "class_count": int(data.get("nc", len(class_names) if class_names else 0)), "class_names": class_names,
        "class_names_sha256": _sha256_json({"class_names": class_names}),
        "train": _public_split_identity(records["train"]), "val": _public_split_identity(records["val"]),
        "_records": records,
    }
    identity["sha256"] = _sha256_json(_public_dataset_identity(identity))
    return identity


def write_dataset_identity_records(identity: dict[str, Any], directory: str | Path) -> dict[str, str]:
    """Persist actual sorted manifests, so a future run cannot silently regenerate a subset."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split in ("train", "val"):
        record = identity["_records"][split]
        for kind, lines in (("images", record["image_paths"]), ("labels", record["label_lines"])):
            path = directory / f"{split}_{kind}_manifest.txt"
            path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            outputs[f"{split}_{kind}"] = str(path)
    return outputs


def _split_identity(split: str, value: Any, root: Path, yaml_parent: Path) -> dict[str, Any]:
    sources = value if isinstance(value, list) else [value]
    images: list[Path] = []
    source_paths = []
    for source in sources:
        if not isinstance(source, str):
            raise ValueError(f"data.yaml {split} must be a path or list of paths.")
        path = _resolve_dataset_source(source, root, yaml_parent)
        source_paths.append(str(path))
        if path.is_dir():
            images.extend(path for path in path.rglob("*") if path.suffix.lower() in _IMAGE_SUFFIXES)
        elif path.suffix.lower() == ".txt":
            images.extend(_resolve_dataset_source(line.strip(), root, path.parent) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        elif path.suffix.lower() in _IMAGE_SUFFIXES:
            images.append(path)
        else:
            raise ValueError(f"Unsupported {split} source: {path}")
    image_paths = sorted({str(path.resolve()) for path in images})
    label_paths = [_label_path_for_image(Path(path)) for path in image_paths]
    label_lines = [f"{path}\t{_sha256_file(path) if path.is_file() else 'MISSING'}" for path in label_paths]
    labels = [path for path in label_paths if path.is_file()]
    object_count, class_coverage = _label_statistics(labels)
    positive = sum(1 for path in labels if path.read_text(encoding="utf-8").strip())
    return {
        "source_paths": source_paths, "source_sha256": _sha256_json({"sources": source_paths}), "image_paths": image_paths,
        "image_manifest_sha256": _sha256_json({"images": image_paths}), "label_lines": label_lines,
        "label_manifest_sha256": _sha256_json({"labels": label_lines}), "image_count": len(image_paths),
        "label_file_count": len(labels), "object_count": object_count, "positive_images": positive,
        "background_images": len(image_paths) - positive, "class_coverage": class_coverage,
    }


def _resolve_dataset_source(value: str, *bases: Path) -> Path:
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise ValueError(f"Unresolved dataset environment variable: {value!r}.")
    path = Path(expanded).expanduser()
    candidates = [path] if path.is_absolute() else [base / path for base in bases]
    existing = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
    if existing is None:
        raise FileNotFoundError(f"Dataset source not found: {value!r}")
    return existing


def _label_path_for_image(image: Path) -> Path:
    parts = list(image.parts)
    lower = [part.lower() for part in parts]
    if "images" in lower:
        parts[lower.index("images")] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def _label_statistics(paths: list[Path]) -> tuple[int, list[int]]:
    objects, coverage = 0, set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) >= 5:
                objects += 1
                try:
                    coverage.add(int(float(fields[0])))
                except ValueError:
                    pass
    return objects, sorted(coverage)


def _public_split_identity(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in {"image_paths", "label_lines"}}


def _public_dataset_identity(identity: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in identity.items() if key not in {"_records", "sha256"}}


def _validate_scientific_subset(shared: dict[str, Any], identity: dict[str, Any]) -> None:
    if shared["initialization"].get("mode", "technical_pretrained_smoke") != "scientific_paired_scratch":
        return
    subset = shared.get("scientific_subset")
    if not isinstance(subset, dict) or not subset.get("provenance"):
        raise ValueError("scientific_paired_scratch requires scientific_subset.provenance.")
    base_count = subset.get("base_train_image_count")
    actual_count = identity["train"]["image_count"]
    if not isinstance(base_count, int) or base_count < actual_count:
        raise ValueError("scientific_subset.base_train_image_count must be >= actual train image count.")
    fraction = actual_count / base_count
    expected, tolerance = subset.get("expected_fraction"), float(subset.get("fraction_tolerance", 0.0))
    if expected is not None and abs(fraction - float(expected)) > tolerance:
        raise ValueError(f"Scientific subset fraction {fraction:.6f} differs from expected {expected}.")
    identity["low_data_subset"] = {"provenance": str(subset["provenance"]), "base_train_image_count": base_count,
                                   "actual_fraction": fraction, "expected_fraction": expected, "fraction_tolerance": tolerance}
    identity["sha256"] = _sha256_json(_public_dataset_identity(identity))


def _software_source_identity(root: Path) -> dict[str, Any]:
    try:
        from yolo_superpixels.integrations.run_config import source_tree_metadata
        tree = source_tree_metadata(root)
    except Exception as error:
        tree = {"sha256": None, "error": str(error)}
    versions = {}
    for name in ("torch", "ultralytics", "numpy", "scikit-image"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    payload = {"python": sys.version, "platform": platform.platform(), "versions": versions, "source_tree": tree}
    try:
        payload["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        payload["git_commit"] = None
    return payload


def torch_slic_repeatability_metadata(device: str = "cuda") -> dict[str, Any]:
    """Small CUDA preflight record; strict equality is reported, never assumed."""

    import torch

    result: dict[str, Any] = {"device": device, "cuda_available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        return result
    from yolo_superpixels.superpixels.algorithms import TorchSLICSuperpixelAlgorithm

    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        image = torch.linspace(0, 1, 3 * 48 * 48, device=device, dtype=torch.float32).reshape(1, 3, 48, 48)
        algorithm = TorchSLICSuperpixelAlgorithm(n_segments=32, compactness=1.0, num_iters=2)
        labels = [algorithm.labels_for_batch(image) for _ in range(3)]
        hashes = [hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for value in labels]
        result.update({
            "deterministic_algorithms_enabled": True, "bit_identical": all(torch.equal(labels[0], value) for value in labels[1:]),
            "label_sha256": hashes, "gpu": torch.cuda.get_device_name(torch.device(device)), "torch": torch.__version__, "cuda": torch.version.cuda,
        })
    finally:
        torch.use_deterministic_algorithms(previous)
    return result


def _resolve_file_identity(value: Any, config_dir: Path, label: str) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"SP-11 {label} path must be a non-empty string.")
    existing = _resolve_path(value, config_dir)
    if not existing.is_file():
        raise FileNotFoundError(f"SP-11 {label} was not found: {existing}")
    return {"path": str(existing), "sha256": _sha256_file(existing), "bytes": existing.stat().st_size}


def _resolve_path(value: str, config_dir: Path) -> Path:
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise ValueError(f"SP-11 path contains an unresolved environment variable: {value!r}.")
    path = Path(expanded).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = [Path.cwd() / path, config_dir / path]
    return next((candidate.resolve() for candidate in candidates if candidate.exists()), candidates[0].resolve())


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
