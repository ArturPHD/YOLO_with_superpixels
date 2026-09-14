"""Run metadata helpers for reproducible superpixel experiments."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterator
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

SOURCE_TREE_INCLUDE_DIRS = ("src", "examples", "scripts", "configs")
SOURCE_TREE_INCLUDE_FILES = (
    ".gitignore",
    "constraints-verified-windows-py310-cu126.txt",
    "pyproject.toml",
    "requirements.txt",
)
SOURCE_TREE_EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
SOURCE_TREE_EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
EXTERNAL_TREE_EXCLUDE_PARTS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
EXTERNAL_TREE_EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def write_superpixel_run_config(save_dir: str | Path, strategy: str, config: dict[str, Any], args: Any = None) -> None:
    """Write a sidecar config file next to Ultralytics ``args.yaml``."""

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(),
        "strategy": strategy,
        "config": config,
        "training_args": _selected_args(args),
        "versions": _package_versions(
            [
                "torch",
                "torchvision",
                "ultralytics",
                "numpy",
                "scikit-image",
                "opencv-contrib-python",
                "cupy-cuda12x",
            ]
        ),
        "git": _git_metadata(_repo_root()),
        "source_tree": source_tree_metadata(_repo_root()),
        "spit": _spit_metadata(_repo_root()),
        "external_files": _external_file_metadata(config, _repo_root()),
    }
    (save_path / "superpixel_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _selected_args(args: Any) -> dict[str, Any]:
    if args is None:
        return {}
    names = ["model", "data", "epochs", "imgsz", "batch", "workers", "device", "seed", "project", "name"]
    return {name: getattr(args, name) for name in names if hasattr(args, name)}


def _package_versions(names: list[str]) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def source_tree_metadata(root: str | Path | None = None) -> dict[str, Any]:
    """Return a small deterministic fingerprint of the experiment source tree."""

    root_path = Path(root).resolve() if root is not None else _repo_root()
    digest = hashlib.sha256()
    total_bytes = 0
    files = list(_iter_source_tree_files(root_path))
    errors: list[str] = []
    for path in files:
        rel = path.relative_to(root_path).as_posix()
        try:
            content = path.read_bytes()
        except OSError as exc:
            errors.append(f"{rel}: {exc}")
            continue
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).hexdigest().encode("ascii"))
        digest.update(b"\0")
        total_bytes += len(content)
    return {
        "sha256": digest.hexdigest(),
        "files": len(files),
        "bytes": total_bytes,
        "include_dirs": list(SOURCE_TREE_INCLUDE_DIRS),
        "include_files": list(SOURCE_TREE_INCLUDE_FILES),
        "errors": errors,
    }


def directory_tree_metadata(root: str | Path, exclude_parts: set[str] | None = None) -> dict[str, Any]:
    """Return a deterministic fingerprint for an external source directory."""

    root_path = Path(root)
    if not root_path.exists():
        return {"path": str(root_path), "sha256": None, "files": 0, "bytes": 0, "errors": ["missing"]}
    exclude_parts = exclude_parts or EXTERNAL_TREE_EXCLUDE_PARTS
    digest = hashlib.sha256()
    total_bytes = 0
    errors: list[str] = []
    files = [
        path
        for path in root_path.rglob("*")
        if path.is_file() and not _is_external_tree_excluded(path, exclude_parts=exclude_parts)
    ]
    files = sorted(files, key=lambda path: path.relative_to(root_path).as_posix())
    for path in files:
        rel = path.relative_to(root_path).as_posix()
        try:
            content = path.read_bytes()
        except OSError as exc:
            errors.append(f"{rel}: {exc}")
            continue
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).hexdigest().encode("ascii"))
        digest.update(b"\0")
        total_bytes += len(content)
    return {
        "path": str(root_path),
        "sha256": digest.hexdigest(),
        "files": len(files),
        "bytes": total_bytes,
        "errors": errors,
    }


def file_metadata(path: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    """Hash a single external file such as pretrained SPiT weights."""

    file_path = Path(path)
    if not file_path.is_absolute() and root is not None:
        file_path = Path(root) / file_path
    if not file_path.exists():
        return {"path": str(file_path), "sha256": None, "bytes": 0, "error": "missing"}
    try:
        content = file_path.read_bytes()
    except OSError as exc:
        return {"path": str(file_path), "sha256": None, "bytes": 0, "error": str(exc)}
    return {
        "path": str(file_path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


def _iter_source_tree_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for name in SOURCE_TREE_INCLUDE_FILES:
        path = root / name
        if path.is_file() and not _is_source_tree_excluded(path):
            files.append(path)
    for name in SOURCE_TREE_INCLUDE_DIRS:
        directory = root / name
        if not directory.is_dir():
            continue
        files.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and not _is_source_tree_excluded(path)
        )
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _is_source_tree_excluded(path: Path) -> bool:
    if any(part in SOURCE_TREE_EXCLUDE_PARTS for part in path.parts):
        return True
    return path.suffix in SOURCE_TREE_EXCLUDE_SUFFIXES


def _is_external_tree_excluded(path: Path, exclude_parts: set[str]) -> bool:
    if any(part in exclude_parts for part in path.parts):
        return True
    return path.suffix in EXTERNAL_TREE_EXCLUDE_SUFFIXES


def _git_metadata(root: Path) -> dict[str, Any]:
    return {
        "commit": _run_git(root, "rev-parse", "HEAD"),
        "branch": _run_git(root, "branch", "--show-current"),
        "status_short": _run_git(root, "status", "--short"),
    }


def _spit_metadata(root: Path) -> dict[str, Any]:
    spit_dir = root / "third_party" / "SPiT"
    head = spit_dir / ".git" / "HEAD"
    if not head.exists():
        return {"path": str(spit_dir), "commit": None, "source_tree": directory_tree_metadata(spit_dir)}
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref: "):
        ref = spit_dir / ".git" / text.removeprefix("ref: ")
        commit = ref.read_text(encoding="utf-8").strip() if ref.exists() else None
    else:
        commit = text
    return {"path": str(spit_dir), "commit": commit, "source_tree": directory_tree_metadata(spit_dir)}


def _external_file_metadata(config: dict[str, Any], root: Path) -> dict[str, Any]:
    values = {}
    for key, value in _iter_config_values(config):
        if key in {"base_model_path", "spit_weights", "weights"} and isinstance(value, str) and value:
            values[key] = file_metadata(value, root=root)
    return values


def _iter_config_values(value: Any, key: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            yield from _iter_config_values(child_value, str(child_key))
    elif isinstance(value, list):
        for child_value in value:
            yield from _iter_config_values(child_value, key)
    else:
        yield key, value


def _run_git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
