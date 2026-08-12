"""Run the long manual COCO benchmark for YOLO-superpixel strategies.

This script is intentionally not a pytest test. It launches expensive training
runs and writes a report for human inspection before moving experiments to HPC.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ["yolo12n.pt", "yolo26n.pt"]
DEFAULT_FUSIONS = ["add", "concat", "cross_attention"]
DEFAULT_SWAP_ALGORITHMS = [
    "torch_slic",
    "color_affinity",
    "gradient_affinity",
    "color_gradient_affinity",
]
DEFAULT_INPUT_CONTROL_ALGORITHMS = ["color_affinity", "torch_grid", "torch_voronoi"]
EDGE_REPRESENTATION_CHOICES = ["same", "boundary", "zero_centered", "boundary_zero_centered"]
INPUT_CHANNEL_MODE_CHOICES = ["adjacency", "soft_interior", "soft_boundary", "soft_boundary_interior"]
COMPLETION_MARKER = "yolo_superpixels_job_complete.json"
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


@dataclass
class BenchmarkJob:
    stage: str
    model: str
    variant: str
    command: list[str]
    run_dir: Path
    log_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobResult:
    stage: str
    model: str
    variant: str
    status: str
    exit_code: int | None
    seconds: float
    run_dir: Path
    log_path: Path
    command: list[str]
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the long manual COCO benchmark: input channels, neck injection, "
            "auxiliary loss, and input-channel algorithm-swap validation."
        )
    )
    parser.add_argument("--data", required=True, help="COCO Ultralytics data.yaml path.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--run-root", type=Path, default=Path("runs/coco_manual_benchmark"))
    parser.add_argument(
        "--stage",
        choices=["all", "rgb", "input", "swap", "neck", "auxiliary"],
        nargs="+",
        default=["all"],
        help="Subset of the benchmark to run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write manifest/report without launching training.")
    parser.add_argument("--rerun", action="store_true", help="Run jobs even when their output already exists.")
    parser.add_argument(
        "--exist-ok",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --exist-ok to Ultralytics training commands. Use --no-exist-ok to disable.",
    )

    parser.add_argument("--input-algorithm", default="torch_slic")
    parser.add_argument(
        "--input-channel-mode",
        choices=INPUT_CHANNEL_MODE_CHOICES,
        default="adjacency",
        help=(
            "Input-channel family for input-stage jobs. Use soft_interior, "
            "soft_boundary, or soft_boundary_interior for distance-map experiments."
        ),
    )
    parser.add_argument(
        "--input-control-algorithm",
        nargs="*",
        default=DEFAULT_INPUT_CONTROL_ALGORITHMS,
        help="Extra input-channel training controls. Use an empty value to disable.",
    )
    parser.add_argument(
        "--no-rgb-baseline",
        action="store_true",
        help="Skip paired RGB-only fine-tunes. Not recommended for scientific comparisons.",
    )
    parser.add_argument("--init-extra", choices=["zero", "tiny", "mean", "kaiming"], default="tiny")
    parser.add_argument(
        "--edge-representation",
        choices=EDGE_REPRESENTATION_CHOICES,
        default="zero_centered",
        help="Input-channel representation for training and swap validation.",
    )
    parser.add_argument("--swap-algorithm", nargs="+", default=DEFAULT_SWAP_ALGORITHMS)
    parser.add_argument("--swap-checkpoints", nargs="+", choices=["best", "last"], default=["best"])
    parser.add_argument("--include-spit-dense-swap", action="store_true")

    parser.add_argument("--neck-fusion", nargs="+", choices=DEFAULT_FUSIONS, default=DEFAULT_FUSIONS)
    parser.add_argument("--neck-token-provider", choices=["region", "spit_labels", "spit"], default="region")
    parser.add_argument("--neck-region-algorithm", choices=["torch_slic", "torch_grid", "torch_voronoi"], default="torch_slic")
    parser.add_argument("--neck-image-size", type=int, default=160)
    parser.add_argument("--allow-random-spit", action="store_true")
    parser.add_argument("--spit-weights", default=None)

    parser.add_argument("--aux-algorithm", default="torch_slic")
    parser.add_argument("--aux-loss-weight", type=float, default=0.05)

    parser.add_argument("--n-segments", type=int, default=300)
    parser.add_argument("--torch-slic-compactness", type=float, default=1.0)
    parser.add_argument("--torch-slic-iters", type=int, default=7)
    parser.add_argument("--soft-max-distance", type=int, default=32)
    parser.add_argument("--soft-boundary-width", type=float, default=6.0)
    parser.add_argument("--soft-affinity-threshold", type=float, default=0.5)

    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--lrf", type=float, default=None)
    parser.add_argument("--optimizer", default=None)
    parser.add_argument("--freeze", type=int, default=None)
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=float, default=None)
    parser.add_argument("--warmup-momentum", type=float, default=None)
    parser.add_argument("--warmup-bias-lr", type=float, default=None)
    return parser.parse_args()


def selected_stages(values: list[str]) -> set[str]:
    if "all" in values:
        return {"rgb", "input", "swap", "neck", "auxiliary"}
    return set(values)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def model_stem(model: str) -> str:
    return safe_name(Path(model).stem)


def command_to_text(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def command_hash(command: list[str]) -> str:
    return hashlib.sha256(json.dumps(command, ensure_ascii=False).encode("utf-8")).hexdigest()


def source_tree_metadata(root: Path = ROOT) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = list(iter_source_tree_files(root))
    total_bytes = 0
    errors: list[str] = []
    for path in files:
        rel = path.relative_to(root).as_posix()
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


def directory_tree_metadata(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {"path": str(root), "sha256": None, "files": 0, "bytes": 0, "errors": ["missing"]}
    digest = hashlib.sha256()
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and not is_external_tree_excluded(path)
    ]
    files = sorted(files, key=lambda path: path.relative_to(root).as_posix())
    total_bytes = 0
    errors: list[str] = []
    for path in files:
        rel = path.relative_to(root).as_posix()
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
        "path": str(root),
        "sha256": digest.hexdigest(),
        "files": len(files),
        "bytes": total_bytes,
        "errors": errors,
    }


def file_metadata(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = ROOT / file_path
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


def external_metadata(args: argparse.Namespace | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "spit_checkout": directory_tree_metadata(ROOT / "third_party" / "SPiT"),
    }
    if args is not None:
        spit_weights = file_metadata(getattr(args, "spit_weights", None))
        if spit_weights is not None:
            metadata["spit_weights"] = spit_weights
    return metadata


def external_fingerprint(args: argparse.Namespace | None = None) -> str:
    return hashlib.sha256(
        json.dumps(external_metadata(args), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def parse_gpu_device_ids(device: str | None) -> str | None:
    if device is None:
        return None
    normalized = str(device).strip().lower()
    if not normalized or normalized == "cpu":
        return None
    normalized = normalized.replace("cuda:", "")
    values = [part.strip() for part in normalized.split(",") if part.strip().isdigit()]
    if not values:
        return None
    return ",".join(values)


class GpuUsageMonitor:
    """Poll nvidia-smi while a subprocess is running."""

    def __init__(self, device: str | None, interval: float = 1.0) -> None:
        self.device_ids = parse_gpu_device_ids(device)
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.available = False
        self.peak_memory_mb: float | None = None
        self.max_utilization_percent: float | None = None
        self.utilization_sum = 0.0
        self.samples = 0
        self.energy_wh = 0.0
        self.last_sample_time: float | None = None
        self.last_power_w: float | None = None

    def __enter__(self) -> "GpuUsageMonitor":
        if self.device_ids is None:
            return self
        self.thread = threading.Thread(target=self._run, name="gpu-usage-monitor", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(self.interval * 2, 2.0))

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(self.interval)
        self.sample()

    def sample(self) -> None:
        command = [
            "nvidia-smi",
            f"--id={self.device_ids}",
            "--query-gpu=memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return
        if result.returncode != 0:
            return
        memory_values: list[float] = []
        utilization_values: list[float] = []
        power_values: list[float] = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                continue
            memory = parse_float(parts[0])
            utilization = parse_float(parts[1])
            power = parse_float(parts[2])
            if memory is not None:
                memory_values.append(memory)
            if utilization is not None:
                utilization_values.append(utilization)
            if power is not None:
                power_values.append(power)
        if not memory_values and not utilization_values and not power_values:
            return
        self.available = True
        now = time.perf_counter()
        total_memory = sum(memory_values)
        if self.peak_memory_mb is None or total_memory > self.peak_memory_mb:
            self.peak_memory_mb = total_memory
        if utilization_values:
            avg_utilization = sum(utilization_values) / len(utilization_values)
            self.max_utilization_percent = max(self.max_utilization_percent or 0.0, avg_utilization)
            self.utilization_sum += avg_utilization
            self.samples += 1
        total_power = sum(power_values)
        if self.last_sample_time is not None and self.last_power_w is not None:
            self.energy_wh += self.last_power_w * (now - self.last_sample_time) / 3600.0
        if power_values:
            self.last_sample_time = now
            self.last_power_w = total_power

    def metrics(self) -> dict[str, str]:
        if not self.available:
            return {}
        values: dict[str, str] = {}
        if self.peak_memory_mb is not None:
            values["peak_gpu_memory_mb"] = f"{self.peak_memory_mb:.0f}"
        if self.max_utilization_percent is not None:
            values["max_gpu_utilization_percent"] = f"{self.max_utilization_percent:.1f}"
        if self.samples:
            values["avg_gpu_utilization_percent"] = f"{self.utilization_sum / self.samples:.1f}"
        if self.energy_wh > 0:
            values["energy_wh"] = f"{self.energy_wh:.3f}"
        return values


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def iter_source_tree_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for name in SOURCE_TREE_INCLUDE_FILES:
        path = root / name
        if path.is_file() and not is_source_tree_excluded(path):
            files.append(path)
    for name in SOURCE_TREE_INCLUDE_DIRS:
        directory = root / name
        if not directory.is_dir():
            continue
        files.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and not is_source_tree_excluded(path)
        )
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def is_source_tree_excluded(path: Path) -> bool:
    if any(part in SOURCE_TREE_EXCLUDE_PARTS for part in path.parts):
        return True
    return path.suffix in SOURCE_TREE_EXCLUDE_SUFFIXES


def is_external_tree_excluded(path: Path) -> bool:
    if any(part in EXTERNAL_TREE_EXCLUDE_PARTS for part in path.parts):
        return True
    return path.suffix in EXTERNAL_TREE_EXCLUDE_SUFFIXES


def input_training_algorithms(args: argparse.Namespace) -> list[str]:
    values: list[str] = []
    for name in [args.input_algorithm, *args.input_control_algorithm]:
        normalized = str(name).strip()
        if normalized and normalized.lower() not in {"none", "rgb"} and normalized not in values:
            values.append(normalized)
    return values


def common_train_args(args: argparse.Namespace, project: Path, name: str) -> list[str]:
    command = [
        "--data",
        str(args.data),
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--device",
        str(args.device),
        "--project",
        str(project),
        "--name",
        name,
    ]
    if args.exist_ok:
        command.append("--exist-ok")
    optional = {
        "--lr0": args.lr0,
        "--lrf": args.lrf,
        "--optimizer": args.optimizer,
        "--freeze": args.freeze,
        "--patience": args.patience,
        "--weight-decay": args.weight_decay,
        "--warmup-epochs": args.warmup_epochs,
        "--warmup-momentum": args.warmup_momentum,
        "--warmup-bias-lr": args.warmup_bias_lr,
    }
    for flag, value in optional.items():
        if value is not None:
            command.extend([flag, str(value)])
    if args.cos_lr:
        command.append("--cos-lr")
    return command


def build_training_jobs(args: argparse.Namespace) -> list[BenchmarkJob]:
    stages = selected_stages(args.stage)
    run_root = Path(args.run_root)
    logs_dir = run_root / "logs"
    jobs: list[BenchmarkJob] = []

    if not args.no_rgb_baseline and stages.intersection({"rgb", "input", "neck", "auxiliary"}):
        project = run_root / "rgb_baseline"
        for model in args.models:
            name = f"rgb_{model_stem(model)}"
            command = [
                sys.executable,
                str(ROOT / "examples" / "train_ultralytics.py"),
                "--model",
                model,
                *common_train_args(args, project, name),
            ]
            jobs.append(
                BenchmarkJob(
                    stage="rgb_baseline",
                    model=model,
                    variant="rgb_only",
                    command=command,
                    run_dir=project / name,
                    log_path=logs_dir / f"{name}.log",
                    metadata={"baseline": True},
                )
            )

    if "input" in stages:
        project = run_root / "input_channels"
        for algorithm in input_training_algorithms(args):
            for model in args.models:
                mode_suffix = "" if args.input_channel_mode == "adjacency" else f"_{args.input_channel_mode}"
                name = f"input_{model_stem(model)}_{safe_name(algorithm)}{mode_suffix}"
                command = [
                    sys.executable,
                    str(ROOT / "examples" / "train_ultralytics.py"),
                    "--model",
                    model,
                    *common_train_args(args, project, name),
                    "--algorithm",
                    algorithm,
                    "--channel-mode",
                    args.input_channel_mode,
                    "--init-extra",
                    args.init_extra,
                    "--edge-representation",
                    args.edge_representation,
                    "--n-segments",
                    str(args.n_segments),
                    "--torch-slic-compactness",
                    str(args.torch_slic_compactness),
                    "--torch-slic-iters",
                    str(args.torch_slic_iters),
                    "--soft-max-distance",
                    str(args.soft_max_distance),
                    "--soft-boundary-width",
                    str(args.soft_boundary_width),
                    "--soft-affinity-threshold",
                    str(args.soft_affinity_threshold),
                ]
                jobs.append(
                    BenchmarkJob(
                        stage="input",
                        model=model,
                        variant=f"{algorithm}_{args.input_channel_mode}",
                        command=command,
                        run_dir=project / name,
                        log_path=logs_dir / f"{name}.log",
                    )
                )

    if "neck" in stages:
        project = run_root / "neck_injection"
        for model in args.models:
            for fusion in args.neck_fusion:
                variant = f"{fusion}_{args.neck_token_provider}_{args.neck_region_algorithm}"
                name = f"neck_{model_stem(model)}_{safe_name(variant)}"
                command = [
                    sys.executable,
                    str(ROOT / "examples" / "train_ultralytics_neck.py"),
                    "--model",
                    model,
                    *common_train_args(args, project, name),
                    "--fusion",
                    fusion,
                    "--token-provider",
                    args.neck_token_provider,
                    "--region-algorithm",
                    args.neck_region_algorithm,
                    "--region-n-segments",
                    str(args.n_segments),
                    "--region-compactness",
                    str(args.torch_slic_compactness),
                    "--region-num-iters",
                    str(args.torch_slic_iters),
                    "--spit-image-size",
                    str(args.neck_image_size),
                ]
                if args.spit_weights:
                    command.extend(["--spit-weights", args.spit_weights])
                if args.allow_random_spit:
                    command.append("--allow-random-spit")
                jobs.append(
                    BenchmarkJob(
                        stage="neck",
                        model=model,
                        variant=variant,
                        command=command,
                        run_dir=project / name,
                        log_path=logs_dir / f"{name}.log",
                    )
                )

    if "auxiliary" in stages:
        project = run_root / "auxiliary_loss"
        for model in args.models:
            name = f"aux_{model_stem(model)}_{safe_name(args.aux_algorithm)}"
            command = [
                sys.executable,
                str(ROOT / "examples" / "train_ultralytics_auxiliary.py"),
                "--model",
                model,
                *common_train_args(args, project, name),
                "--algorithm",
                args.aux_algorithm,
                "--n-segments",
                str(args.n_segments),
                "--compactness",
                str(args.torch_slic_compactness),
                "--num-iters",
                str(args.torch_slic_iters),
                "--loss-weight",
                str(args.aux_loss_weight),
                "--target-mode",
                "labels" if args.aux_algorithm in {"torch_slic", "gpu_slic"} else "auto",
                "--loss-type",
                "balanced_bce",
            ]
            jobs.append(
                BenchmarkJob(
                    stage="auxiliary",
                    model=model,
                    variant=args.aux_algorithm,
                    command=command,
                    run_dir=project / name,
                    log_path=logs_dir / f"{name}.log",
                )
            )
    return jobs


def input_checkpoint_paths(args: argparse.Namespace) -> list[tuple[str, Path]]:
    project = Path(args.run_root) / "input_channels"
    checkpoints: list[tuple[str, Path]] = []
    for model in args.models:
        mode_suffix = "" if args.input_channel_mode == "adjacency" else f"_{args.input_channel_mode}"
        name = f"input_{model_stem(model)}_{safe_name(args.input_algorithm)}{mode_suffix}"
        for checkpoint_name in args.swap_checkpoints:
            checkpoint = project / name / "weights" / f"{checkpoint_name}.pt"
            label = f"{model_stem(model)}_{args.input_channel_mode}_{checkpoint_name}"
            checkpoints.append((label, checkpoint))
    return checkpoints


def build_swap_job(args: argparse.Namespace) -> BenchmarkJob:
    run_root = Path(args.run_root)
    output = run_root / "reports" / "input_channel_swap_matrix.csv"
    algorithms = list(args.swap_algorithm)
    if args.include_spit_dense_swap and "spit_dense_adjacency" not in algorithms:
        algorithms.append("spit_dense_adjacency")

    command = [
        sys.executable,
        str(ROOT / "scripts" / "validate_algorithm_swap_matrix.py"),
        "--data",
        str(args.data),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--device",
        str(args.device),
        "--project",
        str(run_root / "swap_validation_runs"),
        "--output",
        str(output),
        "--algorithm",
        *algorithms,
        "--channel-mode",
        args.input_channel_mode,
        "--edge-representation",
        args.edge_representation,
        "--n-segments",
        str(args.n_segments),
        "--torch-slic-compactness",
        str(args.torch_slic_compactness),
        "--torch-slic-iters",
        str(args.torch_slic_iters),
        "--soft-max-distance",
        str(args.soft_max_distance),
        "--soft-boundary-width",
        str(args.soft_boundary_width),
        "--soft-affinity-threshold",
        str(args.soft_affinity_threshold),
    ]
    for label, checkpoint in input_checkpoint_paths(args):
        command.extend(["--checkpoint", f"{label}={checkpoint}"])
    return BenchmarkJob(
        stage="swap",
        model="input_checkpoints",
        variant="algorithm_swap",
        command=command,
        run_dir=output.parent,
        log_path=run_root / "logs" / "input_channel_swap_matrix.log",
        metadata={"output": output},
    )


def _image_suffixes() -> set[str]:
    return {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def _resolve_dataset_path(data_yaml: Path, value: Any, base_path: Path) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        paths: list[Path] = []
        for item in value:
            paths.extend(_resolve_dataset_path(data_yaml, item, base_path))
        return paths
    path = Path(str(value))
    if path.is_absolute():
        return [path]
    return [base_path / path]


def _image_paths_from_entry(entry: Path) -> list[Path]:
    if entry.is_file() and entry.suffix.lower() == ".txt":
        paths = []
        for line in entry.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if stripped:
                path = Path(stripped)
                paths.append(path if path.is_absolute() else entry.parent / path)
        return paths
    if entry.is_file() and entry.suffix.lower() in _image_suffixes():
        return [entry]
    if entry.is_dir():
        return [path for path in entry.rglob("*") if path.suffix.lower() in _image_suffixes()]
    return []


def _label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = parts.index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    except ValueError:
        return image_path.parent.parent / "labels" / image_path.with_suffix(".txt").name


def _cache_path_for_entry(entry: Path) -> Path | None:
    parts = list(entry.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return None
    parts[index] = "labels"
    return Path(*parts).with_suffix(".cache")


def dataset_split_snapshot_from_cache(entries: list[Path]) -> dict[str, Any] | None:
    cache_paths = [_cache_path_for_entry(entry) for entry in entries]
    cache_paths = [path for path in cache_paths if path is not None and path.exists()]
    if len(cache_paths) != 1:
        return None
    try:
        import numpy as np

        cache = np.load(cache_paths[0], allow_pickle=True).item()
        labels = cache.get("labels") or []
        object_count = sum(int(len(item.get("cls", []))) for item in labels if isinstance(item, dict))
        return {
            "entries": [str(path) for path in entries],
            "images": len(labels),
            "label_files": len(labels),
            "objects": object_count,
            "source": str(cache_paths[0]),
        }
    except Exception:
        return None


def _count_label_objects(label_path: Path) -> int:
    return sum(1 for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())


def count_label_objects(label_paths: list[Path]) -> int:
    if not label_paths:
        return 0
    workers = min(32, max(1, (os.cpu_count() or 1) * 4), len(label_paths))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return sum(executor.map(_count_label_objects, label_paths))


def dataset_split_snapshot(data_yaml: Path, split_value: Any, base_path: Path) -> dict[str, Any]:
    entries = _resolve_dataset_path(data_yaml, split_value, base_path)
    cached = dataset_split_snapshot_from_cache(entries)
    if cached is not None:
        return cached
    image_paths: list[Path] = []
    for entry in entries:
        image_paths.extend(_image_paths_from_entry(entry))
    label_paths = [_label_path_for_image(path) for path in image_paths]
    existing_labels = [path for path in label_paths if path.exists()]
    object_count = count_label_objects(existing_labels)
    return {
        "entries": [str(path) for path in entries],
        "images": len(image_paths),
        "label_files": len(existing_labels),
        "objects": object_count,
    }


def dataset_snapshot(data_yaml: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        return {"data": str(data_yaml), "error": f"PyYAML unavailable: {exc}"}
    if not data_yaml.exists():
        return {"data": str(data_yaml), "error": "data.yaml does not exist"}
    try:
        values = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return {"data": str(data_yaml), "error": f"could not parse data.yaml: {exc}"}
    base_path = Path(values.get("path") or data_yaml.parent)
    if not base_path.is_absolute():
        base_path = data_yaml.parent / base_path
    snapshot = {
        "data": str(data_yaml),
        "path": str(base_path),
        "nc": values.get("nc"),
        "names": values.get("names"),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        if split in values:
            snapshot["splits"][split] = dataset_split_snapshot(data_yaml, values.get(split), base_path)
    return snapshot


def read_last_results_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def last_epoch_from_results(run_dir: Path) -> int | None:
    row = read_last_results_csv(run_dir / "results.csv")
    value = row.get("epoch")
    if value in {None, ""}:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def completion_marker_path(run_dir: Path) -> Path:
    return run_dir / COMPLETION_MARKER


def read_completion_marker(run_dir: Path) -> dict[str, Any]:
    marker_path = completion_marker_path(run_dir)
    if not marker_path.exists():
        return {}
    try:
        return json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_completion_marker(job: BenchmarkJob, args: argparse.Namespace, result: "JobResult") -> None:
    marker = {
        "status": "ok",
        "stage": job.stage,
        "model": job.model,
        "variant": job.variant,
        "requested_epochs": int(args.epochs),
        "last_epoch": last_epoch_from_results(job.run_dir),
        "command_hash": command_hash(job.command),
        "source_tree_sha256": source_tree_metadata()["sha256"],
        "external_fingerprint": external_fingerprint(args),
        "command": job.command,
        "seconds": result.seconds,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    completion_marker_path(job.run_dir).write_text(json.dumps(marker, indent=2), encoding="utf-8")


def train_completion_problem(
    job: BenchmarkJob,
    requested_epochs: int,
    args: argparse.Namespace | None = None,
) -> str | None:
    if not (job.run_dir / "results.csv").exists():
        return "missing results.csv"
    if not (job.run_dir / "weights" / "best.pt").exists():
        return "missing weights/best.pt"
    marker = read_completion_marker(job.run_dir)
    if marker.get("status") != "ok":
        return f"missing or invalid {COMPLETION_MARKER}"
    if marker.get("command_hash") != command_hash(job.command):
        return "completion marker command_hash does not match current command"
    if marker.get("source_tree_sha256") != source_tree_metadata()["sha256"]:
        return "completion marker source_tree_sha256 does not match current source tree"
    if marker.get("external_fingerprint") != external_fingerprint(args):
        return "completion marker external_fingerprint does not match current external dependencies"
    if int(marker.get("requested_epochs", -1)) != int(requested_epochs):
        return "completion marker requested_epochs does not match current request"
    last_epoch = last_epoch_from_results(job.run_dir)
    if last_epoch is None:
        return "could not read last epoch from results.csv"
    if last_epoch < int(requested_epochs):
        return f"last epoch {last_epoch} is below requested epochs {requested_epochs}"
    if marker.get("last_epoch") is not None and int(marker["last_epoch"]) < int(requested_epochs):
        return "completion marker records an incomplete epoch budget"
    return None


def pick_metric(row: dict[str, str], key: str) -> str:
    return row.get(key, "")


def metrics_for_run(run_dir: Path) -> dict[str, Any]:
    row = read_last_results_csv(run_dir / "results.csv")
    metrics = {
        "precision": pick_metric(row, "metrics/precision(B)"),
        "recall": pick_metric(row, "metrics/recall(B)"),
        "map50": pick_metric(row, "metrics/mAP50(B)"),
        "map50_95": pick_metric(row, "metrics/mAP50-95(B)"),
        "train_sp_loss": pick_metric(row, "train/sp_loss"),
        "val_sp_loss": pick_metric(row, "val/sp_loss"),
        "best_pt": str(run_dir / "weights" / "best.pt"),
        "last_pt": str(run_dir / "weights" / "last.pt"),
        "best_exists": (run_dir / "weights" / "best.pt").exists(),
        "last_exists": (run_dir / "weights" / "last.pt").exists(),
    }
    return metrics


def run_job(job: BenchmarkJob, args: argparse.Namespace) -> JobResult:
    start = time.perf_counter()
    job.log_path.parent.mkdir(parents=True, exist_ok=True)

    if job.stage != "swap" and not args.rerun:
        completion_problem = train_completion_problem(job, requested_epochs=args.epochs, args=args)
        if completion_problem is None:
            marker = read_completion_marker(job.run_dir)
            metrics = metrics_for_run(job.run_dir)
            metrics["last_epoch"] = last_epoch_from_results(job.run_dir)
            return JobResult(
                stage=job.stage,
                model=job.model,
                variant=job.variant,
                status="skipped_existing",
                exit_code=0,
                seconds=0.0,
                run_dir=job.run_dir,
                log_path=job.log_path,
                command=job.command,
                metrics=metrics,
                notes=f"Existing completed run found: {marker.get('completed_at', '')}.",
            )
        if (job.run_dir / "results.csv").exists() or (job.run_dir / "weights" / "best.pt").exists():
            print(f"    existing run is not complete: {completion_problem}")

    if args.dry_run:
        job.log_path.write_text(command_to_text(job.command) + "\n", encoding="utf-8")
        return JobResult(
            stage=job.stage,
            model=job.model,
            variant=job.variant,
            status="dry_run",
            exit_code=None,
            seconds=0.0,
            run_dir=job.run_dir,
            log_path=job.log_path,
            command=job.command,
            metrics={},
        )

    monitor = GpuUsageMonitor(args.device)
    with job.log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
        log_handle.write(command_to_text(job.command) + "\n\n")
        log_handle.flush()
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        with monitor:
            process = subprocess.run(
                job.command,
                cwd=str(ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                check=False,
            )

    seconds = time.perf_counter() - start
    status = "ok" if process.returncode == 0 else "failed"
    result = JobResult(
        stage=job.stage,
        model=job.model,
        variant=job.variant,
        status=status,
        exit_code=process.returncode,
        seconds=seconds,
        run_dir=job.run_dir,
        log_path=job.log_path,
        command=job.command,
        metrics={**metrics_for_run(job.run_dir), **monitor.metrics()},
    )
    if status == "ok" and job.stage != "swap":
        last_epoch = last_epoch_from_results(job.run_dir)
        result.metrics["last_epoch"] = last_epoch
        if last_epoch is None or last_epoch < int(args.epochs):
            result.status = "incomplete"
            result.exit_code = 1
            result.notes = f"last epoch {last_epoch} is below requested epochs {args.epochs}"
        else:
            write_completion_marker(job, args, result)
    return result


def run_swap_job(job: BenchmarkJob, args: argparse.Namespace) -> JobResult:
    output = Path(job.metadata["output"])
    if args.dry_run:
        return run_job(job, args)
    if output.exists() and not args.rerun:
        return JobResult(
            stage=job.stage,
            model=job.model,
            variant=job.variant,
            status="skipped_existing",
            exit_code=0,
            seconds=0.0,
            run_dir=job.run_dir,
            log_path=job.log_path,
            command=job.command,
            metrics={"output": str(output), "markdown": str(output.with_suffix(".md"))},
            notes="Existing swap matrix found.",
        )
    missing = [str(path) for _label, path in input_checkpoint_paths(args) if not path.exists()]
    if missing:
        return JobResult(
            stage=job.stage,
            model=job.model,
            variant=job.variant,
            status="missing_checkpoints",
            exit_code=1,
            seconds=0.0,
            run_dir=job.run_dir,
            log_path=job.log_path,
            command=job.command,
            metrics={"missing": "; ".join(missing)},
            notes="Run input-channel training first, or remove --stage swap.",
        )
    return run_job(job, args)


def write_manifest(
    jobs: list[BenchmarkJob],
    swap_job: BenchmarkJob | None,
    output: Path,
    args: argparse.Namespace | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    values = [
        {
            "stage": job.stage,
            "model": job.model,
            "variant": job.variant,
            "run_dir": str(job.run_dir),
            "log_path": str(job.log_path),
            "command": job.command,
        }
        for job in jobs + ([swap_job] if swap_job is not None else [])
    ]
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_tree": source_tree_metadata(),
        "external": external_metadata(args),
        "external_fingerprint": external_fingerprint(args),
        "jobs": values,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_results_csv(results: list[JobResult], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "stage",
        "model",
        "variant",
        "status",
        "exit_code",
        "seconds",
        "peak_gpu_memory_mb",
        "max_gpu_utilization_percent",
        "avg_gpu_utilization_percent",
        "energy_wh",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "train_sp_loss",
        "val_sp_loss",
        "last_epoch",
        "best_exists",
        "last_exists",
        "run_dir",
        "log_path",
        "notes",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {
                "stage": result.stage,
                "model": result.model,
                "variant": result.variant,
                "status": result.status,
                "exit_code": result.exit_code if result.exit_code is not None else "",
                "seconds": f"{result.seconds:.1f}",
                "run_dir": result.run_dir,
                "log_path": result.log_path,
                "notes": result.notes,
            }
            for key in fields:
                if key in result.metrics:
                    row[key] = result.metrics[key]
            writer.writerow(row)


def read_swap_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_report(
    results: list[JobResult],
    swap_output: Path | None,
    output: Path,
    dry_run: bool,
    dataset_info: dict[str, Any] | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    failures = [result for result in results if result.status not in {"ok", "skipped_existing", "dry_run"}]
    if dry_run:
        verdict = "DRY RUN: commands were written, but no training was launched."
    elif failures:
        verdict = "NOT READY: at least one run failed or required outputs are missing."
    else:
        verdict = "READY FOR HUMAN INSPECTION: all requested jobs completed or were already present."

    lines = [
        "# COCO Manual YOLO-Superpixel Benchmark",
        "",
        f"Verdict: **{verdict}**",
        "",
        "## Dataset Snapshot",
        "",
    ]
    if dataset_info:
        if "error" in dataset_info:
            lines.append(f"- Data: `{dataset_info.get('data', '')}`")
            lines.append(f"- Error: {dataset_info['error']}")
        else:
            lines.append(f"- Data: `{dataset_info.get('data', '')}`")
            lines.append(f"- Path: `{dataset_info.get('path', '')}`")
            for split, split_info in dataset_info.get("splits", {}).items():
                source = f", source `{split_info['source']}`" if split_info.get("source") else ""
                lines.append(
                    f"- {split}: {split_info.get('images', 0)} images, "
                    f"{split_info.get('label_files', 0)} label files, "
                    f"{split_info.get('objects', 0)} objects{source}"
                )
    else:
        lines.append("- Not collected.")
    lines.extend(
        [
            "",
            "## Training Runs",
            "",
            "| Stage | Model | Variant | Status | Time (s) | Peak VRAM MB | GPU util avg/max | mAP50-95 | mAP50 | P | R | sp_loss | Weights | Log |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for result in results:
        if result.stage == "swap":
            continue
        metrics = result.metrics
        sp_loss = metrics.get("train_sp_loss") or metrics.get("val_sp_loss") or ""
        best_exists = "best" if metrics.get("best_exists") else "missing"
        last_exists = "last" if metrics.get("last_exists") else "missing"
        gpu_util = ""
        if metrics.get("avg_gpu_utilization_percent") or metrics.get("max_gpu_utilization_percent"):
            gpu_util = (
                f"{metrics.get('avg_gpu_utilization_percent', '')}/"
                f"{metrics.get('max_gpu_utilization_percent', '')}"
            )
        lines.append(
            f"| {result.stage} | `{result.model}` | `{result.variant}` | {result.status} | "
            f"{result.seconds:.1f} | {metrics.get('peak_gpu_memory_mb', '')} | {gpu_util} | "
            f"{metrics.get('map50_95', '')} | {metrics.get('map50', '')} | "
            f"{metrics.get('precision', '')} | {metrics.get('recall', '')} | "
            f"{sp_loss} | {best_exists}/{last_exists}, epoch={metrics.get('last_epoch', '')} | `{result.log_path}` |"
        )

    if swap_output is not None:
        lines.extend(["", "## Input-Channel Algorithm Swap", ""])
        swap_rows = read_swap_rows(swap_output)
        if swap_rows:
            lines.extend(
                [
                    "| Checkpoint | Algorithm | Status | mAP50-95 | mAP50 | P | R | Seconds |",
                    "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in swap_rows:
                lines.append(
                    f"| {row.get('checkpoint', '')} | `{row.get('algorithm', '')}` | "
                    f"{row.get('status', '')} | {row.get('map50_95', '')} | "
                    f"{row.get('map50', '')} | {row.get('precision', '')} | "
                    f"{row.get('recall', '')} | {row.get('seconds', '')} |"
                )
        else:
            lines.append(f"No swap matrix yet: `{swap_output}`")

    lines.extend(
        [
            "",
            "## Manual Review Checklist",
            "",
            "- Compare input-channel mAP against the paired RGB fine-tune on the same COCO split.",
            "- Treat the paired RGB fine-tune as the primary baseline, not the untouched pretrained checkpoint.",
            "- Treat grid/Voronoi as structural controls and color/gradient affinity as non-superpixel controls.",
            "- Inspect swap matrix drops: a large drop means the trained model depends on a specific channel semantics.",
            "- Do not use swap matrix results as evidence that one superpixel method is superior.",
            "- Compare neck `add`, `concat`, and `cross_attention` not only by mAP but also by wall time and GPU memory.",
            "- For auxiliary loss, inspect `train/sp_loss`, detection metrics, and whether the loss destabilizes training.",
            "- Keep raw/full SPiT out of the main matrix unless a separate compute go/no-go passes.",
            "- If all runs complete and the curves are sane, the setup is a reasonable candidate for HPC scaling.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    report_dir = args.run_root / "reports"
    dataset_info = dataset_snapshot(Path(args.data))
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "dataset_snapshot.json").write_text(json.dumps(dataset_info, indent=2), encoding="utf-8")
    jobs = build_training_jobs(args)
    stages = selected_stages(args.stage)
    swap_job = build_swap_job(args) if "swap" in stages else None
    write_manifest(jobs, swap_job, report_dir / "manifest.json", args=args)

    results: list[JobResult] = []
    results_csv = report_dir / "summary.csv"
    report_md = report_dir / "summary.md"
    swap_output = Path(swap_job.metadata["output"]) if swap_job is not None else None

    for job in jobs:
        print(f"==> {job.stage}: {job.model} / {job.variant}")
        result = run_job(job, args)
        results.append(result)
        print(f"    {result.status} in {result.seconds:.1f}s; log={result.log_path}")
        write_results_csv(results, results_csv)
        write_report(results, swap_output, report_md, args.dry_run, dataset_info=dataset_info)
        if result.status == "failed":
            print("Stopping after failure. Re-run with --stage to resume after fixing the issue.")
            break

    if swap_job is not None and not any(result.status == "failed" for result in results):
        print("==> swap: input-channel checkpoints / algorithm_swap")
        swap_result = run_swap_job(swap_job, args)
        results.append(swap_result)
        print(f"    {swap_result.status} in {swap_result.seconds:.1f}s; log={swap_result.log_path}")

    write_results_csv(results, results_csv)
    write_report(results, swap_output, report_md, args.dry_run, dataset_info=dataset_info)
    print(f"saved manifest: {report_dir / 'manifest.json'}")
    print(f"saved summary: {results_csv}")
    print(f"saved report: {report_md}")


if __name__ == "__main__":
    main()
