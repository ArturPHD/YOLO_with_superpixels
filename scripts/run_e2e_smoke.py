"""End-to-end smoke test for YOLO-superpixel training lifecycles.

The parent process creates a tiny detection dataset, then launches child
processes for:

1. train a few batches;
2. load the saved checkpoint in a new process and validate;
3. load the saved checkpoint in another new process and predict.

This is intentionally a manual/opt-in smoke, not a default pytest test.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    # Child processes execute this file from scripts/, while the package lives in src/.
    sys.path.insert(0, str(SRC))

STRATEGIES = ("input", "neck", "backbone_native", "backbone_isolated", "auxiliary", "structural")
INPUT_CHANNEL_MODES = ("adjacency", "soft_interior", "soft_boundary", "soft_boundary_interior")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=STRATEGIES, nargs="+", default=["input"])
    parser.add_argument("--model", default="yolo12n.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=96)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--scratch-dir", type=Path, default=ROOT / "runs" / "e2e_smoke")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--input-channel-mode", choices=INPUT_CHANNEL_MODES, default="adjacency")
    parser.add_argument("--edge-representation", default="zero_centered")
    parser.add_argument("--soft-max-distance", type=int, default=16)
    parser.add_argument("--soft-boundary-width", type=float, default=4.0)
    parser.add_argument("--soft-affinity-threshold", type=float, default=0.5)
    parser.add_argument("--worker", choices=["train", "val", "predict"], default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--run-root", default=None)
    return parser.parse_args()


def make_dataset(dataset_dir: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(20260723)
    first_image: Path | None = None
    for split in ("train", "val"):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in range(2):
            image = rng.integers(0, 255, size=(96, 96, 3), dtype=np.uint8)
            left = 20 + index * 8
            right = left + 38
            image[24:72, left:right] = np.array([220, 60 + index * 40, 80], dtype=np.uint8)
            image_path = image_dir / f"sample_{index}.jpg"
            Image.fromarray(image).save(image_path)
            (label_dir / f"sample_{index}.txt").write_text("0 0.5 0.5 0.42 0.50\n", encoding="utf-8")
            first_image = first_image or image_path

    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_dir),
                "train": "images/train",
                "val": "images/val",
                "nc": 1,
                "names": ["object"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    if first_image is None:
        raise RuntimeError("smoke dataset creation failed")
    return data_yaml, first_image


def ensure_cuda_if_requested(device: str, require_cuda: bool) -> None:
    if not require_cuda:
        return
    if str(device).lower() == "cpu":
        raise RuntimeError("--require-cuda was set but --device=cpu was requested")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("--require-cuda was set but torch.cuda.is_available() is false")


def input_algorithm_kwargs() -> dict[str, Any]:
    return {"n_segments": 16, "compactness": 1.0, "num_iters": 2}


def neck_config():
    from yolo_superpixels import NeckInjectionConfig

    return NeckInjectionConfig(
        fusion="add",
        token_provider="region",
        region_algorithm="torch_slic",
        region_n_segments=16,
        region_num_iters=2,
        emb_dim=16,
        attention_heads=2,
        image_size=64,
        validation_imgsz=64,
    )


def backbone_config(placement: str):
    from yolo_superpixels import BackboneCorrespondenceConfig

    return BackboneCorrespondenceConfig(
        placement=placement,
        fusion="overlap_attention",
        token_provider="region",
        region_algorithm="torch_slic",
        region_n_segments=16,
        region_num_iters=2,
        emb_dim=16,
        attention_heads=2,
        image_size=64,
        validation_imgsz=64,
    )


def auxiliary_config():
    from yolo_superpixels import AuxiliaryLossConfig

    return AuxiliaryLossConfig(
        algorithm_names=["torch_slic"],
        algorithm_kwargs=input_algorithm_kwargs(),
        loss_weight=0.05,
        target_mode="labels",
        loss_type="balanced_bce",
    )


def structural_config():
    from yolo_superpixels import StructuralSupervisionConfig

    return StructuralSupervisionConfig(
        lambda_fg=0.05,
        lambda_sp=0.05,
        region_algorithm="torch_slic",
        region_kwargs={"n_segments": 16, "compactness": 1.0, "num_iters": 2},
        purity_threshold=0.9,
        min_region_size=1,
        auxiliary_init_seed=0,
    )


def build_model(
    strategy: str,
    model_path: str,
    edge_representation: str,
    input_channel_mode: str = "adjacency",
    soft_max_distance: int = 16,
    soft_boundary_width: float = 4.0,
    soft_affinity_threshold: float = 0.5,
):
    if strategy == "input":
        from yolo_superpixels import SuperpixelYOLO

        return SuperpixelYOLO(
            model_path,
            algorithm=["torch_slic"],
            algorithm_kwargs=[input_algorithm_kwargs()],
            channel_mode=input_channel_mode,
            init_extra="tiny",
            edge_representation=edge_representation,
            soft_max_distance=soft_max_distance,
            soft_boundary_width=soft_boundary_width,
            soft_affinity_threshold=soft_affinity_threshold,
        )
    if strategy == "neck":
        from yolo_superpixels import SuperpixelNeckYOLO

        return SuperpixelNeckYOLO(model_path, config=neck_config())
    if strategy in {"backbone_native", "backbone_isolated"}:
        from yolo_superpixels import SuperpixelBackboneYOLO

        placement = "native_backbone" if strategy == "backbone_native" else "isolated_p3"
        return SuperpixelBackboneYOLO(model_path, config=backbone_config(placement))
    if strategy == "auxiliary":
        from yolo_superpixels import SuperpixelAuxiliaryYOLO

        return SuperpixelAuxiliaryYOLO(model_path, config=auxiliary_config())
    if strategy == "structural":
        from yolo_superpixels import SuperpixelStructuralYOLO

        return SuperpixelStructuralYOLO(model_path, config=structural_config())
    raise ValueError(f"Unknown strategy: {strategy}")


def run_project(run_root: str | Path, phase: str) -> str:
    """Return an absolute project path so Ultralytics does not prepend runs/detect."""

    return str((Path(run_root).resolve() / phase))


def worker_train(args: argparse.Namespace) -> None:
    if args.data is None or args.run_root is None:
        raise ValueError("--data and --run-root are required for train worker")
    ensure_cuda_if_requested(args.device, args.require_cuda)
    model = build_model(
        args.strategy[0],
        args.model,
        args.edge_representation,
        args.input_channel_mode,
        args.soft_max_distance,
        args.soft_boundary_width,
        args.soft_affinity_threshold,
    )
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=0,
        device=args.device,
        project=run_project(args.run_root, "train"),
        name=args.strategy[0],
        exist_ok=True,
        plots=False,
        val=True,
        verbose=False,
    )


def worker_val(args: argparse.Namespace) -> None:
    if args.data is None or args.checkpoint is None or args.run_root is None:
        raise ValueError("--data, --checkpoint, and --run-root are required for val worker")
    ensure_cuda_if_requested(args.device, args.require_cuda)
    model = build_model(
        args.strategy[0],
        args.checkpoint,
        args.edge_representation,
        args.input_channel_mode,
        args.soft_max_distance,
        args.soft_boundary_width,
        args.soft_affinity_threshold,
    )
    val_kwargs = {
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": 0,
        "device": args.device,
        "project": run_project(args.run_root, "val"),
        "name": args.strategy[0],
        "verbose": False,
        "plots": False,
    }
    if args.strategy[0] == "input":
        model.val(extra_channels_enabled=True, **val_kwargs)
    elif args.strategy[0] in {"neck", "backbone_native", "backbone_isolated"}:
        model.val(injection_enabled=True, **val_kwargs)
    else:
        model.val(**val_kwargs)


def worker_predict(args: argparse.Namespace) -> None:
    if args.checkpoint is None or args.source is None or args.run_root is None:
        raise ValueError("--checkpoint, --source, and --run-root are required for predict worker")
    ensure_cuda_if_requested(args.device, args.require_cuda)
    model = build_model(
        args.strategy[0],
        args.checkpoint,
        args.edge_representation,
        args.input_channel_mode,
        args.soft_max_distance,
        args.soft_boundary_width,
        args.soft_affinity_threshold,
    )
    predict_kwargs = {
        "source": args.source,
        "imgsz": args.imgsz,
        "device": args.device,
        "project": run_project(args.run_root, "predict"),
        "name": args.strategy[0],
        "save": True,
        "verbose": False,
    }
    if args.strategy[0] == "input":
        model.predict(extra_channels_enabled=True, **predict_kwargs)
    elif args.strategy[0] in {"neck", "backbone_native", "backbone_isolated"}:
        model.predict(injection_enabled=True, **predict_kwargs)
    else:
        model.predict(**predict_kwargs)


def run_child(args: argparse.Namespace, strategy: str, phase: str, extra: list[str]) -> subprocess.CompletedProcess:
    command = [
        args.python,
        str(Path(__file__).resolve()),
        "--worker",
        phase,
        "--strategy",
        strategy,
        "--model",
        args.model,
        "--device",
        args.device,
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--epochs",
        str(args.epochs),
        "--edge-representation",
        args.edge_representation,
        "--input-channel-mode",
        args.input_channel_mode,
        "--soft-max-distance",
        str(args.soft_max_distance),
        "--soft-boundary-width",
        str(args.soft_boundary_width),
        "--soft-affinity-threshold",
        str(args.soft_affinity_threshold),
        *extra,
    ]
    if args.require_cuda:
        command.append("--require-cuda")
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(ROOT))
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.run(
        command,
        cwd=str(ROOT),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=args.timeout,
        check=False,
    )


def run_strategy(args: argparse.Namespace, strategy: str, data_yaml: Path, source: Path) -> dict[str, Any]:
    run_root = args.scratch_dir / strategy
    run_root.mkdir(parents=True, exist_ok=True)
    phases = []
    train = run_child(
        args,
        strategy,
        "train",
        ["--data", str(data_yaml), "--run-root", str(run_root)],
    )
    phases.append(phase_result("train", train))
    if train.returncode != 0:
        return {"strategy": strategy, "status": "failed", "phases": phases}

    checkpoint = run_root / "train" / strategy / "weights" / "best.pt"
    if not checkpoint.exists():
        return {
            "strategy": strategy,
            "status": "failed",
            "phases": phases,
            "error": f"missing checkpoint: {checkpoint}",
        }

    val = run_child(
        args,
        strategy,
        "val",
        ["--data", str(data_yaml), "--checkpoint", str(checkpoint), "--run-root", str(run_root)],
    )
    phases.append(phase_result("val", val))
    if val.returncode != 0:
        return {"strategy": strategy, "status": "failed", "checkpoint": str(checkpoint), "phases": phases}

    predict = run_child(
        args,
        strategy,
        "predict",
        ["--checkpoint", str(checkpoint), "--source", str(source), "--run-root", str(run_root)],
    )
    phases.append(phase_result("predict", predict))
    status = "ok" if predict.returncode == 0 else "failed"
    return {"strategy": strategy, "status": status, "checkpoint": str(checkpoint), "phases": phases}


def phase_result(name: str, result: subprocess.CompletedProcess) -> dict[str, Any]:
    return {
        "phase": name,
        "returncode": result.returncode,
        "stdout_tail": (result.stdout or "")[-2000:],
        "stderr_tail": (result.stderr or "")[-2000:],
    }


def run_parent(args: argparse.Namespace) -> int:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT))
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    data_yaml, source = make_dataset(args.scratch_dir / "dataset")
    started = time.perf_counter()
    results = [run_strategy(args, strategy, data_yaml, source) for strategy in args.strategy]
    report = {
        "status": "ok" if all(result["status"] == "ok" for result in results) else "failed",
        "model": args.model,
        "device": args.device,
        "imgsz": args.imgsz,
        "seconds": time.perf_counter() - started,
        "data": str(data_yaml),
        "source": str(source),
        "results": results,
    }
    report_path = args.scratch_dir / "e2e_smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved e2e report: {report_path}")
    print(json.dumps({"status": report["status"], "seconds": round(report["seconds"], 1)}, indent=2))
    return 0 if report["status"] == "ok" else 1


def main() -> None:
    args = parse_args()
    if args.worker == "train":
        worker_train(args)
        return
    if args.worker == "val":
        worker_val(args)
        return
    if args.worker == "predict":
        worker_predict(args)
        return
    raise SystemExit(run_parent(args))


if __name__ == "__main__":
    main()
