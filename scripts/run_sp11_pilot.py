"""Run or inspect the versioned matched SP-11 A/B/C/D pilot protocol.

The launcher deliberately accepts no training hyperparameter overrides. Edit the
single shared protocol, review its dry-run manifest, then freeze it for a pilot.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs" / "sp11" / "pilot_v1.yaml")
    parser.add_argument("--arm", choices=("A", "B", "C", "D"), nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Resolve and write the frozen manifest without training.")
    parser.add_argument(
        "--loss-scale-diagnostic-batches",
        "--diagnostic-batches",
        dest="diagnostic_batches",
        type=int,
        default=0,
        help="Write component loss/gradient records for initial batches; does not tune lambda.",
    )
    parser.add_argument(
        "--runtime-profile",
        action="store_true",
        help="Synchronize C/D structural preprocessing for explicit timing; use only for a short smoke.",
    )
    parser.add_argument(
        "--diagnostic-gradient-norms",
        action="store_true",
        help="Also append P3 foreground-logit gradient norms during diagnostic batches.",
    )
    return parser.parse_args()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _runtime_metrics(model: Any, elapsed_seconds: float, epochs: int, structural: bool) -> dict[str, Any]:
    import torch

    trainer = getattr(getattr(model, "yolo", model), "trainer", None)
    seen = getattr(trainer, "seen", None)
    device = getattr(trainer, "device", None)
    metrics: dict[str, Any] = {
        "wall_seconds": elapsed_seconds,
        "wall_seconds_per_epoch": elapsed_seconds / max(int(epochs), 1),
        "images_seen": int(seen) if isinstance(seen, int) else None,
        "images_per_second": float(seen) / elapsed_seconds if isinstance(seen, int) and elapsed_seconds else None,
        "peak_vram_bytes": None,
        "structural_preprocess_seconds": None,
        "structural_preprocess_calls": None,
        "structural_consistency_seconds": None,
        "structural_consistency_calls": None,
    }
    if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
        metrics["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated(device))
    if structural:
        inner_model = getattr(getattr(model, "yolo", model), "model", None)
        context = getattr(inner_model, "_yolo_superpixels_structural_context", None)
        if context is not None:
            metrics["structural_preprocess_seconds"] = context.structural_preprocess_seconds
            metrics["structural_preprocess_calls"] = context.structural_preprocess_calls
            metrics["structural_consistency_seconds"] = context.structural_consistency_seconds
            metrics["structural_consistency_calls"] = context.structural_consistency_calls
    return metrics


def _run_arm(
    resolved_arm,
    manifest_path: Path,
    diagnostic_batches: int,
    runtime_profile: bool,
    diagnostic_gradient_norms: bool,
) -> dict[str, Any]:
    import torch
    from yolo_superpixels import StructuralSupervisionConfig, SuperpixelStructuralYOLO

    resolved = resolved_arm.resolved
    training = dict(resolved["training"])
    training["name"] = f"sp11_{resolved_arm.arm}_{resolved_arm.resolved_sha256[:12]}"
    training["exist_ok"] = False
    training["plots"] = False
    project = Path(training["project"]).resolve()
    diagnostic_path = project / training["name"] / "loss_scale_diagnostics.jsonl"
    is_structural = resolved["strategy"] == "structural_supervision"
    if is_structural:
        structural_values = dict(resolved["structural_supervision"])
        structural_values.update(
            {
                "base_model_path": resolved_arm.base_checkpoint["path"],
                "base_model_sha256": resolved_arm.base_checkpoint["sha256"],
                "auxiliary_init_seed": resolved["initialization"]["auxiliary_init_seed"],
                "diagnostic_batches": diagnostic_batches,
                "diagnostic_output_path": str(diagnostic_path) if diagnostic_batches else None,
                "diagnostic_gradient_norms": diagnostic_gradient_norms or bool(diagnostic_batches),
                "runtime_profile": runtime_profile or structural_values.get("runtime_profile", False),
            }
        )
        model = SuperpixelStructuralYOLO(
            resolved_arm.base_checkpoint["path"], config=StructuralSupervisionConfig(**structural_values)
        )
    else:
        from ultralytics import YOLO

        model = YOLO(resolved_arm.base_checkpoint["path"])

    requested_device = str(training.get("device", ""))
    use_cuda = torch.cuda.is_available() and requested_device.lower() not in {"cpu", "mps"}
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train(**training)
    runtime = _runtime_metrics(model, time.perf_counter() - started, training["epochs"], is_structural)
    save_dir = Path(getattr(getattr(model, "yolo", model), "trainer").save_dir)
    arm_record = {
        "manifest_path": str(manifest_path),
        "manifest_protocol_sha256": resolved_arm.protocol_sha256,
        "arm": resolved_arm.manifest_entry(),
        "runtime": runtime,
        "diagnostic_path": str(diagnostic_path) if diagnostic_batches else None,
    }
    if diagnostic_batches:
        summary = _summarize_diagnostic(diagnostic_path)
        _write_json(save_dir / "loss_scale_diagnostic_summary.json", summary)
        arm_record["diagnostic_summary_path"] = str(save_dir / "loss_scale_diagnostic_summary.json")
    _write_json(save_dir / "sp11_pilot_arm.json", arm_record)
    return arm_record


def _summarize_diagnostic(path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = (
        "yolo_loss_raw", "fg_aux_loss_raw", "fg_aux_loss_weighted", "sp_consistency_loss_raw",
        "sp_consistency_loss_weighted", "aux_total_weighted", "foreground_logit_gradient_norm_fg",
        "foreground_logit_gradient_norm_sp", "foreground_logit_gradient_ratio_sp_over_fg",
    )
    return {
        "kind": "sp11_loss_scale_diagnostic", "batches": len(records),
        "medians": {key: statistics.median(values) if (values := [row[key] for row in records if row.get(key) is not None]) else None for key in keys},
        "note": "Diagnostic-only; freeze lambda_fg and lambda_sp by human review before the scientific pilot.",
    }


def _print_comparison(resolved_arms) -> None:
    first = resolved_arms[0]
    training = first.resolved["training"]
    print("SP-11 paired protocol")
    print("arm strategy                 lambda_fg lambda_sp region      checkpoint    dataset       protocol     arm")
    for arm in resolved_arms:
        structural = arm.resolved.get("structural_supervision", {})
        print(
            f"{arm.arm:<3} {arm.resolved['strategy']:<24} {structural.get('lambda_fg', '-!s'):<9} "
            f"{structural.get('lambda_sp', '-!s'):<9} {structural.get('region_algorithm', '-'):<11} "
            f"{arm.base_checkpoint['sha256'][:12]} {arm.dataset_identity['sha256'][:12]} "
            f"{arm.shared_protocol_sha256[:12]} {arm.arm_sha256[:12]}"
        )
    print(f"common: epochs={training['epochs']} patience={training['patience']} batch={training.get('batch')} seed={training.get('seed')}")


def main() -> None:
    from yolo_superpixels.strategies.structural_supervision.protocol import (
        load_sp11_pilot,
        pilot_manifest,
        torch_slic_repeatability_metadata,
        write_dataset_identity_records,
    )

    args = parse_args()
    if args.diagnostic_batches < 0:
        raise ValueError("--diagnostic-batches must be non-negative.")
    resolved_arms = load_sp11_pilot(args.protocol, args.arm)
    manifest = pilot_manifest(resolved_arms)
    manifest["selected_arms"] = list(args.arm)
    manifest["dry_run"] = bool(args.dry_run)
    try:
        manifest["torch_slic_cuda_repeatability"] = torch_slic_repeatability_metadata()
    except Exception as error:
        manifest["torch_slic_cuda_repeatability"] = {"error": str(error)}
    manifest_path = args.manifest.resolve() if args.manifest else (ROOT / "runs" / "sp11" / "sp11_pilot_manifest.json")
    manifest["dataset_manifest_files"] = write_dataset_identity_records(
        resolved_arms[0].dataset_identity, manifest_path.parent / "dataset_identity"
    )
    _write_json(manifest_path, manifest)
    _print_comparison(resolved_arms)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"SP-11 manifest written to {manifest_path}")
    if args.dry_run:
        print("PAIRED_PROTOCOL_CHECK: PASS")
        return

    results = []
    for arm in resolved_arms:
        results.append(
            _run_arm(
                arm,
                manifest_path,
                args.diagnostic_batches,
                args.runtime_profile,
                args.diagnostic_gradient_norms,
            )
        )
    manifest["runs"] = results
    _write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
