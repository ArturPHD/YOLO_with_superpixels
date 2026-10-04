"""Audit whether trained neck-injection checkpoints use token information.

Each case reloads the same checkpoint and changes only the validation-time token
values presented to its already-trained fusion module. It supports full SPiT and
Q* checkpoints. This is an OOD dependency diagnostic, not evidence that any
perturbation is a new useful representation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from yolo_superpixels import NeckInjectionConfig, SuperpixelNeckYOLO  # noqa: E402
from yolo_superpixels.strategies.neck_injection import SuperpixelNeckFusion  # noqa: E402


TOKEN_CASES = {
    "normal": (True, "normal"),
    "injection_disabled": (False, "normal"),
    "zero": (True, "zero"),
    "constant": (True, "constant"),
    "batch_shuffle": (True, "batch_shuffle"),
    "spatial_shuffle": (True, "spatial_shuffle"),
    "random_matched": (True, "random_matched"),
}
METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)


@dataclass(frozen=True)
class CheckpointSpec:
    run_dir: Path
    checkpoint: Path
    weight_kind: str
    config: dict[str, Any]
    payload: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--token-provider",
        choices=["spit", "qstar"],
        default="spit",
        help="Checkpoint provider to discover. The default preserves the historical SPiT workflow.",
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", choices=["best", "last", "both"], default="both")
    parser.add_argument("--cases", nargs="+", choices=sorted(TOKEN_CASES), default=list(TOKEN_CASES))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spit-weights", help="Optional local override if a checkpoint lacks its embedded provider.")
    parser.add_argument("--annotations", type=Path, help="COCO instances JSON; enables AP_S/M/L and per-class AP.")
    parser.add_argument("--save-json", action="store_true", help="Save COCO predictions even without --annotations.")
    parser.add_argument("--keep-predictions", action="store_true", help="Retain prediction JSON after COCOeval.")
    parser.add_argument("--diagnostic-cases", nargs="+", choices=sorted(TOKEN_CASES), default=["normal"])
    parser.add_argument("--diagnostic-forwards", type=int, default=8)
    parser.add_argument("--diagnostic-queries", type=int, default=256)
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "neck_dependency_ablation")
    parser.add_argument("--name", default="dependency")
    parser.add_argument("--limit", type=int, help="Limit checkpoints for a technical smoke.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"Invalid superpixel config: {path}")
    return payload


def discover_checkpoints(
    roots: list[Path], weight_choice: str, token_provider: str
) -> list[CheckpointSpec]:
    kinds = ("best", "last") if weight_choice == "both" else (weight_choice,)
    specs: list[CheckpointSpec] = []
    seen: set[tuple[Path, str]] = set()
    for root in roots:
        for config_path in sorted(root.rglob("superpixel_config.json")):
            payload = _load_payload(config_path)
            config = payload["config"]
            if payload.get("strategy") != "neck_injection" or config.get("token_provider") != token_provider:
                continue
            run_dir = config_path.parent
            for kind in kinds:
                checkpoint = run_dir / "weights" / f"{kind}.pt"
                key = (checkpoint.resolve(), kind)
                if checkpoint.exists() and key not in seen:
                    specs.append(CheckpointSpec(run_dir, checkpoint, kind, config, payload))
                    seen.add(key)
    return sorted(specs, key=lambda item: (item.run_dir.name, item.weight_kind))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _job_id(payload: dict[str, Any]) -> str | None:
    project = str(payload.get("training_args", {}).get("project", ""))
    match = re.search(r"/(\d+_\d+)_", project)
    return match.group(1) if match else None


def _metric_value(metrics: Any, key: str) -> float | None:
    results = getattr(metrics, "results_dict", {}) or {}
    if key in results:
        return float(results[key])
    box = getattr(metrics, "box", None)
    fallbacks = {
        "metrics/precision(B)": "mp",
        "metrics/recall(B)": "mr",
        "metrics/mAP50(B)": "map50",
        "metrics/mAP50-95(B)": "map",
    }
    value = getattr(box, fallbacks[key], None) if box is not None else None
    return float(value) if value is not None else None


def _safe_name(text: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in text)


def _build_config(spec: CheckpointSpec, args: argparse.Namespace, enabled: bool, ablation: str, diagnostics: bool) -> NeckInjectionConfig:
    fields = NeckInjectionConfig.__dataclass_fields__
    values = {key: value for key, value in spec.config.items() if key in fields}
    config = NeckInjectionConfig.from_jsonable(values)
    config.enabled = enabled
    config.token_ablation = ablation
    config.token_ablation_seed = args.seed
    config.feature_diagnostics = diagnostics and enabled
    config.feature_diagnostics_max_forwards = args.diagnostic_forwards if diagnostics and enabled else 0
    config.feature_diagnostics_max_queries = args.diagnostic_queries
    if args.spit_weights and args.token_provider == "spit":
        config.spit_weights = str(args.spit_weights)
    return config


def _find_fusion(model: SuperpixelNeckYOLO) -> SuperpixelNeckFusion:
    matches = [module for module in model.yolo.model.modules() if isinstance(module, SuperpixelNeckFusion)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one neck fusion module, found {len(matches)}.")
    return matches[0]


def _coco_metrics(predictions: Path, annotations: Path) -> tuple[dict[str, float], dict[str, float], str]:
    try:
        import numpy as np
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("AP_S/M/L and per-class AP require pycocotools.") from exc

    coco_gt = COCO(str(annotations))
    prediction_rows = json.loads(predictions.read_text(encoding="utf-8"))
    category_ids = {int(row["category_id"]) for row in prediction_rows}
    coco_category_ids = set(coco_gt.getCatIds())
    mapping_name = "native_category_ids"
    evaluation_predictions = predictions
    if category_ids and not category_ids.issubset(coco_category_ids):
        # Ultralytics writes class_id + 1 when the data YAML is not literally
        # named coco.yaml. Convert its contiguous 1..80 IDs to COCO category IDs
        # before direct COCOeval, while retaining the original predictions file.
        from ultralytics.data.converter import coco80_to_coco91_class

        mapping = coco80_to_coco91_class()
        remapped_rows = []
        for row in prediction_rows:
            class_index = int(row["category_id"]) - 1
            if class_index < 0 or class_index >= len(mapping):
                raise ValueError(f"Unexpected non-COCO prediction category ID: {row['category_id']}")
            remapped_rows.append({**row, "category_id": mapping[class_index]})
        evaluation_predictions = predictions.with_name(f"{predictions.stem}_coco91.json")
        evaluation_predictions.write_text(json.dumps(remapped_rows), encoding="utf-8")
        mapping_name = "ultralytics_coco80_to_coco91"

    coco_dt = coco_gt.loadRes(str(evaluation_predictions))
    evaluation = COCOeval(coco_gt, coco_dt, "bbox")
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    labels = {
        "coco_AP": float(evaluation.stats[0]),
        "coco_AP50": float(evaluation.stats[1]),
        "coco_AP_S": float(evaluation.stats[3]),
        "coco_AP_M": float(evaluation.stats[4]),
        "coco_AP_L": float(evaluation.stats[5]),
    }
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    precision = evaluation.eval["precision"]
    per_class = {}
    for index, category in enumerate(categories):
        values = precision[:, :, index, 0, -1]
        values = values[values > -1]
        per_class[str(category["name"])] = float(np.mean(values)) if values.size else float("nan")
    return labels, per_class, mapping_name


def run_case(spec: CheckpointSpec, case: str, args: argparse.Namespace) -> dict[str, Any]:
    enabled, ablation = TOKEN_CASES[case]
    diagnostics = case in args.diagnostic_cases
    row: dict[str, Any] = {
        "checkpoint": str(spec.checkpoint),
        "checkpoint_sha256": _sha256(spec.checkpoint),
        "run_dir": str(spec.run_dir),
        "weight_kind": spec.weight_kind,
        "case": case,
        "token_provider": args.token_provider,
        "injection_enabled": enabled,
        "token_ablation": ablation,
        "job_id": _job_id(spec.payload),
        "training_args": json.dumps(spec.payload.get("training_args", {}), sort_keys=True),
        "source_tree_sha256": spec.payload.get("source_tree", {}).get("sha256"),
        "spit_source_sha256": spec.payload.get("spit", {}).get("source_tree", {}).get("sha256"),
    }
    run_name = _safe_name(f"{args.name}_{spec.run_dir.name}_{spec.weight_kind}_{case}")
    row["val_run"] = str(args.project / run_name)
    if args.dry_run:
        row["status"] = "dry_run"
        return row

    config = _build_config(spec, args, enabled, ablation, diagnostics)
    if case == "batch_shuffle" and config.fusion != "cross_attention":
        row["status"] = "not_applicable"
        row["reason"] = "batch_shuffle is defined only for cross_attention fusion"
        return row
    start = time.perf_counter()
    model = SuperpixelNeckYOLO(str(spec.checkpoint), config=config)
    model.set_injection_enabled(enabled)
    model.set_token_ablation(ablation, args.seed)
    model.set_feature_diagnostics(diagnostics and enabled, args.diagnostic_queries, args.diagnostic_forwards)
    val_kwargs: dict[str, Any] = {
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(args.project),
        "name": run_name,
        "plots": False,
        "save_json": bool(args.save_json or args.annotations),
        "exist_ok": True,
    }
    if args.device is not None:
        val_kwargs["device"] = args.device
    metrics = model.val(**val_kwargs)
    row["validation_seconds"] = round(time.perf_counter() - start, 3)
    for key in METRIC_KEYS:
        row[key] = _metric_value(metrics, key)

    if diagnostics and enabled:
        row.update(_find_fusion(model).diagnostic_summary())
    if args.annotations:
        predictions = args.project / run_name / "predictions.json"
        coco_metrics, per_class, mapping_name = _coco_metrics(predictions, args.annotations)
        row.update(coco_metrics)
        row["coco_category_mapping"] = mapping_name
        (args.project / run_name / "per_class_ap.json").write_text(
            json.dumps(per_class, indent=2, sort_keys=True), encoding="utf-8"
        )
        if not args.keep_predictions:
            predictions.unlink(missing_ok=True)
            predictions.with_name(f"{predictions.stem}_coco91.json").unlink(missing_ok=True)
    row["status"] = "ok"
    return row


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    specs = discover_checkpoints(args.experiment_root, args.weights, args.token_provider)
    if args.limit is not None:
        specs = specs[: args.limit]
    if not specs:
        raise SystemExit(f"No {args.token_provider} neck-injection checkpoints found.")

    rows: list[dict[str, Any]] = []
    for spec in specs:
        for case in args.cases:
            try:
                rows.append(run_case(spec, case, args))
            except Exception as exc:  # Preserve other checkpoints when one archival artifact is incompatible.
                rows.append(
                    {
                        "checkpoint": str(spec.checkpoint),
                        "run_dir": str(spec.run_dir),
                        "weight_kind": spec.weight_kind,
                        "case": case,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    write_summary(rows, args.project)
    failures = [row for row in rows if row.get("status") == "error"]
    if failures:
        raise SystemExit(f"Dependency ablation completed with {len(failures)} failed case(s).")


if __name__ == "__main__":
    main()
