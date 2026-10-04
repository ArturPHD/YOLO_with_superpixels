"""Regression tests for generic neck-token dependency-ablation discovery."""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

from scripts import validate_spit_dependency_ablation as ablation


def _write_checkpoint_sidecar(run_dir: Path, token_provider: str, fusion: str = "add") -> Path:
    (run_dir / "weights").mkdir(parents=True)
    checkpoint = run_dir / "weights" / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    payload = {
        "strategy": "neck_injection",
        "config": {"token_provider": token_provider, "fusion": fusion},
        "training_args": {},
    }
    (run_dir / "superpixel_config.json").write_text(json.dumps(payload), encoding="utf-8")
    return checkpoint


def test_dependency_ablation_discovers_only_requested_token_provider(tmp_path: Path):
    qstar_run = tmp_path / "qstar"
    spit_run = tmp_path / "spit"
    qstar_checkpoint = _write_checkpoint_sidecar(qstar_run, "qstar")
    _write_checkpoint_sidecar(spit_run, "spit")

    specs = ablation.discover_checkpoints([tmp_path], "best", "qstar")

    assert len(specs) == 1
    assert specs[0].checkpoint == qstar_checkpoint
    assert specs[0].config["token_provider"] == "qstar"


def test_batch_shuffle_is_not_applicable_to_dense_fusion(tmp_path: Path):
    run_dir = tmp_path / "qstar"
    checkpoint = _write_checkpoint_sidecar(run_dir, "qstar", fusion="add")
    spec = ablation.CheckpointSpec(
        run_dir=run_dir,
        checkpoint=checkpoint,
        weight_kind="best",
        config={"token_provider": "qstar", "fusion": "add"},
        payload={"training_args": {}},
    )
    args = Namespace(
        token_provider="qstar",
        spit_weights=None,
        seed=0,
        diagnostic_forwards=1,
        diagnostic_queries=16,
        diagnostic_cases=["normal"],
        project=tmp_path / "output",
        name="test",
        dry_run=False,
    )

    row = ablation.run_case(spec, "batch_shuffle", args)

    assert row["status"] == "not_applicable"
    assert "cross_attention" in row["reason"]
