import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from yolo_superpixels.strategies.structural_supervision.protocol import load_sp11_pilot, pilot_manifest

ROOT = Path(__file__).resolve().parents[1]


def write_protocol_fixture(tmp_path, expected_sha=None):
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"frozen-checkpoint")
    data = tmp_path / "data.yaml"
    data.write_text("path: dataset\ntrain: images/train\nval: images/val\n", encoding="utf-8")
    protocol = {
        "protocol": "sp11_pilot_v1",
        "shared": {
            "initialization": {
                "base_model_path": str(checkpoint),
                "base_model_sha256": expected_sha,
                "auxiliary_init_seed": 0,
            },
            "training": {"data": str(data), "imgsz": 64, "epochs": 1, "batch": 1, "optimizer": "SGD"},
            "structural_supervision": {"lambda_fg": 0.05, "region_balance_policy": "balanced_fg_bg"},
        },
    }
    protocol_path = tmp_path / "pilot_v1.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    arms = {
        "A": {"arm": "A", "strategy": "rgb_baseline"},
        "B": {"arm": "B", "strategy": "structural_supervision", "structural_supervision": {"lambda_sp": 0.0}},
        "C": {
            "arm": "C", "strategy": "structural_supervision",
            "structural_supervision": {"lambda_sp": 0.05, "region_algorithm": "torch_slic", "region_kwargs": {"n_segments": 8}},
        },
        "D": {
            "arm": "D", "strategy": "structural_supervision",
            "structural_supervision": {"lambda_sp": 0.05, "region_algorithm": "torch_grid", "region_kwargs": {"n_segments": 8}},
        },
    }
    names = {"A": "A_rgb_baseline.yaml", "B": "B_foreground_aux.yaml", "C": "C_foreground_aux_slic.yaml", "D": "D_foreground_aux_grid.yaml"}
    for arm, payload in arms.items():
        (tmp_path / names[arm]).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return protocol_path, checkpoint, data


def test_protocol_resolves_matched_arms_and_records_identities(tmp_path):
    protocol, checkpoint, data = write_protocol_fixture(tmp_path)
    arms = load_sp11_pilot(protocol, ["A", "B", "C", "D"])
    manifest = pilot_manifest(arms)
    assert len({arm.shared_protocol_sha256 for arm in arms}) == 1
    assert manifest["base_checkpoint"]["path"] == str(checkpoint.resolve())
    assert manifest["data_yaml"]["path"] == str(data.resolve())
    assert "structural_supervision" not in arms[0].resolved
    assert arms[1].resolved["structural_supervision"]["lambda_sp"] == 0.0
    assert arms[2].resolved["structural_supervision"]["region_algorithm"] == "torch_slic"
    assert arms[3].resolved["structural_supervision"]["region_algorithm"] == "torch_grid"


def test_protocol_fails_on_expected_checkpoint_hash_mismatch(tmp_path):
    protocol, _, _ = write_protocol_fixture(tmp_path, expected_sha="0" * 64)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_sp11_pilot(protocol, ["A"])


def test_protocol_rejects_shared_training_override_in_arm(tmp_path):
    protocol, _, _ = write_protocol_fixture(tmp_path)
    arm_path = tmp_path / "B_foreground_aux.yaml"
    payload = yaml.safe_load(arm_path.read_text(encoding="utf-8"))
    payload["training"] = {"batch": 999}
    arm_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="shared fields are forbidden"):
        load_sp11_pilot(protocol, ["B"])


def test_pilot_launcher_dry_run_writes_resolved_manifest(tmp_path):
    protocol, checkpoint, data = write_protocol_fixture(tmp_path)
    manifest = tmp_path / "manifest.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_sp11_pilot.py"), "--protocol", str(protocol), "--arm", "A", "B", "C", "D", "--manifest", str(manifest), "--dry-run"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["base_checkpoint"]["path"] == str(checkpoint.resolve())
    assert payload["data_yaml"]["path"] == str(data.resolve())
    assert len(payload["arms"]) == 4
