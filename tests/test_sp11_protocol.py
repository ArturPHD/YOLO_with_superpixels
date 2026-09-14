import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from yolo_superpixels.strategies.structural_supervision.protocol import build_dataset_identity, load_sp11_pilot, pilot_manifest

ROOT = Path(__file__).resolve().parents[1]


def write_protocol_fixture(tmp_path, expected_sha=None):
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"frozen-checkpoint")
    dataset = tmp_path / "dataset"
    for split in ("train", "val"):
        image = dataset / "images" / split / f"{split}.jpg"
        label = dataset / "labels" / split / f"{split}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")
        label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    data = tmp_path / "data.yaml"
    data.write_text(f"path: {dataset}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['object']\n", encoding="utf-8")
    protocol = {
        "protocol": "sp11_pilot_v1",
        "shared": {
            "initialization": {
                "mode": "technical_pretrained_smoke",
                "base_model_path": str(checkpoint),
                "base_model_sha256": expected_sha,
                "auxiliary_init_seed": 0,
            },
            "training": {"data": str(data), "imgsz": 64, "epochs": 1, "batch": 1, "optimizer": "SGD", "patience": 0},
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
    assert "PAIRED_PROTOCOL_CHECK: PASS" in result.stdout


def test_dataset_identity_changes_when_labels_or_train_list_change(tmp_path):
    protocol, _, data = write_protocol_fixture(tmp_path)
    first = build_dataset_identity(data)
    label = tmp_path / "dataset" / "labels" / "train" / "train.txt"
    label.write_text("0 0.1 0.1 0.1 0.1\n", encoding="utf-8")
    second = build_dataset_identity(data)
    assert first["sha256"] != second["sha256"]
    train_list = tmp_path / "dataset" / "train.txt"
    train_list.write_text(str(tmp_path / "dataset" / "images" / "train" / "train.jpg") + "\n", encoding="utf-8")
    data.write_text(f"path: {tmp_path / 'dataset'}\ntrain: train.txt\nval: images/val\n", encoding="utf-8")
    third = build_dataset_identity(data)
    train_list.write_text("\n", encoding="utf-8")
    fourth = build_dataset_identity(data)
    assert third["sha256"] != fourth["sha256"]


def test_scientific_paired_scratch_checkpoint_is_created_once_and_reused(tmp_path):
    pytest.importorskip("ultralytics")
    protocol, _, data = write_protocol_fixture(tmp_path)
    payload = yaml.safe_load(protocol.read_text(encoding="utf-8"))
    source_yaml = ROOT / "configs" / "sp11" / "yolov12n.yaml"
    checkpoint = tmp_path / "paired_scratch.pt"
    payload["shared"]["initialization"] = {
        "mode": "scientific_paired_scratch",
        "source_model_yaml": str(source_yaml),
        "base_model_path": str(checkpoint),
        "base_initialization_seed": 0,
        "auxiliary_init_seed": 0,
        "create_if_missing": True,
    }
    payload["shared"]["scientific_subset"] = {
        "provenance": "pytest fixture",
        "base_train_image_count": 10,
        "expected_fraction": 0.1,
        "fraction_tolerance": 0.0,
    }
    protocol.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    arms = load_sp11_pilot(protocol, ["A", "B", "C", "D"])
    assert checkpoint.is_file() and checkpoint.with_suffix(".json").is_file()
    assert len({arm.base_checkpoint["sha256"] for arm in arms}) == 1
    assert all(arm.base_checkpoint["base_initialization_seed"] == 0 for arm in arms)


def test_ultralytics_patience_zero_disables_early_stopping_for_fixed_horizon():
    pytest.importorskip("ultralytics")
    from ultralytics.utils.torch_utils import EarlyStopping

    stopper = EarlyStopping(patience=0)
    assert stopper.patience == float("inf")
    assert stopper(10, 0.0) is False
