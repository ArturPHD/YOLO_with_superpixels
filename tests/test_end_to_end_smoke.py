import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end_smoke_opt_in(tmp_path):
    if os.environ.get("YOLO_SUPERPIXELS_RUN_E2E") != "1":
        pytest.skip("Set YOLO_SUPERPIXELS_RUN_E2E=1 to run the slow lifecycle smoke test.")
    weights = ROOT / os.environ.get("YOLO_SUPERPIXELS_E2E_MODEL", "yolo12n.pt")
    if not weights.exists():
        pytest.skip(f"{weights.name} is not available locally.")
    device = os.environ.get("YOLO_SUPERPIXELS_E2E_DEVICE", "0")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_e2e_smoke.py"),
        "--strategy",
        os.environ.get("YOLO_SUPERPIXELS_E2E_STRATEGY", "input"),
        "--model",
        str(weights),
        "--device",
        device,
        "--scratch-dir",
        str(tmp_path),
        "--timeout",
        os.environ.get("YOLO_SUPERPIXELS_E2E_TIMEOUT", "900"),
    ]
    if os.environ.get("YOLO_SUPERPIXELS_E2E_REQUIRE_CUDA") == "1":
        command.append("--require-cuda")

    result = subprocess.run(
        command,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=int(os.environ.get("YOLO_SUPERPIXELS_E2E_TIMEOUT", "900")),
        check=False,
    )

    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    report = json.loads((tmp_path / "e2e_smoke_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "ok"
