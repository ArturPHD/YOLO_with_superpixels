"""Check which superpixel-channel methods run on the tensor/CUDA path.

The check is intentionally runtime-based: each candidate receives a CUDA tensor
and must return Bx2xHxW edge channels on the same CUDA device. The report also
marks whether the implementation is strict GPU-only or merely GPU-compatible.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


METHODS: list[dict[str, Any]] = [
    {
        "name": "color_affinity",
        "family": "local color affinity",
        "kwargs": {"temperature": 0.1, "reduction": "l2"},
        "strict_gpu": True,
        "recommended": True,
    },
    {
        "name": "gradient_affinity",
        "family": "local gradient/texture affinity",
        "kwargs": {"temperature": 0.1, "reduction": "l2"},
        "strict_gpu": True,
        "recommended": True,
    },
    {
        "name": "color_gradient_affinity",
        "family": "local color + gradient affinity",
        "kwargs": {"temperature": 0.1, "reduction": "l2", "color_weight": 1.0, "gradient_weight": 1.0},
        "strict_gpu": True,
        "recommended": True,
    },
    {
        "name": "torch_slic",
        "family": "SLIC-like color/spatial clustering",
        "kwargs": {"n_segments": 300, "compactness": 1.0, "num_iters": 7},
        "strict_gpu": True,
        "recommended": True,
    },
    {
        "name": "torch_grid",
        "family": "regular grid baseline",
        "kwargs": {"n_segments": 300},
        "strict_gpu": True,
        "recommended": False,
    },
    {
        "name": "torch_voronoi",
        "family": "deterministic spatial Voronoi baseline",
        "kwargs": {"n_segments": 300, "jitter": 0.35, "seed": 0, "chunk_size": 65536},
        "strict_gpu": True,
        "recommended": False,
    },
    {
        "name": "spit_voronoi",
        "family": "SPiT/RViT PCA-tree pseudo-Voronoi",
        "kwargs": {"num_cells": 256},
        "strict_gpu": False,
        "recommended": False,
        "note": "Image tensor stays on CUDA, but SPiT initializes Sobol centroids on CPU.",
    },
    {
        "name": "spit",
        "family": "SPiT DenseSPEdgeEmbedder",
        "kwargs": {"device": "auto"},
        "strict_gpu": False,
        "recommended": False,
        "timeout_s": 20,
        "note": "Expected to be GPU/CuPy on a compatible stack; this Windows setup may hang.",
    },
    {
        "name": "spit_dense_adjacency",
        "family": "SPiT DenseSPEdgeEmbedder segmentation adjacency",
        "kwargs": {"device": "auto"},
        "strict_gpu": False,
        "recommended": True,
        "timeout_s": 20,
        "note": "Uses SPiT segment labels and converts them to binary right/down adjacency channels.",
    },
]


CPU_ONLY_METHODS = [
    "slic",
    "opencv_slic",
    "opencv_slico",
    "opencv_mslic",
    "opencv_lsc",
    "opencv_seeds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def run_one(method: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    timeout_s = int(method.get("timeout_s", args.timeout))
    code = f"""
import json
import time
import torch
from yolo_superpixels.integrations.ultralytics import _build_algorithm_from_name
from yolo_superpixels.superpixels.adjacency import AdjacencyChannelBuilder

name = {method["name"]!r}
kwargs = {method["kwargs"]!r}
device = torch.device({args.device!r})
if device.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
x = torch.rand(({args.batch}, 3, {args.height}, {args.width}), device=device)
if device.type == "cuda":
    torch.cuda.synchronize(device)
start = time.perf_counter()
algorithm = _build_algorithm_from_name(name, kwargs)
builder = AdjacencyChannelBuilder(algorithm)
edges = builder.edges_for_batch(x)
if device.type == "cuda":
    torch.cuda.synchronize(device)
elapsed = time.perf_counter() - start
print(json.dumps({{
    "ok": True,
    "shape": list(edges.shape),
    "device": str(edges.device),
    "dtype": str(edges.dtype),
    "min": float(edges.min().detach().cpu()),
    "max": float(edges.max().detach().cpu()),
    "elapsed_s": elapsed,
}}))
"""
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": f"timeout after {timeout_s}s",
            "stdout": exc.stdout,
            "stderr": exc.stderr,
        }

    if completed.returncode != 0:
        return {
            "ok": False,
            "error": f"exit code {completed.returncode}",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return {"ok": False, "error": "no JSON output", "stdout": completed.stdout, "stderr": completed.stderr}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "error": f"could not parse JSON: {exc}",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }


def main() -> None:
    args = parse_args()
    report = {
        "device": args.device,
        "input_shape": [args.batch, 3, args.height, args.width],
        "methods": [],
        "cpu_only_methods": CPU_ONLY_METHODS,
    }

    for method in METHODS:
        result = run_one(method, args)
        expected_shape = [args.batch, 2, args.height, args.width]
        same_device = result.get("device", "").startswith(args.device) or result.get("device") == args.device
        passed = bool(result.get("ok")) and result.get("shape") == expected_shape and same_device
        item = {
            **method,
            "passed_tensor_cuda_check": passed,
            "result": result,
        }
        report["methods"].append(item)
        status = "PASS" if passed else "FAIL"
        strict = "strict" if method["strict_gpu"] else "non-strict"
        elapsed = result.get("elapsed_s")
        elapsed_text = f"{elapsed:.3f}s" if isinstance(elapsed, (int, float)) else "-"
        print(f"{status:4} {method['name']:<24} {strict:<10} {elapsed_text}")
        if not passed:
            print(f"     {result.get('error', result)}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"saved report: {args.output}")


if __name__ == "__main__":
    main()
