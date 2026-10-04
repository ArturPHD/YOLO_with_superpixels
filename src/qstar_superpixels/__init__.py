"""Import shim for the MIT Q* source vendored with yolo-superpixels.

The Q* pipeline is intentionally retained under ``third_party`` with its own
licence and provenance. This package exposes that source on the installed
Python path so PyTorch can unpickle a YOLO checkpoint containing a Q* provider
in a fresh process.
"""

from __future__ import annotations

from pathlib import Path


for _parent in Path(__file__).resolve().parents:
    _vendor = _parent / "third_party" / "qstar_superpixels" / "qstar_superpixels"
    if (_vendor / "pipeline.py").is_file():
        __path__ = [str(_vendor)]
        break
else:  # pragma: no cover - installation error, not ordinary package behavior
    raise ImportError("Vendored Q* source is missing from third_party/qstar_superpixels.")

from .pipeline import PipelineConfig, SuperpixelPipeline, SuperpixelResult  # noqa: E402

__all__ = ["PipelineConfig", "SuperpixelPipeline", "SuperpixelResult"]
