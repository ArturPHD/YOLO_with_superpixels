# third_party

Use this directory only for external repositories that are not convenient to
install from PyPI.

Recommended default:

- install `ultralytics` from PyPI;
- install `scikit-image` from PyPI for SLIC;
- clone SPiT here and link it through a `.pth` file. Its `setup.py` is not ideal
  for this project: the wheel can miss `spit.tokenizer`, and the metadata asks
  for source `cupy>=13.0.0` even when `cupy-cuda12x` is installed.

Preferred SPiT install on Windows:

```bash
pip install cupy-cuda12x
git clone https://github.com/dsb-ifi/SPiT.git third_party/SPiT
python -c "import site, pathlib; p=pathlib.Path(next(x for x in site.getsitepackages() if x.endswith('site-packages'))) / 'yolo_superpixels_spit.pth'; p.write_text(str(pathlib.Path('third_party/SPiT').resolve()) + '\n')"
```

Example layout for a local SPiT checkout:

```text
third_party/
  SPiT/
    spit/
      tokenizer/
        densesp.py
```

Then pass the import path to the adapter:

```python
SuperpixelYOLO(
    "yolo11n.pt",
    algorithm="spit",
    algorithm_kwargs={
        "module_path": "spit.tokenizer.densesp",
        "class_name": "DenseSPEdgeEmbedder",
    },
)
```

If the external repository is important for reproducibility, prefer adding it as
a git submodule instead of copying files by hand.
