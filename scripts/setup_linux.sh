#!/usr/bin/env bash
set -euo pipefail

PYTHON_CMD="${PYTHON:-}"
TORCH="auto"
TORCH_VERSION="2.13.0"
FLASH_ATTN_WHEEL=""
SPIT_REVISION="9ad9cb81130ddbefac894094e37699807811f8ea"
RESOLVED_TORCH=""
NO_CONSTRAINTS=0
SKIP_TESTS=0
RUN_SMOKE=0
RUN_NECK_SMOKE=0
RUN_AUX_SMOKE=0
SMOKE_ALGORITHM="torch_slic"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_CMD="$2"
      shift 2
      ;;
    --torch)
      TORCH="$2"
      shift 2
      ;;
    --torch-version)
      TORCH_VERSION="$2"
      shift 2
      ;;
    --flash-attn-wheel)
      FLASH_ATTN_WHEEL="$2"
      shift 2
      ;;
    --spit-revision)
      SPIT_REVISION="$2"
      shift 2
      ;;
    --no-constraints)
      NO_CONSTRAINTS=1
      shift
      ;;
    --skip-tests)
      SKIP_TESTS=1
      shift
      ;;
    --run-smoke)
      RUN_SMOKE=1
      shift
      ;;
    --run-neck-smoke)
      RUN_NECK_SMOKE=1
      shift
      ;;
    --run-aux-smoke)
      RUN_AUX_SMOKE=1
      shift
      ;;
    --smoke-algorithm)
      SMOKE_ALGORITHM="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  bash scripts/setup_linux.sh [--python python3.10] [--torch cpu|cu118|cu121|cu124|cu126|auto] [--torch-version VERSION] [--flash-attn-wheel PATH_OR_URL] [--spit-revision COMMIT] [--no-constraints] [--skip-tests] [--run-smoke] [--run-neck-smoke] [--run-aux-smoke] [--smoke-algorithm slic|torch_slic|gpu_slic|spit|spit_dense_adjacency|spit_adjacency|spit_voronoi|pseudo_voronoi|torch_grid|gpu_grid|torch_voronoi|gpu_voronoi|color_affinity|gpu_affinity|gradient_affinity|texture_affinity|color_gradient_affinity|opencv_slic|opencv_slico|opencv_mslic|opencv_lsc|opencv_seeds]

Examples:
  bash scripts/setup_linux.sh --torch cpu
  bash scripts/setup_linux.sh --torch cu121 --run-smoke --smoke-algorithm torch_slic
  bash scripts/setup_linux.sh --python python3.11 --torch cu118 --torch-version 2.2.2 --flash-attn-wheel ./flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT/.venv"
VENV_PY="$VENV_DIR/bin/python"
THIRD_PARTY_DIR="$ROOT/third_party"
SPIT_DIR="$THIRD_PARTY_DIR/SPiT"
CONSTRAINTS_FILE="$ROOT/constraints-verified-windows-py310-cu126.txt"

step() {
  printf '\n==> %s\n' "$1"
}

resolve_python() {
  if [[ -n "$PYTHON_CMD" ]]; then
    command -v "$PYTHON_CMD" >/dev/null 2>&1 || {
      echo "Python command not found: $PYTHON_CMD" >&2
      exit 1
    }
    echo "$PYTHON_CMD"
    return
  fi

  for candidate in python3.10 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
        echo "$candidate"
        return
      fi
    fi
  done

  echo "Python 3.10+ was not found. Install Python 3.10/3.11 first." >&2
  exit 1
}

install_torch() {
  if [[ "$TORCH" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      resolved_torch="cu126"
    else
      resolved_torch="cpu"
    fi
  else
    resolved_torch="$TORCH"
  fi
  RESOLVED_TORCH="$resolved_torch"

  local index
  if [[ "$resolved_torch" == "cpu" ]]; then
    index="https://download.pytorch.org/whl/cpu"
  elif [[ "$resolved_torch" =~ ^cu(118|121|124|126)$ ]]; then
    index="https://download.pytorch.org/whl/$resolved_torch"
  else
    echo "Invalid --torch value: $resolved_torch" >&2
    exit 1
  fi

  current_torch="$("$VENV_PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
  if [[ -n "$current_torch" ]]; then
    if [[ "$resolved_torch" == "cpu" ]]; then
      wanted_tag="+cpu"
    else
      wanted_tag="+$resolved_torch"
    fi
    if [[ "$current_torch" == *"$wanted_tag"* ]] && { [[ -z "$TORCH_VERSION" ]] || [[ "$current_torch" == "$TORCH_VERSION"* ]]; }; then
      step "Using existing torch $current_torch"
      return
    fi
  fi

  step "Installing torch/torchvision from $index"
  if [[ -n "$TORCH_VERSION" ]]; then
    "$VENV_PY" -m pip install --upgrade --force-reinstall "torch==$TORCH_VERSION" torchvision --index-url "$index"
  else
    "$VENV_PY" -m pip install --upgrade --force-reinstall torch torchvision --index-url "$index"
  fi
}

link_spit_checkout() {
  step "Linking local SPiT checkout"
  "$VENV_PY" -m pip uninstall -y spit >/dev/null 2>&1 || true
  rm -rf "$SPIT_DIR/spit.egg-info"
  site_packages="$("$VENV_PY" - <<'PY'
import site
print(next(p for p in site.getsitepackages() if p.endswith("site-packages")))
PY
)"
  printf '%s\n' "$SPIT_DIR" > "$site_packages/yolo_superpixels_spit.pth"
  echo "SPiT checkout linked via $site_packages/yolo_superpixels_spit.pth"
}

install_flash_attn() {
  if [[ -z "$FLASH_ATTN_WHEEL" ]]; then
    return
  fi
  step "Installing flash-attn wheel"
  "$VENV_PY" -m pip install "$FLASH_ATTN_WHEEL"
}

install_spit_cuda_helpers() {
  if [[ "${RESOLVED_TORCH:-}" == "cpu" ]]; then
    return
  fi

  step "Installing CUDA helper libraries for CuPy/SPiT RawKernel"
  "$VENV_PY" -m pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12

  site_packages="$("$VENV_PY" - <<'PY'
import site
print(next(p for p in site.getsitepackages() if p.endswith("site-packages")))
PY
)"
  cuda_runtime_dir="$site_packages/nvidia/cuda_runtime"
  cuda_nvrtc_dir="$site_packages/nvidia/cuda_nvrtc"
  if [[ ! -d "$cuda_runtime_dir" || ! -d "$cuda_nvrtc_dir" ]]; then
    echo "CUDA helper packages installed, but expected directories were not found." >&2
    return
  fi

  cat > "$VENV_DIR/spit_cuda_env.sh" <<EOF
export CUDA_PATH="$cuda_runtime_dir"
export LD_LIBRARY_PATH="$cuda_nvrtc_dir/lib:$cuda_runtime_dir/lib:\${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$cuda_nvrtc_dir/lib:$cuda_runtime_dir/lib:\${LIBRARY_PATH:-}"
export CPATH="$cuda_nvrtc_dir/include:$cuda_runtime_dir/include:\${CPATH:-}"
export CUPY_CACHE_DIR="\${CUPY_CACHE_DIR:-$ROOT/.cache/cupy}"
mkdir -p "\$CUPY_CACHE_DIR"
EOF
  echo "SPiT CUDA environment helper written to $VENV_DIR/spit_cuda_env.sh"
}

cd "$ROOT"

step "Resolving Python"
PY="$(resolve_python)"
"$PY" --version

step "Creating virtual environment at $VENV_DIR"
"$PY" -m venv "$VENV_DIR"

step "Upgrading pip tooling"
"$VENV_PY" -m pip install --upgrade pip setuptools wheel

install_torch

step "Installing project dependencies"
if [[ "$NO_CONSTRAINTS" -eq 0 && -f "$CONSTRAINTS_FILE" ]]; then
  "$VENV_PY" -m pip install -c "$CONSTRAINTS_FILE" -e ".[ultralytics,slic,opencv,dev,spit]"
else
  "$VENV_PY" -m pip install -e ".[ultralytics,slic,opencv,dev,spit]"
fi
install_spit_cuda_helpers

step "Preparing SPiT checkout"
mkdir -p "$THIRD_PARTY_DIR"
if [[ ! -d "$SPIT_DIR/.git" ]]; then
  command -v git >/dev/null 2>&1 || {
    echo "git was not found. Install Git or clone https://github.com/dsb-ifi/SPiT.git into third_party/SPiT manually." >&2
    exit 1
  }
  git clone https://github.com/dsb-ifi/SPiT.git "$SPIT_DIR"
else
  echo "SPiT already exists at $SPIT_DIR"
fi
git -C "$SPIT_DIR" fetch origin "$SPIT_REVISION"
git -C "$SPIT_DIR" checkout --detach "$SPIT_REVISION"

link_spit_checkout
install_flash_attn

export YOLO_CONFIG_DIR="$ROOT"

step "Checking imports"
"$VENV_PY" - <<'PY'
import sys
import torch
import torchvision
import ultralytics
import skimage
import cv2
import cupy
from spit.tokenizer.densesp import DenseSPEdgeEmbedder
from yolo_superpixels import AuxiliaryLossConfig, NeckInjectionConfig, SuperpixelAuxiliaryYOLO, SuperpixelNeckYOLO, SuperpixelYOLO
from yolo_superpixels.superpixels.qstar import QStarRunner
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
print("ultralytics", ultralytics.__version__)
print("skimage", skimage.__version__)
print("opencv", cv2.__version__, "ximgproc", hasattr(cv2, "ximgproc"))
print("cupy", cupy.__version__)
print("spit", DenseSPEdgeEmbedder.__name__)
print("yolo_superpixels", SuperpixelYOLO.__name__)
print("yolo_superpixels_neck", SuperpixelNeckYOLO.__name__, NeckInjectionConfig.__name__)
print("yolo_superpixels_auxiliary", SuperpixelAuxiliaryYOLO.__name__, AuxiliaryLossConfig.__name__)
print("qstar", QStarRunner.__name__)
PY

if [[ "$SKIP_TESTS" -eq 0 ]]; then
  step "Running tests"
  export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  "$VENV_PY" -B -m pytest -q -p no:cacheprovider
fi

if [[ "$RUN_SMOKE" -eq 1 ]]; then
  step "Running one-image YOLO12 smoke training"
  "$VENV_PY" examples/smoke_train_one_image.py --algorithm "$SMOKE_ALGORITHM" --model yolo12n.pt --imgsz 96
fi

if [[ "$RUN_NECK_SMOKE" -eq 1 ]]; then
  step "Running one-image YOLO12 neck-injection smoke training"
  if command -v nvidia-smi >/dev/null 2>&1; then
    neck_device="0"
  else
    neck_device="cpu"
  fi
  "$VENV_PY" examples/smoke_train_neck_injection.py \
    --model yolo12n.pt \
    --device "$neck_device" \
    --imgsz 64 \
    --fusion add concat cross_attention \
    --token-provider region \
    --region-algorithm torch_slic
fi

if [[ "$RUN_AUX_SMOKE" -eq 1 ]]; then
  step "Running one-image YOLO12 auxiliary-loss smoke training"
  if command -v nvidia-smi >/dev/null 2>&1; then
    aux_device="0"
  else
    aux_device="cpu"
  fi
  "$VENV_PY" examples/smoke_train_auxiliary_loss.py \
    --model yolo12n.pt \
    --device "$aux_device" \
    --imgsz 64 \
    --algorithm torch_slic
fi

step "Done"
echo "Activate with:"
echo "  source .venv/bin/activate"
