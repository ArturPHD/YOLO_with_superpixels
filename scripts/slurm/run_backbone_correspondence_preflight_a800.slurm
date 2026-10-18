#!/bin/bash
#SBATCH --job-name=spit_p3_preflight
#SBATCH --partition=unrestricted
#SBATCH --gres=gpu:a800:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/export/home/radzivilov/YoloSuperpixel/YOLO_with_superpixels/runs/slurm/spit_p3_preflight_%j.out
#SBATCH --error=/export/home/radzivilov/YoloSuperpixel/YOLO_with_superpixels/runs/slurm/spit_p3_preflight_%j.err
#SBATCH --no-requeue

# One short real GPU lifecycle check before a multi-day P3/SPiT matrix. It uses
# frozen pretrained SPiT-S16, the native-backbone placement, and a tiny fraction
# of the existing actual-10 COCO subset. It never writes a scientific result.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/YoloSuperpixel/YOLO_with_superpixels}"
DATA_YAML="${DATA_YAML:-${PROJECT_DIR}/SmallData/yamls/coco_smalldata_actual_10.yaml}"
SPIT_WEIGHTS="${SPIT_WEIGHTS:-${PROJECT_DIR}/weights/spit/SPiT_model_SPiT_S16_grad.pth}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/preflight}"

module purge
module load devel/anaconda3
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

export PROJECT_DIR DATA_YAML SPIT_WEIGHTS RUN_ROOT
mkdir -p "${PROJECT_DIR}/runs/slurm" "${RUN_ROOT}"
test -f "${DATA_YAML}"
test -f "${SPIT_WEIGHTS}"

echo "Node: $(hostname)"
echo "Git commit: $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

python - <<'PY'
import os
from pathlib import Path

import torch

from yolo_superpixels import BackboneCorrespondenceConfig, SuperpixelBackboneYOLO

assert torch.cuda.is_available(), "A CUDA device was not allocated to the preflight job."
project = Path(os.environ["PROJECT_DIR"])
config = BackboneCorrespondenceConfig(
    placement="native_backbone",
    fusion="overlap_attention",
    token_provider="spit",
    spit_weights=os.environ["SPIT_WEIGHTS"],
    spit_train_mode="frozen",
    emb_dim=384,
    num_bins=16,
    heads=6,
    depth=12,
    keep_k=0,
    image_size=160,
    tokenizer="superpixel",
    bbox_reg=False,
    gate_init="tiny",
)
model = SuperpixelBackboneYOLO(project / "yolo12n.pt", config=config)
model.train(
    data=os.environ["DATA_YAML"],
    epochs=1,
    imgsz=160,
    batch=1,
    workers=0,
    fraction=0.001,
    device=0,
    seed=0,
    val=False,
    plots=False,
    save=False,
    project=os.environ["RUN_ROOT"],
    name="native_backbone_spit_s16",
    exist_ok=True,
)
print("P3/SPiT GPU preflight passed.")
PY
