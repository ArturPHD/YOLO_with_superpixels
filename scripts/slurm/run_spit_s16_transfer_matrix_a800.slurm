#!/bin/bash
#SBATCH --job-name=spit_s16_xfer
#SBATCH --partition=unrestricted
#SBATCH --nodelist=gpu-a800.cluster
#SBATCH --gres=gpu:a800:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=96:00:00
#SBATCH --array=0-5%1
#SBATCH --output=/export/home/radzivilov/YoloSuperpixel/YOLO_with_superpixels/runs/slurm/spit_s16_xfer_%A_%a.out
#SBATCH --error=/export/home/radzivilov/YoloSuperpixel/YOLO_with_superpixels/runs/slurm/spit_s16_xfer_%A_%a.err
#SBATCH --no-requeue

set -euo pipefail

PROJECT_DIR="${HOME}/YoloSuperpixel/YOLO_with_superpixels"
DATA_YAML="${HOME}/YoloSuperpixel/coco2017/data.yaml"
SPIT_WEIGHTS="${PROJECT_DIR}/weights/spit/SPiT_model_SPiT_S16_grad.pth"
SPIT_MD5="e49be7009c639c0ccda4bd68ed34e5af"
EPOCHS="${EPOCHS:-150}"
SPIT_FREEZE_EPOCHS="${SPIT_FREEZE_EPOCHS:-21}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

module purge
module load devel/anaconda3

cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

mkdir -p "${PROJECT_DIR}/runs/slurm"

if [ -f "${PROJECT_DIR}/.venv/spit_cuda_env.sh" ]; then
  source "${PROJECT_DIR}/.venv/spit_cuda_env.sh"
else
  export CUDA_PATH="${PROJECT_DIR}/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime"
  export LD_LIBRARY_PATH="${PROJECT_DIR}/.venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${PROJECT_DIR}/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${PROJECT_DIR}/.venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${PROJECT_DIR}/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:${LIBRARY_PATH:-}"
  export CPATH="${PROJECT_DIR}/.venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/include:${PROJECT_DIR}/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/include:${CPATH:-}"
  export CUPY_CACHE_DIR="${PROJECT_DIR}/.cache/cupy"
fi
mkdir -p "${CUPY_CACHE_DIR}"

case "${TASK_ID}" in
  0)
    MODEL="yolo12n.pt"
    MODEL_TAG="yolo12n"
    MODE="frozen"
    LR_SCALE="0.01"
    ;;
  1)
    MODEL="yolo12n.pt"
    MODEL_TAG="yolo12n"
    MODE="delayed"
    LR_SCALE="0.01"
    ;;
  2)
    MODEL="yolo12n.pt"
    MODEL_TAG="yolo12n"
    MODE="scratch"
    LR_SCALE="1.0"
    ;;
  3)
    MODEL="yolo26n.pt"
    MODEL_TAG="yolo26n"
    MODE="frozen"
    LR_SCALE="0.01"
    ;;
  4)
    MODEL="yolo26n.pt"
    MODEL_TAG="yolo26n"
    MODE="delayed"
    LR_SCALE="0.01"
    ;;
  5)
    MODEL="yolo26n.pt"
    MODEL_TAG="yolo26n"
    MODE="scratch"
    LR_SCALE="1.0"
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=${TASK_ID}" >&2
    exit 2
    ;;
esac

RUN_KEY="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${TASK_ID}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${PROJECT_DIR}/runs/full_spit_s16_transfer/${RUN_KEY}"
PROJECT_OUT="${RUN_DIR}/experiments"
RUN_NAME="${MODEL_TAG}_spit_s16_${MODE}_e${EPOCHS}"
mkdir -p "${PROJECT_OUT}"

exec > >(tee -a "${RUN_DIR}/launcher.log") 2>&1

COMMON_ARGS=(
  --model "${PROJECT_DIR}/${MODEL}"
  --data "${DATA_YAML}"
  --epochs "${EPOCHS}"
  --imgsz 640
  --batch 32
  --workers 8
  --device 0
  --fusion cross_attention
  --token-provider spit
  --spit-tokenizer superpixel
  --spit-emb-dim 384
  --spit-num-bins 16
  --spit-heads 6
  --spit-depth 12
  --spit-keep-k 0
  --spit-image-size 160
  --no-bbox-reg
  --spit-train-mode "${MODE}"
  --spit-lr-scale "${LR_SCALE}"
  --patience "${EPOCHS}"
  --project "${PROJECT_OUT}"
  --name "${RUN_NAME}"
  --exist-ok
)

EXTRA_ARGS=()
if [ "${MODE}" = "scratch" ]; then
  EXTRA_ARGS+=(--allow-random-spit)
else
  test -f "${SPIT_WEIGHTS}"
  echo "${SPIT_MD5}  ${SPIT_WEIGHTS}" | md5sum -c -
  EXTRA_ARGS+=(--spit-weights "${SPIT_WEIGHTS}")
fi
if [ "${MODE}" = "delayed" ]; then
  EXTRA_ARGS+=(--spit-freeze-epochs "${SPIT_FREEZE_EPOCHS}")
fi

cat > "${RUN_DIR}/manifest.txt" <<EOF
strategy=neck_injection
model=${MODEL_TAG}
data=${DATA_YAML}
epochs=${EPOCHS}
imgsz=640
batch=32
workers=8
fusion=cross_attention
token_provider=spit
spit_architecture=SPiT-S16-grad
spit_emb_dim=384
spit_num_bins=16
spit_heads=6
spit_depth=12
spit_keep_k=0
spit_image_size=160
spit_bbox_reg=false
spit_train_mode=${MODE}
spit_lr_scale=${LR_SCALE}
spit_freeze_epochs=$([ "${MODE}" = "delayed" ] && echo "${SPIT_FREEZE_EPOCHS}" || echo 0)
spit_weights=$([ "${MODE}" = "scratch" ] && echo random || echo "${SPIT_WEIGHTS}")
spit_weights_md5=$([ "${MODE}" = "scratch" ] && echo none || echo "${SPIT_MD5}")
EOF

echo "Run key: ${RUN_KEY}"
echo "Task: ${TASK_ID} -> ${MODEL_TAG} / ${MODE}"
echo "Epochs: ${EPOCHS}"
echo "SPiT freeze epochs: ${SPIT_FREEZE_EPOCHS}"
echo "SPiT LR scale: ${LR_SCALE}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Git commit: $(git rev-parse HEAD || true)"
echo "Git status:"
git status --short || true
nvidia-smi

python examples/train_ultralytics_neck.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"

echo "Finished at: $(date)"
nvidia-smi
