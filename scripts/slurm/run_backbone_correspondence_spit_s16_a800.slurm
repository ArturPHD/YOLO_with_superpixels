#!/bin/bash
#SBATCH --job-name=spit_p3corr
#SBATCH --partition=unrestricted
#SBATCH --nodelist=gpu-a800.cluster
#SBATCH --gres=gpu:a800:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --time=72:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/export/home/radzivilov/YoloSuperpixel/YOLO_with_superpixels/runs/slurm/spit_p3corr_%A_%a.out
#SBATCH --error=/export/home/radzivilov/YoloSuperpixel/YOLO_with_superpixels/runs/slurm/spit_p3corr_%A_%a.err
#SBATCH --no-requeue

# Submit twice with a frozen dataset identity: once for full COCO and once for
# SmallData actual-10. The two array tasks are native_backbone and isolated_p3.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${HOME}/YoloSuperpixel/YOLO_with_superpixels}"
DATA_YAML="${DATA_YAML:?Set DATA_YAML to the exact COCO data YAML before sbatch.}"
DATA_LABEL="${DATA_LABEL:?Set DATA_LABEL, for example full_coco or smalldata_actual10.}"
SPIT_WEIGHTS="${SPIT_WEIGHTS:-${PROJECT_DIR}/weights/spit/SPiT_model_SPiT_S16_grad.pth}"
SPIT_MD5="${SPIT_MD5:-e49be7009c639c0ccda4bd68ed34e5af}"
MODEL="${MODEL:-yolo12n.pt}"
EPOCHS="${EPOCHS:-70}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-0}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

module purge
module load devel/anaconda3
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

if [ -f "${PROJECT_DIR}/.venv/spit_cuda_env.sh" ]; then
  source "${PROJECT_DIR}/.venv/spit_cuda_env.sh"
fi
export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-${PROJECT_DIR}/.cache/cupy}"
mkdir -p "${CUPY_CACHE_DIR}" "${PROJECT_DIR}/runs/slurm"

test -f "${DATA_YAML}"
test -f "${SPIT_WEIGHTS}"
echo "${SPIT_MD5}  ${SPIT_WEIGHTS}" | md5sum -c -

case "${TASK_ID}" in
  0) PLACEMENT="native_backbone" ;;
  1) PLACEMENT="isolated_p3" ;;
  *) echo "Unknown SLURM_ARRAY_TASK_ID=${TASK_ID}" >&2; exit 2 ;;
esac

RUN_KEY="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${TASK_ID}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${PROJECT_DIR}/runs/backbone_correspondence/${DATA_LABEL}/${RUN_KEY}"
RUN_NAME="${MODEL%.pt}_spit_s16_${PLACEMENT}_${DATA_LABEL}_e${EPOCHS}"
mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/launcher.log") 2>&1

cat > "${RUN_DIR}/manifest.txt" <<EOF
strategy=backbone_correspondence
placement=${PLACEMENT}
fusion=overlap_attention
model=${MODEL}
data_yaml=${DATA_YAML}
data_label=${DATA_LABEL}
epochs=${EPOCHS}
imgsz=640
batch=${BATCH}
workers=${WORKERS}
seed=${SEED}
spit_architecture=SPiT-S16-grad
spit_train_mode=frozen
spit_emb_dim=384
spit_num_bins=16
spit_heads=6
spit_depth=12
spit_keep_k=0
spit_image_size=160
spit_bbox_reg=false
spit_weights=${SPIT_WEIGHTS}
spit_weights_md5=${SPIT_MD5}
EOF

echo "Node: $(hostname)"
echo "Run: ${RUN_NAME}"
echo "Git commit: $(git rev-parse HEAD)"
git status --short
nvidia-smi

python examples/train_ultralytics_backbone.py \
  --model "${PROJECT_DIR}/${MODEL}" \
  --data "${DATA_YAML}" \
  --epochs "${EPOCHS}" \
  --imgsz 640 \
  --batch "${BATCH}" \
  --workers "${WORKERS}" \
  --device 0 \
  --seed "${SEED}" \
  --placement "${PLACEMENT}" \
  --fusion overlap_attention \
  --token-provider spit \
  --spit-weights "${SPIT_WEIGHTS}" \
  --spit-train-mode frozen \
  --spit-emb-dim 384 \
  --spit-num-bins 16 \
  --spit-heads 6 \
  --spit-depth 12 \
  --spit-keep-k 0 \
  --spit-image-size 160 \
  --no-bbox-reg \
  --patience "${EPOCHS}" \
  --project "${RUN_DIR}/experiments" \
  --name "${RUN_NAME}" \
  --exist-ok

echo "Finished at: $(date)"
nvidia-smi
