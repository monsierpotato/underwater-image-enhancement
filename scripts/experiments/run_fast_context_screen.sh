#!/usr/bin/env bash
set -euo pipefail

# Reproducible fast architecture screen for Kaggle/Linux CUDA.
# Override any setting as an environment variable, for example:
#   DATA_ROOT=/kaggle/input/euvp-dataset/EUVP EPOCHS=12 bash scripts/experiments/run_fast_context_screen.sh

DATA_ROOT="${DATA_ROOT:-./datasets/EUVP}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./fast_screen}"
EUVP_SUBSET="${EUVP_SUBSET:-underwater_scenes}"
PRIOR_METHOD="${PRIOR_METHOD:-gupdm}"
CROP_SIZE="${CROP_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-12}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SEED="${SEED:-42}"
THREADS="${THREADS:-4}"
NUM_GPUS="${NUM_GPUS:-2}"
L1_WEIGHT="${L1_WEIGHT:-1.0}"
PERCEPTUAL_WEIGHT="${PERCEPTUAL_WEIGHT:-1.0}"
SSIM_WEIGHT="${SSIM_WEIGHT:-0.0}"

MODELS=(
  unet_5ch
  asppunet_5ch
  mambabottleneck_5ch
  mambaaspp_5ch
)

mkdir -p "${OUTPUT_ROOT}/checkpoints" "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/results"

python - <<'PY'
import torch
from uwir.models.mamba_unet import _MAMBA_CUDA

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Fused Mamba selective scan: {_MAMBA_CUDA}")
if not torch.cuda.is_available():
    print("WARNING: this screen is intended for a Kaggle GPU accelerator.")
if not _MAMBA_CUDA:
    print("WARNING: mamba-ssm is unavailable; Mamba variants will use the slower fallback.")
PY

for model in "${MODELS[@]}"; do
  echo "Running ${model} (seed=${SEED})"
  python -m uwir.cli.train \
    --model "${model}" \
    --dataset euvp \
    --data-train-euvp "${DATA_ROOT}" \
    --euvp-subset "${EUVP_SUBSET}" \
    --prior-method "${PRIOR_METHOD}" \
    --crop-size "${CROP_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LEARNING_RATE}" \
    --seed "${SEED}" \
    --threads "${THREADS}" \
    --num-gpus "${NUM_GPUS}" \
    --l1-weight "${L1_WEIGHT}" \
    --perceptual-weight "${PERCEPTUAL_WEIGHT}" \
    --ssim-weight "${SSIM_WEIGHT}" \
    --pretrained-backbone false \
    --snapshots "${EPOCHS}" \
    --early-stop-patience "${EPOCHS}" \
    --checkpoint-dir "${OUTPUT_ROOT}/checkpoints" \
    --log-dir "${OUTPUT_ROOT}/logs" \
    --val-folder "${OUTPUT_ROOT}/results" \
    --run-name "fast_${model}_seed${SEED}"
done

python - "${OUTPUT_ROOT}/checkpoints" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for history_path in root.glob("fast_*/training_history.json"):
    history = json.loads(history_path.read_text())
    if not history.get("val_psnr"):
        continue
    best_index = max(range(len(history["val_psnr"])), key=history["val_psnr"].__getitem__)
    rows.append(
        (
            history_path.parent.name,
            best_index + 1,
            history["val_psnr"][best_index],
            history["val_ssim"][best_index],
            history["val_loss"][best_index],
        )
    )

rows.sort(key=lambda row: row[2], reverse=True)
print("\nFast-screen ranking (validation PSNR):")
print(f"{'run':<55} {'epoch':>5} {'PSNR':>8} {'SSIM':>8} {'val loss':>10}")
for run, epoch, psnr, ssim, val_loss in rows:
    print(f"{run:<55} {epoch:>5} {psnr:>8.3f} {ssim:>8.4f} {val_loss:>10.4f}")
PY
