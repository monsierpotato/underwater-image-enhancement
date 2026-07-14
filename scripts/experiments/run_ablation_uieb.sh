#!/usr/bin/env bash
# =============================================================================
# run_ablation_uieb.sh
# Run ablation_train_multi_run_uieb.py inside a persistent tmux session.
#
# 4 variants x 2 priors (UDCP + GUPDM) x 3 seeds  =  24 training runs
# Dataset: UIEB  (800 train/val  |  90 test)
#
# Usage:
#   bash run_ablation_uieb.sh
#   SESSION=myrun bash run_ablation_uieb.sh   # custom tmux session name
# =============================================================================

SESSION="${SESSION:-ablation_uieb}"
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-uwir}"

# ---- Configurable arguments (override via env vars) ----------------------
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/datasets/UIEB}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_DIR}/checkpoints/ablation_uieb}"
VAL_FOLDER="${VAL_FOLDER:-${PROJECT_DIR}/results/ablation_uieb}"
PRIOR_METHODS="${PRIOR_METHODS:-udcp gupdm}"   # space-separated list
N_EPOCHS="${N_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_RUNS="${NUM_RUNS:-3}"
SEEDS="${SEEDS:-0 1 2}"

# ---- Derived paths -------------------------------------------------------
mkdir -p "${PROJECT_DIR}/logs"
TS=$(date +"%Y%m%d_%H%M%S")
LOGFILE="${PROJECT_DIR}/logs/ablation_uieb_${TS}.log"
DONE_MARKER="${PROJECT_DIR}/logs/ablation_uieb_${TS}.done"
INNER_SCRIPT="${PROJECT_DIR}/logs/ablation_uieb_${TS}_inner.sh"

# ---- Guard: don't double-launch ------------------------------------------
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[run_ablation_uieb] tmux session '${SESSION}' already exists."
    echo "  Attach with : tmux attach -t ${SESSION}"
    echo "  Kill first  : tmux kill-session -t ${SESSION}"
    exit 0
fi

CONDA_BASE=$(conda info --base 2>/dev/null || echo "/root/miniconda3")

# ---- Write inner script --------------------------------------------------
cat > "${INNER_SCRIPT}" << INNEREOF
#!/usr/bin/env bash
set -uo pipefail
trap 'echo ""; echo "[ERROR] Script failed at line \$LINENO — keeping tmux open for 60s"; sleep 60' ERR

LOGFILE="${LOGFILE}"
DONE_MARKER="${DONE_MARKER}"
PROJECT_DIR="${PROJECT_DIR}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "================================================================"
echo " ablation_train_multi_run_uieb  |  started at \$(date)"
echo " python  -> \$(which python)"
echo " log     -> \${LOGFILE}"
echo "================================================================"

cd "\${PROJECT_DIR}"

python ablation_train_multi_run_uieb.py \
    --data_uieb      "${DATA_ROOT}" \
    --checkpoint_dir "${CKPT_DIR}" \
    --val_folder     "${VAL_FOLDER}" \
    --prior_methods  ${PRIOR_METHODS} \
    --nEpochs        "${N_EPOCHS}" \
    --batchSize      "${BATCH_SIZE}" \
    --num_runs       "${NUM_RUNS}" \
    --seeds          ${SEEDS} \
    2>&1 | tee "\${LOGFILE}"

EXIT_CODE=\${PIPESTATUS[0]}

echo ""
echo "================================================================"
echo " Job finished at \$(date)  |  exit code: \${EXIT_CODE}"
echo "================================================================"
echo "\${EXIT_CODE}" > "\${DONE_MARKER}"
echo "Results JSON : ${VAL_FOLDER}/ablation_uieb_results.json"
echo "Log file     : \${LOGFILE}"
echo ""
echo "Re-attach with: tmux attach -t ${SESSION}"
INNEREOF

chmod +x "${INNER_SCRIPT}"

# ---- Launch --------------------------------------------------------------
tmux new-session -d -s "${SESSION}" bash "${INNER_SCRIPT}"

echo "[run_ablation_uieb] Session '${SESSION}' started."
echo ""
echo "  Variants       : 4 (unet_3ch, unet_4ch_t, unet_4ch_b, unet_5ch)"
echo "  Priors         : ${PRIOR_METHODS}"
echo "  Seeds          : ${SEEDS}  =>  $(($(echo ${SEEDS} | wc -w))) runs/combo"
echo "  Total runs     : 24"
echo ""
echo "  Attach (watch live)   : tmux attach -t ${SESSION}"
echo "  Detach (keep running) : Ctrl+B then D"
echo "  Tail the log          : tail -f ${LOGFILE}"
echo "  Check if done         : cat ${DONE_MARKER}  # 0 = success"
echo "  Kill session          : tmux kill-session -t ${SESSION}"
