#!/usr/bin/env bash
# =============================================================================
# run_ablation.sh  –  Run ablation_train_multi_run.py inside a tmux session
#
# Features:
#   • Persistent tmux session (survives SSH disconnect)
#   • Full output tee'd to a timestamped log file under ./logs/
#   • A .done file (containing exit code) is written on completion
#
# Usage:
#   bash run_ablation.sh
#   SESSION=myrun bash run_ablation.sh   # custom tmux session name
# =============================================================================

SESSION="${SESSION:-ablation}"
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-uwir}"

# Training arguments (edit as needed)
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/datasets/EUVP}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_DIR}/checkpoints/ablation_multi}"
VAL_FOLDER="${VAL_FOLDER:-${PROJECT_DIR}/results/ablation_multi}"
PRIOR="${PRIOR:-udcp}"
N_EPOCHS="${N_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_RUNS="${NUM_RUNS:-5}"
SEEDS="${SEEDS:-0 1 2 3 4}"

# Derived paths
mkdir -p "${PROJECT_DIR}/logs"
TS=$(date +"%Y%m%d_%H%M%S")
LOGFILE="${PROJECT_DIR}/logs/ablation_run_${TS}.log"
DONE_MARKER="${PROJECT_DIR}/logs/ablation_run_${TS}.done"
INNER_SCRIPT="${PROJECT_DIR}/logs/ablation_run_${TS}_inner.sh"

# Check if session already exists
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[run_ablation] tmux session '${SESSION}' already exists."
    echo "  Attach with:  tmux attach -t ${SESSION}"
    echo "  Kill first:   tmux kill-session -t ${SESSION}"
    exit 0
fi

# Write the inner script to a real file (avoids quoting issues)
# conda activate requires sourcing conda's init first in non-interactive shells
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/root/miniconda3")

cat > "${INNER_SCRIPT}" <<INNEREOF
#!/usr/bin/env bash

# Keep the tmux window open on error so the message is readable
set -uo pipefail
trap 'echo ""; echo "[ERROR] Script failed at line $LINENO — tmux window staying open for 60s"; sleep 60' ERR

LOGFILE="${LOGFILE}"
DONE_MARKER="${DONE_MARKER}"
PROJECT_DIR="${PROJECT_DIR}"

# Activate conda environment
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "================================================================"
echo " ablation_train_multi_run  |  started at \$(date)"
echo " python   -> \$(which python)"
echo " log      -> \${LOGFILE}"
echo "================================================================"

cd "\${PROJECT_DIR}"

python ablation_train_multi_run.py \\
    --data_train_euvp "${DATA_ROOT}" \\
    --checkpoint_dir  "${CKPT_DIR}" \\
    --val_folder      "${VAL_FOLDER}" \\
    --prior_method    "${PRIOR}" \\
    --nEpochs         "${N_EPOCHS}" \\
    --batchSize       "${BATCH_SIZE}" \\
    --num_runs        "${NUM_RUNS}" \\
    --seeds ${SEEDS} \\
    2>&1 | tee "\${LOGFILE}"

EXIT_CODE=\${PIPESTATUS[0]}

echo ""
echo "================================================================"
echo " Job finished at \$(date)  |  exit code: \${EXIT_CODE}"
echo "================================================================"

echo "\${EXIT_CODE}" > "\${DONE_MARKER}"
echo "Results JSON : ${VAL_FOLDER}/ablation_train_multi_run_results.json"
echo "Log file     : \${LOGFILE}"
echo ""
echo "Re-attach with: tmux attach -t ${SESSION}"
INNEREOF

chmod +x "${INNER_SCRIPT}"

# Launch tmux session running the script file
tmux new-session -d -s "${SESSION}" bash "${INNER_SCRIPT}"

echo "[run_ablation] Session '${SESSION}' started."
echo ""
echo "  Attach (watch live):   tmux attach -t ${SESSION}"
echo "  Detach (keep running): Ctrl+B then D"
echo "  Tail the log:          tail -f ${LOGFILE}"
echo "  Check if done:         cat ${DONE_MARKER}   # 0 = success"
echo "  Kill session:          tmux kill-session -t ${SESSION}"
