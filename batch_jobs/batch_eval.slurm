#!/bin/bash
#SBATCH -J qcgpt_eval
#SBATCH -p mit_preemptable 
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o /home/caiosiq/qcgpt/batch_jobs/logs/obb_%j.out
#SBATCH -e /home/caiosiq/qcgpt/batch_jobs/logs/obb_%j.err

source ~/.bashrc
module load miniforge/24.3.0-0 || true
conda activate qcgpt

cd /orcd/home/002/caiosiq/qcgpt || cd /home/caiosiq/qcgpt
export PYTHONPATH=/orcd/home/002/caiosiq/qcgpt:/home/caiosiq/qcgpt:$PYTHONPATH

# Ensure optional deps for Qiskit drawing are present
pip install --no-input --quiet pylatexenc || true

# Parameters (override via environment)
# CKPT, RUN_NAME, NUM_ROWS, NUM_COLS, MAX_LEN, MAX_GATES, OUT_DIR
OUT_DIR=${OUT_DIR:-model_evaluations}
RUN_NAME=${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}
CKPT=${CKPT:-model_checkpoints/20251121_103655/transformer_v1_best.pt}
NUM_ROWS=${NUM_ROWS:-16}
NUM_COLS=${NUM_COLS:-16}
MAX_LEN=${MAX_LEN:-32}
MAX_GATES=${MAX_GATES:-6}

python3 scripts/eval_grid.py \
  --ckpt ${CKPT} \
  --num_rows ${NUM_ROWS} \
  --num_cols ${NUM_COLS} \
  --max_len ${MAX_LEN} \
  --max_gates_ref ${MAX_GATES} \
  --out_dir ${OUT_DIR} \
  --run_name ${RUN_NAME}