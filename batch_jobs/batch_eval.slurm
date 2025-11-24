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
# CKPT, RUN_NAME, NUM_ROWS, NUM_COLS, MAX_LEN, MAX_GATES, OUT_DIR, DECODE_STRATEGY, BEAM_WIDTH, TOP_K, TOP_P
DECODE_STRATEGY=${DECODE_STRATEGY:-topk}
BEAM_WIDTH=${BEAM_WIDTH:-5}
TOP_K=${TOP_K:-5}
TOP_P=${TOP_P:-0.9}
OUT_DIR=${OUT_DIR:-model_evaluations}
RUN_NAME=${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}_${DECODE_STRATEGY}
CKPT=${CKPT:-/home/caiosiq/qcgpt/model_checkpoints/20251124_005954/transformer_v1_unitary_stash_sup0p0_u1p0_best.pt}
NUM_ROWS=${NUM_ROWS:-16}
NUM_COLS=${NUM_COLS:-16}
MAX_LEN=${MAX_LEN:-32}
MAX_GATES=${MAX_GATES:-12}


python3 scripts/eval_grid.py \
  --ckpt ${CKPT} \
  --num_rows ${NUM_ROWS} \
  --num_cols ${NUM_COLS} \
  --max_len ${MAX_LEN} \
  --max_gates_ref ${MAX_GATES} \
  --out_dir ${OUT_DIR} \
  --run_name ${RUN_NAME} \
  --decode_strategy ${DECODE_STRATEGY} \
  --beam_width ${BEAM_WIDTH} \
  --top_k ${TOP_K} \
  --top_p ${TOP_P}