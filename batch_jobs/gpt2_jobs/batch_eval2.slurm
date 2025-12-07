#!/bin/bash
#SBATCH -J qcgpt2
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH -o /home/caiosiq/qcgpt/batch_jobs/logs/obb_%j.out
#SBATCH -e /home/caiosiq/qcgpt/batch_jobs/logs/obb_%j.err

source ~/.bashrc
module load miniforge/24.3.0-0 || true
conda activate qcgpt

cd /orcd/home/002/caiosiq/qcgpt || cd /home/caiosiq/qcgpt
export PYTHONPATH=/orcd/home/002/caiosiq/qcgpt:/home/caiosiq/qcgpt:$PYTHONPATH

# Parameters

NUM_ROWS=${NUM_ROWS:-25}
NUM_COLS=${NUM_COLS:-25}
MAX_LEN=${MAX_LEN:-32}
MAX_GATES=${MAX_GATES:-6}
DECODE_STRATEGY=${DECODE_STRATEGY:-greedy}
BEAM_WIDTH=${BEAM_WIDTH:-5}
TOP_K=${TOP_K:-5}
TOP_P=${TOP_P:-0.9}

OUT_DIR=${OUT_DIR:-model_evaluations}
RUN_NAME=${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_$DECODE_STRATEGY}
CKPT=${CKPT:-/home/caiosiq/qcgpt/model_checkpoints/qcgpt2_mid_20251202_211824/transformer_v2_best.pt}
python3 qcgpt2/scripts2/eval_grid2.py \
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
