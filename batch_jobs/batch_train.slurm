#!/bin/bash
#SBATCH -J qcgpt
#SBATCH -p mit_preemptable 
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH -o /home/caiosiq/qcgpt/batch_jobs/logs/obb_%j.out
#SBATCH -e /home/caiosiq/qcgpt/batch_jobs/logs/obb_%j.err

source ~/.bashrc
module load miniforge/24.3.0-0 || true
conda activate qcgpt

cd /orcd/home/002/caiosiq/qcgpt || cd /home/caiosiq/qcgpt
export PYTHONPATH=/orcd/home/002/caiosiq/qcgpt:/home/caiosiq/qcgpt:$PYTHONPATH

PREFIX=${PREFIX:-transformer_v1}
RUN_NAME=${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=model_checkpoints
EPOCHS=${EPOCHS:-300}
SAMPLES=${SAMPLES:-20000}
BATCH=${BATCH:-256}
RAWDEPTH=${RAWDEPTH:-32}
BASIS_ONLY=${BASIS_ONLY:-1}
N_RANDOM=${N_RANDOM:-0}
CKPT_PATH=/home/caiosiq/qcgpt/model_checkpoints/20251121_003310/transformer_v1_best.pt

python3 scripts/train_supervised.py \
  --num_epochs ${EPOCHS} \
  --num_samples ${SAMPLES} \
  --batch_size ${BATCH} \
  --raw_max_depth ${RAWDEPTH} \
  $( [ -n "${CKPT_PATH}" ] && echo "--ckpt ${CKPT_PATH}" ) \
  $( [ "$BASIS_ONLY" = "1" ] && echo "--basis_only" ) \
  --n_random_states ${N_RANDOM} \
  --out_dir ${OUT_DIR} \
  --run_name ${RUN_NAME} \
  --prefix ${PREFIX}

