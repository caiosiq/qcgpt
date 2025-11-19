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
conda activate base

cd /home/caiosiq/qcgpt

# Parameters (override via environment: PREFIX, RUN_NAME, CKPT_PATH, EPOCHS, SAMPLES, BATCH, RAWDEPTH)
PREFIX=${PREFIX:-transformer_v1}
RUN_NAME=${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}
OUT_DIR=model_checkpoints
EPOCHS=${EPOCHS:-100}
SAMPLES=${SAMPLES:-10000}
BATCH=${BATCH:-64}
RAWDEPTH=${RAWDEPTH:-16}

python scripts/train_supervised.py \
  --num_epochs ${EPOCHS} \
  --num_samples ${SAMPLES} \
  --batch_size ${BATCH} \
  --raw_max_depth ${RAWDEPTH} \
  ${CKPT_PATH:+--ckpt ${CKPT_PATH}} \
  --basis_only \
  --out_dir ${OUT_DIR} \
  --run_name ${RUN_NAME} \
  --prefix ${PREFIX}

