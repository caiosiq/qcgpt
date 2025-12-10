#!/bin/bash
#SBATCH -J qcgpt2_unitary_debug
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -o /home/caiosiq/qcgpt/batch_jobs/logs/obb_%j.out
#SBATCH -e /home/caiosiq/qcgpt/batch_jobs/logs/obb_%j.err

source ~/.bashrc
module load miniforge/24.3.0-0 || true
conda activate qcgpt

cd /orcd/home/002/caiosiq/qcgpt || cd /home/caiosiq/qcgpt
export PYTHONPATH=/orcd/home/002/caiosiq/qcgpt:/home/caiosiq/qcgpt:$PYTHONPATH
export PYTHONUNBUFFERED=1

# Parameters
CKPT=${CKPT:-/home/caiosiq/qcgpt/model_checkpoints/qcgpt2_sup_20251129_042451/transformer_v2_best.pt}

python3 -u qcgpt2/scripts2/eval_unitary_debug.py \
  --ckpt ${CKPT} \
  --num_samples 50
