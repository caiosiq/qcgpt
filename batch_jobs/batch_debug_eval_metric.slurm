#!/bin/bash
#SBATCH -J qcgpt2_compare
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o /home/caiosiq/qcgpt/batch_jobs/logs/eval_metric_%j.out

source ~/.bashrc
module load miniforge/24.3.0-0 || true
conda activate qcgpt
cd /orcd/home/002/caiosiq/qcgpt

# Use your BEST Pre-Trained model
CKPT="/home/caiosiq/qcgpt/model_checkpoints/qcgpt2_sup_20251129_042451/transformer_v2_best.pt"

python3 qcgpt2/scripts2/debug_eval_metric.py --ckpt ${CKPT}