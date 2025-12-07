#!/bin/bash
#SBATCH -J qcgpt2_compare
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o /home/caiosiq/qcgpt/batch_jobs/logs/compare_results.out

source ~/.bashrc
module load miniforge/24.3.0-0 || true
conda activate qcgpt

cd /orcd/home/002/caiosiq/qcgpt || cd /home/caiosiq/qcgpt
export PYTHONPATH=/orcd/home/002/caiosiq/qcgpt:/home/caiosiq/qcgpt:$PYTHONPATH
# PATHS TO YOUR MODELS
# 1. The best model from Phase 1 (Imitation)
# CKPT_PRE="/home/caiosiq/qcgpt/model_checkpoints/qcgpt2_sup_20251129_042451/transformer_v2_best.pt"
CKPT_PRE="/home/caiosiq/qcgpt/model_checkpoints/qcgpt2_mid_20251204_095733/transformer_v2_best.pt"

# 2. The best model from Phase 2 (Physics/Mid-Training) - Use the one you just trained
# CKPT_MID="/home/caiosiq/qcgpt/model_checkpoints/qcgpt2_mid_20251204_095733/transformer_v2_best.pt"
CKPT_MID="/home/caiosiq/qcgpt/model_checkpoints/finetuned_noised_20251206_1133/transformer_v2_e200.pt"

# Output folder
OUT="/home/caiosiq/qcgpt/comparison_results/$(date +%Y%m%d_%H%M%S)"

python3 qcgpt2/scripts2/compare_models.py \
  --ckpt_pre ${CKPT_PRE} \
  --ckpt_mid ${CKPT_MID} \
  --num_samples 5000 \
  --batch_size 256 \
  --out_dir ${OUT} \
  --raw_max_depth 8