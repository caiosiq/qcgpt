#!/bin/bash
#SBATCH -J qcgpt2_plot
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:30:00
#SBATCH -o /home/caiosiq/qcgpt/batch_jobs/logs/analyse_%j.out
#SBATCH -e /home/caiosiq/qcgpt/batch_jobs/logs/analyse_%j.err
source ~/.bashrc
conda activate qcgpt
cd /orcd/home/002/caiosiq/qcgpt

# Define your 3 champion checkpoints
CKPT_PRE="/home/caiosiq/qcgpt/model_checkpoints/qcgpt2_sup_20251129_042451/transformer_v2_best.pt"
CKPT_MID="/home/caiosiq/qcgpt/model_checkpoints/qcgpt2_mid_20251204_095733/transformer_v2_best.pt"
CKPT_FINE="/home/caiosiq/qcgpt/model_checkpoints/qcgpt2_finetuned_noised_20251206_1133/transformer_v2_best.pt"

# Run comparison
python3 qcgpt2/scripts2/analyse_current_training.py\
  --ckpt_pre ${CKPT_PRE} \
  --ckpt_mid ${CKPT_MID} \
  --ckpt_fine ${CKPT_FINE} \
  --num_samples 2000 \
  --raw_max_depth 8 \
  --out_dir final_report_depth8