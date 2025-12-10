#!/bin/bash
#SBATCH -J qcgpt2_mid_sgd_curr
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH -o /home/caiosiq/qcgpt/batch_jobs/logs/mid_sgd_curr_%j.out
#SBATCH -e /home/caiosiq/qcgpt/batch_jobs/logs/mid_sgd_curr_%j.err

source ~/.bashrc
module load miniforge/24.3.0-0 || true
conda activate qcgpt
cd /orcd/home/002/caiosiq/qcgpt
export PYTHONPATH=$PYTHONPATH:.

# --- INPUT: PRE-TRAINED MODEL (Phase 1) ---
# REPLACE THIS with your actual Pre-Trained (Syntax-only) checkpoint path
CKPT_INIT="/home/caiosiq/qcgpt/model_checkpoints/qcgpt2_sup_20251129_042451/transformer_v2_best.pt"

RUN_NAME="qcgpt2_mid_sgd_curr_$(date +%Y%m%d_%H%M)"

# --- CONFIG ---
OPTIMIZER="sgd"
MOMENTUM=0.0
# SGD needs higher LR than Adam. 1e-4 is standard for Transformers with SGD.
LR_ENC=1e-4
LR_DEC=5e-4
SCHEDULER="cosine" 

# --- CURRICULUM ---
USE_CURRICULUM=1
START_DEPTH=2
END_DEPTH=8
DEPTH_STEP=2
INCREASE_EVERY=20 

# --- LOSS (Unitary Focus) ---
LAMBDA_SUP=0.1    # Anchor to syntax
LAMBDA_U=1.0      # Focus on Math
LAMBDA_NOISE=0.0  # No noise yet
NOISE_SCALE=1.0   # Irrelevant here

# --- DYNAMICS ---
SOFTMAX_TEMP=1.5  # Constant, stable
TEMP_MIN=1.5
TEMP_SCHEDULE="none"

python3 -u qcgpt2/scripts2/train_supervised2.py \
  --ckpt ${CKPT_INIT} \
  --run_name ${RUN_NAME} \
  --num_epochs 200 \
  --num_samples 100000 \
  --batch_size 512 \
  --lr_enc ${LR_ENC} \
  --lr_dec ${LR_DEC} \
  --optimizer ${OPTIMIZER} \
  --momentum ${MOMENTUM} \
  --scheduler ${SCHEDULER} \
  --warmup_epochs 0 \
  --num_workers 16 \
  --pin_memory \
  --use_curriculum \
  --curriculum_start_depth ${START_DEPTH} \
  --curriculum_end_depth ${END_DEPTH} \
  --curriculum_depth_step ${DEPTH_STEP} \
  --curriculum_increase_every ${INCREASE_EVERY} \
  --basis_only \
  --use_unitary_loss \
  --lambda_sup ${LAMBDA_SUP} \
  --lambda_U ${LAMBDA_U} \
  --use_noise 0 \
  --lambda_noise ${LAMBDA_NOISE} \
  --noise_scale ${NOISE_SCALE} \
  --softmax_temp ${SOFTMAX_TEMP} \
  --temp_min ${TEMP_MIN} \
  --temp_schedule ${TEMP_SCHEDULE} \
  --out_dir model_checkpoints