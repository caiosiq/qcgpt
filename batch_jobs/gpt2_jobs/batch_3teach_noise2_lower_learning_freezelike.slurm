#!/bin/bash
#SBATCH -J qcgpt2_optim_shock
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH -o /home/caiosiq/qcgpt/batch_jobs/logs/shock_lower_learning_freezelike_%j.out
#SBATCH -e /home/caiosiq/qcgpt/batch_jobs/logs/shock_lower_learning_freezelike_%j.err

source ~/.bashrc
module load miniforge/24.3.0-0 || true
conda activate qcgpt
cd /orcd/home/002/caiosiq/qcgpt
export PYTHONPATH=$PYTHONPATH:.

# --- INPUT ---
# Start from the high-fidelity Depth 8 expert
CKPT_INIT="/home/caiosiq/qcgpt/model_checkpoints/qcgpt2_mid_20251204_095733/transformer_v2_best.pt"

RUN_NAME="qcgpt2_finetuned_noised_lower_learning_freezelike_$(date +%Y%m%d_%H%M)"

# --- DATA ---
RAWDEPTH=8
SAMPLES=100000
BATCH=512

# --- DYNAMICS ---
EPOCHS=200

# High Learning Rate for Optimization (3e-5)
# Protected by Warmup
LR_ENC=1e-8
LR_DEC=2e-7
SCHEDULER="exp"
GAMMA=0.985



# --- LOSS CONFIG (The "Freedom" Config) ---
# Cut the tether. Pure RL.
LAMBDA_SUP=0.0
LAMBDA_U=1.0      # Physics is the primary constraint.
LAMBDA_NOISE=1.0
# Scale mapping from raw error rates to fidelity loss
NOISE_SCALE=5
# --- NOISE CONFIG (The "Shock") ---
USE_NOISE=1
SOFTMAX_TEMP=0.5
TEMP_MIN=0.1      # Same as SOFTMAX_TEMP for fixed temperature
TEMP_SCHEDULE="cosine"  # Fixed temperature, no annealing
T_MAX=${T_MAX:-200}
python3 -u qcgpt2/scripts2/train_supervised2.py \
  --ckpt ${CKPT_INIT} \
  --run_name ${RUN_NAME} \
  --num_epochs ${EPOCHS} \
  --num_samples ${SAMPLES} \
  --batch_size ${BATCH} \
  --lr_enc ${LR_ENC} \
  --lr_dec ${LR_DEC} \
  --weight_decay 0.01 \
  --scheduler ${SCHEDULER} \
  --gamma ${GAMMA} \
  --num_workers 16 \
  --pin_memory \
  --raw_max_depth ${RAWDEPTH} \
  --basis_only \
  --use_unitary_loss \
  --lambda_sup ${LAMBDA_SUP} \
  --lambda_U ${LAMBDA_U} \
  --use_noise \
  --lambda_noise ${LAMBDA_NOISE} \
  --noise_scale ${NOISE_SCALE} \
  --softmax_temp ${SOFTMAX_TEMP} \
  --temp_min ${TEMP_MIN} \
  --temp_schedule ${TEMP_SCHEDULE} \
  --t_max ${T_MAX} \
  --out_dir model_checkpoints
