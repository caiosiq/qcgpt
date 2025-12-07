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
export PYTHONUNBUFFERED=1

# --- Configuration ---
PREFIX=${PREFIX:-transformer_v2}
RUN_NAME=${RUN_NAME:-qcgpt2_sup_$(date +%Y%m%d_%H%M%S)}
OUT_DIR=model_checkpoints

# DATA STRATEGY: More unique samples, fewer repeats
SAMPLES=${SAMPLES:-500000}  # Increased from 10k
EPOCHS=${EPOCHS:-240}       # Decreased from 300 (Total steps approx same, but more variety)
VAL_SPLIT=${VAL_SPLIT:-0.05} # MOVED UP

# TRAINING HYPERPARAMS
BATCH=${BATCH:-4096}
LR=${LR:-2e-4}              # Bumped slightly for large batch size
WD=${WD:-0.00}
SCHEDULER=cosine
T_MAX=${EPOCHS}
STEP_SIZE=${STEP_SIZE:-100}
GAMMA=${GAMMA:-0.5}

# WORKER CONFIG
NUM_WORKERS=${NUM_WORKERS:-16}
PIN_MEMORY=${PIN_MEMORY:-1}

# MODEL/TASK CONFIG
RAWDEPTH=${RAWDEPTH:-8}
BASIS_ONLY=${BASIS_ONLY:-1}
N_RANDOM=${N_RANDOM:-0}

# LOSS CONFIG (Supervised Baseline)
USE_UNITARY_LOSS=${USE_UNITARY_LOSS:-0}
LAMBDA_SUP=${LAMBDA_SUP:-1.0}
LAMBDA_U=${LAMBDA_U:-0.0}
SOFTMAX_TEMP=${SOFTMAX_TEMP:-1.0}

python3 -u qcgpt2/scripts2/train_supervised2.py \
  --num_epochs ${EPOCHS} \
  --num_samples ${SAMPLES} \
  --val_split ${VAL_SPLIT} \
  --batch_size ${BATCH} \
  --lr ${LR} \
  --weight_decay ${WD} \
  --scheduler ${SCHEDULER} \
  --t_max ${T_MAX} \
  --step_size ${STEP_SIZE} \
  --gamma ${GAMMA} \
  --num_workers ${NUM_WORKERS} \
  $( [ "$PIN_MEMORY" = "1" ] && echo "--pin_memory" ) \
  --raw_max_depth ${RAWDEPTH} \
  $( [ "$BASIS_ONLY" = "1" ] && echo "--basis_only" ) \
  --n_random_states ${N_RANDOM} \
  $( [ "$USE_UNITARY_LOSS" = "1" ] && echo "--use_unitary_loss" ) \
  --lambda_sup ${LAMBDA_SUP} \
  --lambda_U ${LAMBDA_U} \
  --softmax_temp ${SOFTMAX_TEMP} \
  --out_dir ${OUT_DIR} \
  --run_name ${RUN_NAME} \
  --prefix ${PREFIX}
