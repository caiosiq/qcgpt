#!/bin/bash
#SBATCH -J qcgpt
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
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

PREFIX=${PREFIX:-transformer_v1_unitary_stash_sup0p0_u1p0}
RUN_NAME=${RUN_NAME:-$PREFIX_$(date +%Y%m%d_%H%M%S)}
OUT_DIR=model_checkpoints
EPOCHS=${EPOCHS:-80}
SAMPLES=${SAMPLES:-50000} # Keep samples high so it generalizes!
BATCH=${BATCH:-2048}
RAWDEPTH=${RAWDEPTH:-32}
BASIS_ONLY=${BASIS_ONLY:-1}
N_RANDOM=${N_RANDOM:-0}
# CKPT_PATH=/home/caiosiq/qcgpt/model_checkpoints/20251122_213036/transformer_v1_unitary_stash_sup0p9_u0p1_e40.pt
USE_UNITARY_LOSS=${USE_UNITARY_LOSS:-1}
LAMBDA_SUP=${LAMBDA_SUP:-0.0}
LAMBDA_U=${LAMBDA_U:-1.0}

LR=${LR:-1e-4}
SCHEDULER=${SCHEDULER:-cosine}
T_MAX=${T_MAX:-100}
STEP_SIZE=${STEP_SIZE:-50}
GAMMA=${GAMMA:-0.5}
MEM_LOG_INTERVAL=${MEM_LOG_INTERVAL:-0}
SOFTMAX_TEMP=${SOFTMAX_TEMP:-0.1}

python3 scripts/train_supervised.py \
  --num_epochs ${EPOCHS} \
  --num_samples ${SAMPLES} \
  --batch_size ${BATCH} \
  --raw_max_depth ${RAWDEPTH} \
  $( [ -n "${CKPT_PATH}" ] && echo "--ckpt ${CKPT_PATH}" ) \
  $( [ "$BASIS_ONLY" = "1" ] && echo "--basis_only" ) \
  --n_random_states ${N_RANDOM} \
  $( [ "$USE_UNITARY_LOSS" = "1" ] && echo "--use_unitary_loss" ) \
  --lambda_sup ${LAMBDA_SUP} \
  --lambda_U ${LAMBDA_U} \
  --lr ${LR} \
  --scheduler ${SCHEDULER} \
  --t_max ${T_MAX} \
  --step_size ${STEP_SIZE} \
  --gamma ${GAMMA} \
  --softmax_temp ${SOFTMAX_TEMP} \
  --out_dir ${OUT_DIR} \
  --run_name ${RUN_NAME} \
  --prefix ${PREFIX}

export MEM_LOG_INTERVAL=${MEM_LOG_INTERVAL}

