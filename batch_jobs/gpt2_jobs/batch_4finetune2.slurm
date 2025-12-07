#!/bin/bash
#SBATCH -J qcgpt2_midtrain
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

# Config
PREFIX=${PREFIX:-transformer_v2}
RUN_NAME=${RUN_NAME:-qcgpt2_curriculum_$(date +%Y%m%d_%H%M%S)}
OUT_DIR=model_checkpoints
CKPT_INIT=${CKPT_INIT:=/home/caiosiq/qcgpt/model_checkpoints/qcgpt2_mid_20251204_095733/transformer_v2_best.pt}
SAMPLES=${SAMPLES:-200000}
EPOCHS=${EPOCHS:-200}
VAL_SPLIT=${VAL_SPLIT:-0.05}
BATCH=${BATCH:-256}
LR_ENC=${LR_ENC:-1e-6}
LR_DEC=${LR_DEC:-1e-5}
WD=${WD:-0.01}
SCHEDULER=${SCHEDULER:-exp}
T_MAX=${T_MAX:-200}
STEP_SIZE=${STEP_SIZE:-100}
GAMMA=${GAMMA:-0.985}
NUM_WORKERS=${NUM_WORKERS:-16}
PIN_MEMORY=${PIN_MEMORY:-1}
RAWDEPTH=${RAWDEPTH:-8} # Not used for curriculum learning, can be set to any value
BASIS_ONLY=${BASIS_ONLY:-1}
N_RANDOM=${N_RANDOM:-0}

# Loss mix: soft unitary reconstruction, temp anneal 1.0→0.1
LAMBDA_SUP=${LAMBDA_SUP:-1.0}
LAMBDA_U=${LAMBDA_U:-1.0}
USE_NOISE=${USE_NOISE:-1}
LAMBDA_NOISE=${LAMBDA_NOISE:-0.0} # Not used for curriculum learning, can be set to any value
SOFTMAX_TEMP=${SOFTMAX_TEMP:-1.0}
TEMP_MIN=${TEMP_MIN:-0.1}
TEMP_SCHEDULE=${TEMP_SCHEDULE:-cosine}

# Curriculum learning
USE_CURRICULUM=${USE_CURRICULUM:-1}
CURRICULUM_START_DEPTH=${CURRICULUM_START_DEPTH:-8}
CURRICULUM_END_DEPTH=${CURRICULUM_END_DEPTH:-32}
CURRICULUM_MAX_NOISE=${CURRICULUM_MAX_NOISE:-0.1}
CURRICULUM_NOISE_WARMUP=${CURRICULUM_NOISE_WARMUP:-30}
CURRICULUM_DEPTH_STEP=${CURRICULUM_DEPTH_STEP:-4}
CURRICULUM_INCREASE_EVERY=${CURRICULUM_INCREASE_EVERY:-10}

python3 -u qcgpt2/scripts2/train_supervised2.py \
  --ckpt ${CKPT_INIT} \
  --num_epochs ${EPOCHS} \
  --num_samples ${SAMPLES} \
  --val_split ${VAL_SPLIT} \
  --batch_size ${BATCH} \
  --lr_enc ${LR_ENC} \
  --lr_dec ${LR_DEC} \
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
  --use_unitary_loss \
  --lambda_sup ${LAMBDA_SUP} \
  --lambda_U ${LAMBDA_U} \
  $( [ "$USE_NOISE" = "1" ] && echo "--use_noise" ) \
  --lambda_noise ${LAMBDA_NOISE} \
  $( [ "$USE_CURRICULUM" = "1" ] && echo "--use_curriculum" ) \
  --curriculum_start_depth ${CURRICULUM_START_DEPTH} \
  --curriculum_end_depth ${CURRICULUM_END_DEPTH} \
  --curriculum_max_noise ${CURRICULUM_MAX_NOISE} \
  --curriculum_noise_warmup ${CURRICULUM_NOISE_WARMUP} \
  --curriculum_depth_step ${CURRICULUM_DEPTH_STEP} \
  --curriculum_increase_every ${CURRICULUM_INCREASE_EVERY} \
  --softmax_temp ${SOFTMAX_TEMP} \
  --temp_min ${TEMP_MIN} \
  --temp_schedule ${TEMP_SCHEDULE} \
  --out_dir ${OUT_DIR} \
  --run_name ${RUN_NAME} \
  --prefix ${PREFIX}
