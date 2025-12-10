#!/bin/bash
#SBATCH -J qcgpt2_eval_noise
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o evaluations_models/logs/eval_noise_%j.out
#SBATCH -e evaluations_models/logs/eval_noise_%j.err

source ~/.bashrc
module load miniforge/24.3.0-0 || true
conda activate qcgpt

cd /orcd/home/002/caiosiq/qcgpt || cd /home/caiosiq/qcgpt
export PYTHONPATH=$PYTHONPATH:.

# --- CONFIGURATION ---
# Point this to your NOISE-TRAINED model (Phase 3)
CKPT_MODEL="model_checkpoints/qcgpt2_finetuned_noised_from_pre_trained_cold_20251209_1411/transformer_v2_best.pt"

# Set the noise scale (should match training or be 5.0 for the test)
SCALE=8.0
SAMPLES=300
OUTPUT_DIR="evaluations_models/results_section_5_2"

echo "------------------------------------------------"
echo "Running Section 5.2 Evaluation: NISQ Robustness"
echo "Model: $(basename $CKPT_MODEL)"
echo "Noise Scale: ${SCALE}x"
echo "------------------------------------------------"
python3 qcgpt2/scripts2/evaluate_model_noise_robustness.py \
  --ckpt "$CKPT_MODEL" \
  --out "$OUTPUT_DIR" \
  --scale "$SCALE" \
  --samples "$SAMPLES"

echo "Done. Check $OUTPUT_DIR"