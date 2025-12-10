#!/bin/bash
#SBATCH -J qcgpt2_eval_fid
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o evaluations_models/logs/eval_fid_%j.out
#SBATCH -e evaluations_models/logs/eval_fid_%j.err

source ~/.bashrc
module load miniforge/24.3.0-0 || true
conda activate qcgpt

cd /orcd/home/002/caiosiq/qcgpt || cd /home/caiosiq/qcgpt
export PYTHONPATH=$PYTHONPATH:.

# --- CONFIGURATION ---
# Phase 1: The original Supervised Pre-trained Model
CKPT_PHASE1="model_checkpoints/qcgpt2_sup_20251129_042451/transformer_v2_best.pt"

# Phase 2: The New SGD Mid-Trained Model (UPDATE THIS PATH!)
CKPT_PHASE2="model_checkpoints/qcgpt2_mid_20251204_095733/transformer_v2_best.pt"

OUTPUT_DIR="evaluations_models/results_section_5_1"
SAMPLES=500

echo "------------------------------------------------"
echo "Running Section 5.1 Evaluation: Fidelity Recovery"
echo "Phase 1: $(basename $CKPT_PHASE1)"
echo "Phase 2: $(basename $CKPT_PHASE2)"
echo "------------------------------------------------"

python3 qcgpt2/scripts2/evaluate_model_fidelities.py \
  --p1 "$CKPT_PHASE1" \
  --p2 "$CKPT_PHASE2" \
  --out "$OUTPUT_DIR" \
  --samples "$SAMPLES"

echo "Done. Results saved to $OUTPUT_DIR"