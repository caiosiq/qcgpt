#!/bin/bash
#SBATCH -J qcgpt2_eval_algos
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH -o evaluations_models/logs/eval_algos_%j.out
#SBATCH -e evaluations_models/logs/eval_algos_%j.err

source ~/.bashrc
module load miniforge/24.3.0-0 || true
conda activate qcgpt

cd /orcd/home/002/caiosiq/qcgpt || cd /home/caiosiq/qcgpt
export PYTHONPATH=$PYTHONPATH:.

# Use your best Unitary Model (Phase 2 is usually best for exact algos)
CKPT_MODEL="model_checkpoints/qcgpt2_mid_20251204_095733/transformer_v2_best.pt"
OUTPUT_DIR="evaluations_models/results_section_5_3"

echo "------------------------------------------------"
echo "Running Section 5.3: Algorithm Synthesis"
echo "Model: $(basename $CKPT_MODEL)"
echo "------------------------------------------------"

python3 qcgpt2/scripts2/evaluate_model_creates_qft.py \
  --ckpt "$CKPT_MODEL" \
  --out "$OUTPUT_DIR"

echo "Done."