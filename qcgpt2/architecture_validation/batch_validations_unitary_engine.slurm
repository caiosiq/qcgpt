#!/bin/bash
#SBATCH -J qcgpt2_valid_unitary
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o /home/caiosiq/qcgpt/qcgpt2/architecture_validation/logs/validation_unitary_%j.out
#SBATCH -e /home/caiosiq/qcgpt/qcgpt2/architecture_validation/logs/validation_unitary_%j.err

source ~/.bashrc
module load miniforge/24.3.0-0 || true
conda activate qcgpt
cd /orcd/home/002/caiosiq/qcgpt
export PYTHONPATH=$PYTHONPATH:.

echo "------------------------------------------------"
echo "Running Architecture Validation Suite"
echo "Partition: mit_preemptable_gpu"
echo "Date: $(date)"
echo "------------------------------------------------"

echo "Validating Differentiable Unitary Engine..."
python3 -u qcgpt2/architecture_validation/validate_unitary_engine.py

echo "------------------------------------------------"
echo "Validation Complete. Check 'qcgpt2/architecture_validation/results/' for plots."