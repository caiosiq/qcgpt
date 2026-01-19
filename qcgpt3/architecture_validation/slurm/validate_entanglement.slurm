#!/bin/bash
#SBATCH -J qcgpt3_entang
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o /home/caiosiq/qcgpt/qcgpt3/architecture_validation/logs/entanglement_%j.out
#SBATCH -e /home/caiosiq/qcgpt/qcgpt3/architecture_validation/logs/entanglement_%j.err

source ~/.bashrc
module load miniforge/24.3.0-0 || true
conda activate qcgpt
cd /home/caiosiq/qcgpt
export PYTHONPATH=$PYTHONPATH:.

echo "------------------------------------------------"
echo "Running QCGPT3 Entanglement Validation"
echo "Date: $(date)"
echo "------------------------------------------------"

python3 -u qcgpt3/architecture_validation/validate_entanglement.py
