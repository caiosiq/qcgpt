#!/bin/bash
#SBATCH --job-name=qcgpt3_debug
#SBATCH --output=qcgpt3/slurm/logs/debug_%j.out
#SBATCH --error=qcgpt3/slurm/logs/debug_%j.err
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Environment Setup
source ~/.bashrc
module load miniforge/24.3.0-0 || true
conda activate qcgpt

export PYTHONPATH=$PYTHONPATH:.

# Python Path
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Running QCGPT3 Debug Script..."
echo "Date: $(date)"
echo "------------------------------------------------"

python qcgpt3/reproduce_issue.py

echo "------------------------------------------------"
echo "Debug Run Complete."
