#!/bin/bash
#SBATCH -J qcgpt3_s1
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o /home/caiosiq/qcgpt/qcgpt3/slurm/logs/big_entangle_stage1_%j.out
#SBATCH -e /home/caiosiq/qcgpt/qcgpt3/slurm/logs/big_entangle_stage1_%j.err

source ~/.bashrc
module load miniforge/24.3.0-0 || true
conda activate qcgpt
cd /home/caiosiq/qcgpt
export PYTHONPATH=$PYTHONPATH:.

# Create logs directory if not exists
mkdir -p /home/caiosiq/qcgpt/qcgpt3/slurm/logs

echo "------------------------------------------------"
echo "Starting Stage 1: The Bootcamp"
echo "Date: $(date)"
echo "------------------------------------------------"

python3 -u qcgpt3/scripts/train.py --config qcgpt3/configs/big_entangle_stage1_bootcamp.json

# --- RESUME OPTION ---
# If the job timed out, comment out the line above and uncomment the line below to resume:
# python3 -u qcgpt3/scripts/train.py --config qcgpt3/configs/big_entangle_stage1_bootcamp.json --resume_dir model_checkpoints/qcgpt3_big_entangle_stage1_bootcamp

echo "Stage 1 Complete."
