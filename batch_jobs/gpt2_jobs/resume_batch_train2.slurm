#!/bin/bash
#SBATCH -J qcgpt2_resume
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

RESUME_DIR=${RESUME_DIR:-$1}
if [ -z "$RESUME_DIR" ]; then
  echo "ERROR: Provide checkpoint folder. Usage: sbatch resume_batch_train2.slurm /path/to/checkpoint_folder" >&2
  exit 1
fi
export PYTHONUNBUFFERED=1

python3 -u qcgpt2/scripts2/train_supervised2.py --resume_dir "${RESUME_DIR}"
