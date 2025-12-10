#!/bin/bash
#SBATCH -J noise_test
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH -o /home/caiosiq/qcgpt/batch_jobs/logs/verification_%j.out
#SBATCH -e /home/caiosiq/qcgpt/batch_jobs/logs/verification_%j.err

source ~/.bashrc
module load miniforge/24.3.0-0 || true
conda activate qcgpt

echo USING_ENV=$CONDA_DEFAULT_ENV
conda run -n qcgpt python -V || true
conda run -n qcgpt python -c 'import sys, pkgutil; print("PYTHON", sys.executable); print("HAS_QISKIT", pkgutil.find_loader("qiskit") is not None); print("HAS_AER", pkgutil.find_loader("qiskit_aer") is not None)'

export PYTHONPATH=/orcd/home/002/caiosiq/qcgpt:/home/caiosiq/qcgpt:$PYTHONPATH
cd /orcd/home/002/caiosiq/qcgpt || cd /home/caiosiq/qcgpt

# Run the verification test strictly inside env
conda run -n qcgpt python -u qcgpt2/tests2/test_cost_gates.py
