#!/bin/bash
#SBATCH --job-name=hypalign-probe-650m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yining_yang@tamu.edu

module purge
module load GCC/12.2.0 CUDA/12.4.0 Miniconda3/23.5.2-0

conda activate hypalign

# Repo root assumed at $SCRATCH/HypAlign (adjust if different)
REPO=$SCRATCH/HypAlign
cd $REPO

echo "=== ESM-2 650M Frozen Probe ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python -u experiments/hyp_ssf_probe/run_experiment_go.py

echo "End: $(date)"
