#!/bin/bash
#SBATCH --job-name=hypalign-probe-v2-msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yining_yang@tamu.edu

module purge
module load GCC/12.2.0 CUDA/12.4.0 Miniconda3/23.5.2-0

conda activate hypalign

REPO=$SCRATCH/HypAlign
cd $REPO

echo "=== HypAlign Probe v2 (ESM2-650M, MLP, MulSupCon) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"

python -u experiments/hyp_ssf_probe/run_experiment_go_v2.py --loss mulsupcon

echo "End: $(date)"
