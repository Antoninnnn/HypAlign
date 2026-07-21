#!/bin/bash
#SBATCH --job-name=hypalign-ft-650m-asl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=16:00:00
#SBATCH --partition=gpu
#SBATCH --output=/scratch/group/aibi/Protein_LLM/HypAlign/experiments/hyp_ssf_probe/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yining_yang@tamu.edu

set -euo pipefail
umask 0002

module purge
ml GCC/12.2.0 CUDA/12.4.0 Anaconda3

export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL

REPO=/scratch/group/aibi/Protein_LLM/HypAlign
SHARED_ROOT=$REPO
CONDA_ENV=$REPO/.conda/envs/hypalign
RUN_LABEL=${RUN_LABEL:-ESM2-650M-FT-ASL}

export HYPALIGN_SHARED_ROOT=$SHARED_ROOT
export HYPALIGN_CACHE_ROOT=$SHARED_ROOT/.cache
export HF_HOME=$HYPALIGN_CACHE_ROOT/huggingface
export HF_HUB_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export XDG_CACHE_HOME=$HYPALIGN_CACHE_ROOT/xdg
export TORCH_HOME=$HYPALIGN_CACHE_ROOT/torch
export PIP_CACHE_DIR=$HYPALIGN_CACHE_ROOT/pip
export CONDA_PKGS_DIRS=$SHARED_ROOT/.conda/pkgs
export MPLCONFIGDIR=$HYPALIGN_CACHE_ROOT/matplotlib

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$MPLCONFIGDIR"

cd "$REPO"
mkdir -p experiments/hyp_ssf_probe/logs experiments/hyp_ssf_probe/checkpoints experiments/hyp_ssf_probe/results

echo "=== ESM-2 650M Fine-tuned GO-MF ASL Baseline ==="
echo "Node: $(hostname)"
echo "GPU(s):"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "Start: $(date)"
echo "Shared root: $SHARED_ROOT"
echo "Conda env: $CONDA_ENV"
echo "HF_HOME: $HF_HOME"

conda run --no-capture-output -p "$CONDA_ENV" torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    experiments/hyp_ssf_probe/run_finetune_go.py \
    --esm-model facebook/esm2_t33_650M_UR50D \
    --label "$RUN_LABEL" \
    --loss asl \
    --epochs 50 \
    --batch-size 8 \
    --grad-accum 4 \
    --backbone-lr 5e-5 \
    --head-lr 3e-4 \
    --pos-weight-cap 10 \
    --num-workers 2 \
    --save-every 5 \
    --no-data-parallel \
    --amp

echo "End: $(date)"
