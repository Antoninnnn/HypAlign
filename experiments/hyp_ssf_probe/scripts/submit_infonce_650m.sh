#!/bin/bash
#SBATCH --job-name=hypalign-infonce-650m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --output=/scratch/group/aibi/Protein_LLM/HypAlign/experiments/hyp_ssf_probe/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yining_yang@tamu.edu

set -euo pipefail
umask 0002

module purge
ml GCC/12.2.0 CUDA/12.4.0 Anaconda3

REPO=/scratch/group/aibi/Protein_LLM/HypAlign
SHARED_ROOT=$REPO
CONDA_ENV=$REPO/.conda/envs/hypalign

export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
mkdir -p experiments/hyp_ssf_probe/logs \
         experiments/hyp_ssf_probe/cache \
         experiments/hyp_ssf_probe/checkpoints \
         experiments/hyp_ssf_probe/results

echo "=== ESM2-650M + NeuML PubMedBERT InfoNCE baselines ==="
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "Start: $(date)"
echo "Repo: $REPO"
echo "Conda env: $CONDA_ENV"
echo "HF_HOME: $HF_HOME"

echo
echo "=== Data/model preflight ==="
conda run --no-capture-output -p "$CONDA_ENV" python -u \
    experiments/hyp_ssf_probe/scripts/prefetch_data.py \
    --esm-model facebook/esm2_t33_650M_UR50D \
    --pubmedbert-model NeuML/pubmedbert-base-embeddings \
    --prepare-go-vocab \
    --skip-models

COMMON_ARGS=(
    --esm-model facebook/esm2_t33_650M_UR50D
    --text-model NeuML/pubmedbert-base-embeddings
    --epochs 200
    --embed-batch-size 16
    --text-batch-size 64
    --max-len 512
    --proj-dim 256
    --hidden-dim 512
    --lr 1e-3
    --weight-decay 1e-5
    --eval-every 5
    --init-logit-scale 10
    --max-logit-scale 100
    --seed 0
    --compute-protein-feats
)

echo
echo "=== Variant 1: pair InfoNCE, batch 256 ==="
conda run --no-capture-output -p "$CONDA_ENV" python -u \
    experiments/hyp_ssf_probe/run_infonce_go.py \
    "${COMMON_ARGS[@]}" \
    --loss pair_infonce \
    --batch-size 256 \
    --label infonce_esm2_650m_neuml_pair_bs256

echo
echo "=== Variant 2: pair InfoNCE, big batch 2048 ==="
conda run --no-capture-output -p "$CONDA_ENV" python -u \
    experiments/hyp_ssf_probe/run_infonce_go.py \
    "${COMMON_ARGS[@]}" \
    --loss pair_infonce \
    --batch-size 2048 \
    --label infonce_esm2_650m_neuml_pair_bs2048

echo
echo "=== Variant 3: multi-positive InfoNCE, batch 1024 ==="
conda run --no-capture-output -p "$CONDA_ENV" python -u \
    experiments/hyp_ssf_probe/run_infonce_go.py \
    "${COMMON_ARGS[@]}" \
    --loss multipos_infonce \
    --batch-size 1024 \
    --label infonce_esm2_650m_neuml_multipos_bs1024

echo
echo "=== Variant 4: full multi-positive InfoNCE over all GO terms, batch 1024 ==="
conda run --no-capture-output -p "$CONDA_ENV" python -u \
    experiments/hyp_ssf_probe/run_infonce_go.py \
    "${COMMON_ARGS[@]}" \
    --loss full_multipos_infonce \
    --batch-size 1024 \
    --label infonce_esm2_650m_neuml_fullmultipos_bs1024

echo "End: $(date)"
