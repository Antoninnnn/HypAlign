#!/bin/bash
#SBATCH --job-name=hypalign-v2-cleanmf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --output=/scratch/group/aibi/Protein_LLM/HypAlign/experiments/hyp_ssf_probe/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yining_yang@tamu.edu

set -euo pipefail
umask 0002

LOSS=${1:?Usage: sbatch submit_probe_v2_clean_mf.sh bce|mulsupcon}
if [[ "$LOSS" != "bce" && "$LOSS" != "mulsupcon" ]]; then
    echo "LOSS must be one of: bce, mulsupcon" >&2
    exit 2
fi

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

echo "=== HypAlign Probe v2 clean GO-MF (${LOSS}, ESM2-650M, NeuML mean) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"
echo "Repo: $REPO"
echo "Conda env: $CONDA_ENV"
echo "HF_HOME: $HF_HOME"

conda run --no-capture-output -p "$CONDA_ENV" python -u \
    experiments/hyp_ssf_probe/run_experiment_go_v2.py \
    --loss "$LOSS" \
    --text-model NeuML/pubmedbert-base-embeddings \
    --pooling mean \
    --splits-cache protst_go_mf_decoded.pt \
    --vocab-file go_mf_vocab.json \
    --feature-cache esm2_esm2_t33_650M_UR50D_protst_go_mf_feats.pt \
    --term-embedding-cache go_terms_protst_go_mf_NeuML_pubmedbert-base-embeddings.pt \
    --run-tag _cleanmf

echo "End: $(date)"
