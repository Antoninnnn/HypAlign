# HPRC Cluster Jobs

Jobs that require more compute than the local RTX 4500 Ada (25.2 GB) can handle
efficiently. Submit to TAMU HPRC (Grace or FASTER cluster, A100 nodes).

---

## Job 1 — ESM-2 650M Fine-tuned Euclidean Baseline

**Why on HPRC:** ESM-2 650M has 650M parameters. Fine-tuning with full attention
(batch=32, max_len=512) needs ~40 GB GPU memory. The local 4500 Ada has 25.2 GB,
which is too tight for batch_size > 4 without sacrificing sequence length.

**Script:** `run_finetune_go.py`  
**Change needed:** In the config block at the top, set:
```python
ESM_MODEL  = "facebook/esm2_t33_650M_UR50D"   # 1280-dim
BATCH_SIZE = 8
GRAD_ACCUM = 8     # effective batch = 64
```

**Expected SLURM config:**
```bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

module load GCC/12.2.0 CUDA/12.0 Miniconda3
conda activate pannot-infer

cd $SCRATCH/NLP/hyperbolic/experiments/hyp_ssf_probe
python run_finetune_go.py
```

**Expected output:** Fmax > 0.45 (comparable to ProtST-ESM-2-650M baseline)

**Comparison target (ProtST paper, Table 3):**
| Model           | GO-MF Fmax | GO-MF AUPR |
|-----------------|-----------|-----------|
| ESM-2 650M + FT | ~0.46     | ~0.54     |
| ProtST-ESM-2    | ~0.55     | ~0.62     |

---

## Job 2 — Hyperbolic Variants with Fine-tuned ESM-2 650M

**Why on HPRC:** Same as Job 1. Once we have a strong Euclidean baseline on HPRC,
run the hyperbolic variants:
- `Hyp-FT`: Lorentz head + BCE
- `Hyp-FT+MERU`: Lorentz head + BCE + MERU entailment
- `Hyp-FT+MERU+DAG`: Lorentz head + BCE + MERU + DAG

**Script:** `run_experiment_go_v2.py` (to be written after Job 1 baseline is confirmed)

**Expected wall time:** ~6h per condition on A100 (3 conditions = ~18h)

---

## Job 3 — ESM-2 3B Baseline (Exploratory)

**Why on HPRC:** ESM-2 3B requires ~24 GB model weights alone, plus gradients and
optimizer states (~48 GB). Even A100 80 GB would need gradient checkpointing + mixed
precision. Multi-GPU if needed.

**Status:** Lower priority. Do after Job 1 and 2 establish the 650M baseline.

**SLURM config:**
```bash
#SBATCH --gres=gpu:a100:2    # 2x A100 for model parallelism or DDP
#SBATCH --time=24:00:00
```

---

## Environment Setup on HPRC (one-time)

```bash
# On login node
module load GCC/12.2.0 CUDA/12.0 Miniconda3
conda create -n pannot-infer python=3.10 -y
conda activate pannot-infer
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.0 accelerate scikit-learn scipy umap-learn matplotlib
```

## Data Transfer to HPRC

```bash
# From local machine
rsync -avz --progress \
    /home/yining_yang/NLP/hyperbolic/experiments/hyp_ssf_probe/cache/ \
    netid@grace.hprc.tamu.edu:$SCRATCH/NLP/hyperbolic/experiments/hyp_ssf_probe/cache/

rsync -avz --progress \
    /home/yining_yang/NLP/hyperbolic/experiments/hyp_ssf_probe/*.py \
    netid@grace.hprc.tamu.edu:$SCRATCH/NLP/hyperbolic/experiments/hyp_ssf_probe/
```

All cached data (ESM-2 features, GO embeddings, OBO file, decoded sequences) in
`cache/` should transfer. ESM-2 model weights will be downloaded from HuggingFace
on first run (or pre-download with `huggingface-cli download`).

---

## Local Jobs (running or completed)

| Job | Model | Status |
|-----|-------|--------|
| Euclidean probe (v1) | ESM-2 8M frozen, InfoNCE | Done — Fmax=0.0748 |
| Hyp probe (v1) | ESM-2 8M frozen, InfoNCE | Done — Fmax=0.0650 |
| Hyp+MERU (v1) | ESM-2 8M frozen, InfoNCE+MERU | Done — Fmax=0.0828 |
| Hyp+MERU+DAG (v1) | ESM-2 8M frozen, InfoNCE+MERU+DAG | Done — Fmax=0.0545 |
| ESM-2 150M FT baseline | ESM-2 150M fine-tuned, ASL | **Running locally** |
