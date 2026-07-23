# Hyperbolic SSF Project — Agent Harness

## What this project is

We study whether **hyperbolic geometry** (Lorentz model) improves joint
protein Sequence–Structure–Function (SSF) embedding compared to Euclidean
baselines. The active experiment (`experiments/hyp_ssf_probe/`) aligns frozen
ESM-2 sequence features with PubMedBERT GO-term descriptions in either
Euclidean or hyperbolic space, then evaluates GO-MF function retrieval using
Fmax / AUPR (CAFA-standard metrics).

Comparison baseline is ProtST (Zhang et al. 2023). Reference for hyperbolic
loss design is MERU (Desai et al. 2023).

## Repository layout

```
experiments/hyp_ssf_probe/
├── run_experiment_go.py      # v1 probe: 4 conditions, frozen ESM-2 8M + InfoNCE
├── run_finetune_go.py        # ESM-2 150M end-to-end fine-tune + ASL (strong baseline)
├── evaluate_checkpoints.py   # Fmax/AUPR/wFmax/nDCG/MAP on saved checkpoints
├── analysis.py               # 7 analysis figures (radius dist, geodesic, UMAP, …)
├── HPRC_JOBS.md              # SLURM job specs for A100 runs
├── report.md                 # Written analysis report
├── cache/                    # Data files (not in git — see Data section)
├── checkpoints/              # Saved .pt probe checkpoints (not in git)
├── results/                  # JSON metric files (not in git)
└── figures/                  # PNG figures (not in git)
```

## Environment

```bash
conda activate pannot-infer   # Python 3.10, PyTorch 2.x, transformers 4.40
```

All HuggingFace model weights are cached locally. ESM-2 150M is at
`~/.cache/huggingface/hub/models--facebook--esm2_t30_150M_UR50D/`.

## Data

Historical note: the `cache/` directory is **not in git** (large binary files).
The old GO-MF prototype used the generic names below. Current analysis-ready
GO-MF/BP/CC runs use namespace-specific cache names documented in
`experiments/hyp_ssf_probe/ANALYSIS_READY_RESULTS.md` and `README.md`.

Legacy GO-MF files:

| File | Size | How to get |
|------|------|-----------|
| `protst_go_mf_decoded.pt` | ~50 MB | Run `run_experiment_go.py` once (downloads from HF) |
| `esm2_go_feats.pt` | ~30 MB | Same — computed during first run |
| `go_term_embs.pt` | ~1.4 MB | Same |
| `go_mf_vocab.json` | tiny | Same |
| `go-basic.obo` | 32 MB | Downloaded automatically |

On HPRC: rsync the entire `cache/` from local before running:
```bash
rsync -avz cache/ netid@grace.hprc.tamu.edu:$SCRATCH/hyp_ssf_probe/cache/
```

## Experiments

### Completed (local, RTX 4500 Ada 25.2 GB)

| Condition | Fmax | AUPR | Notes |
|-----------|------|------|-------|
| Euclidean | 0.0748 | 0.0318 | frozen ESM-2 8M, cosine sim |
| Hyp | 0.0650 | 0.0298 | Lorentz, InfoNCE |
| Hyp+MERU λ=0.5 | 0.0828 | 0.0337 | + entailment cone loss |
| Hyp+MERU+DAG λ=0.5 | 0.0545 | 0.0261 | + DAG supervision (hurts) |

Checkpoints saved in `checkpoints/`. Analysis figures in `figures/`.
Full written report in `report.md`.

### In progress (local)

- `run_finetune_go.py`: ESM-2 150M fine-tuned end-to-end with Asymmetric Loss.
  Used as reference upper bound — not the geometry comparison experiment.
  Expected Fmax ~0.3–0.4 on local GPU.

### Pending (requires HPRC A100)

See `HPRC_JOBS.md` for full specs. Priority order:

1. **ESM-2 650M frozen** (geometry comparison) — run `run_experiment_go.py`
   with `ESM2_HF = "facebook/esm2_t33_650M_UR50D"` and `ESM_DIM = 1280`.
   This is the clean comparison: same task, only backbone size changes.

2. **ESM-2 650M fine-tuned** (supervised ceiling) — run `run_finetune_go.py`
   with `ESM_MODEL = "facebook/esm2_t33_650M_UR50D"`.
   Reference point only; not part of geometry experiment.

## Key design decisions

- **Why not fine-tune ESM-2 for geometry comparison?** Fine-tuning on GO labels
  makes ESM-2 learn to predict GO terms directly. The geometry effect gets
  swamped by backbone adaptation. The clean comparison keeps ESM-2 frozen and
  varies only the embedding space geometry.

- **Protein→GO is the meaningful retrieval direction.** 489 GO terms serve as
  the retrieval space; each test protein retrieves its true GO annotations.
  GO→Protein is not meaningful because proteins are indistinguishable by a
  single GO term.

- **InfoNCE false-negative problem.** In multi-label setting (avg 4.3 GO terms
  per protein), other proteins sharing GO terms are wrongly treated as negatives.
  Fine-tuned BCE baseline avoids this.

- **ASL loss** (Asymmetric Loss, Zamir 2021): γ+=0, γ-=4, clip=0.05. Reduces
  gradient from easy negatives in the severe class imbalance of GO terms.

## Skills (slash commands)

| Command | What it does |
|---------|-------------|
| `/train-probe` | Run v1 4-condition probe experiment |
| `/train-finetune` | Run ESM-2 150M fine-tune baseline |
| `/evaluate` | Evaluate all saved checkpoints |
| `/hprc-submit` | Generate and submit SLURM job to HPRC |
