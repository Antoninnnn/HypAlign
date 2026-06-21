# Hyperbolic Protein Function Embedding

Joint protein sequence–GO function embedding in Euclidean vs hyperbolic (Lorentz)
space. Evaluated on GO-MF function retrieval using CAFA-standard Fmax/AUPR.

## Setup

```bash
conda activate pannot-infer
cd experiments/hyp_ssf_probe
```

## Experiments

**Probe experiment** — frozen ESM-2 features + geometry comparison:
```bash
python run_experiment_go.py           # trains 4 conditions, ~60 epochs
python evaluate_checkpoints.py        # Fmax / AUPR / wFmax / nDCG / MAP
python analysis.py                    # generates figures/
```

**Fine-tuned baseline** — ESM-2 150M end-to-end (reference upper bound):
```bash
python run_finetune_go.py             # 50 epochs, requires ≥20 GB GPU
```

## Results (v1, frozen ESM-2 8M)

| Model | Fmax | AUPR | wFmax |
|-------|------|------|-------|
| Euclidean | 0.075 | 0.032 | 0.061 |
| Lorentz | 0.065 | 0.030 | 0.055 |
| Lorentz + MERU | **0.083** | **0.034** | **0.070** |
| Lorentz + MERU + DAG | 0.055 | 0.026 | 0.046 |

## Data

Downloaded automatically from `mila-intel/ProtST-GeneOntology-MF` on first run
and cached in `experiments/hyp_ssf_probe/cache/`.

## Reproducing on HPRC (TAMU Grace cluster)

```bash
# 1. Clone and set up environment
git clone git@github.com:Antoninnnn/HypAlign.git $SCRATCH/HypAlign
cd $SCRATCH/HypAlign
conda env create -f environment.yml          # creates 'hypalign' env

# 2. Pre-fetch models and data on login node (no GPU needed, ~10 min)
conda activate hypalign
python experiments/hyp_ssf_probe/scripts/prefetch_data.py

# 3. Submit probe job (ESM-2 650M frozen, 4 geometry conditions, ~4 h on A100)
cd experiments/hyp_ssf_probe/scripts
sbatch submit_probe_650m.sh
```

Results land in `experiments/hyp_ssf_probe/results/results_go.json`.
Checkpoints in `experiments/hyp_ssf_probe/checkpoints/`.

## References

- ProtST: Zhang et al. 2023 — protein text pre-training baseline
- MERU: Desai et al. 2023 — hyperbolic image-text contrastive learning
- ASL: Zamir et al. ICCV 2021 — asymmetric loss for multi-label classification
