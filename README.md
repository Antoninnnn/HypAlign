# Hyperbolic Protein Function Embedding

Joint protein sequence–GO-MF function embedding in Euclidean vs hyperbolic
(Lorentz) space, evaluated with CAFA-standard Fmax/AUPR.

## Results (v2 — frozen ESM2-650M, MLP projector)

| Model | Loss | Fmax | AUPR |
|-------|------|------|------|
| Euclidean | BCE | 0.112 | 0.040 |
| **Lorentz** | **BCE** | **0.143** | **0.059** |
| Lorentz + MERU | BCE | 0.143 | 0.051 |
| Lorentz + MERU + DAG | BCE | 0.144 | 0.052 |
| **Euclidean** | **MulSupCon** | **0.209** | **0.083** |
| Lorentz | MulSupCon | 0.175 | 0.069 |
| Lorentz + MERU | MulSupCon | 0.178 | 0.072 |
| Lorentz + MERU + DAG | MulSupCon | 0.180 | 0.076 |

*v1 reference (InfoNCE, ESM2-8M, linear): Euclidean Fmax=0.075 / Lorentz+MERU Fmax=0.083*

## Data

### Files tracked in this repository

| File | Size | Description |
|------|------|-------------|
| `experiments/hyp_ssf_probe/data/go_mf_vocab.json` | 64 KB | GO term ID ↔ index mapping (489 MF terms with 50–5000 annotations) |
| `experiments/hyp_ssf_probe/data/go_term_embs.pt` | 1.5 MB | PubMedBERT CLS embeddings for 489 GO term descriptions, shape `[489, 768]` |
| `experiments/hyp_ssf_probe/results/results_v2_bce.json` | tiny | Test Fmax/AUPR for v2 BCE conditions |
| `experiments/hyp_ssf_probe/results/results_v2_msc.json` | tiny | Test Fmax/AUPR for v2 MulSupCon conditions |

### Files generated during setup (not in git)

All generated files land in `experiments/hyp_ssf_probe/cache/`.

| File | Size | How generated |
|------|------|---------------|
| `gene_ontology_mf_{train,valid,test}.csv` | 122 MB | Downloaded from HuggingFace `mila-intel/ProtST-GeneOntology-MF` by `prefetch_data.py` |
| `go-basic.obo` | 31 MB | Downloaded from `purl.obolibrary.org` by `prefetch_data.py` |
| `protst_go_mf_decoded.pt` | 75 MB | Parsed from CSVs by `prefetch_data.py` — protein sequences (decoded from ESM2 token IDs) + multi-hot GO-MF target tensors for each split |
| `esm2_esm2_t33_650M_UR50D_go_feats.pt` | 164 MB | Mean-pool ESM2-650M last hidden state over residues, computed by `prepare_features.py` (GPU required, ~15 min on A100) |

### Dataset details

- **Source:** `mila-intel/ProtST-GeneOntology-MF` (Zhang et al., ProtST 2023)
- **Splits:** train 27,496 / validation 3,053 / test 2,991 proteins
- **Labels:** 489 GO Molecular Function terms (filtered to 50–5000 protein annotations)
- **Multi-label:** avg 4.3 GO terms per protein (min 1, max ~30)
- **Protein representation:** ESM2-650M mean-pool over sequence residues → `[N, 1280]`
- **GO term representation:** PubMedBERT CLS on text `"FUNCTION: <term name>."` → `[489, 768]`

### Preprocessing pipeline

```
CSV columns: prot_seq (ESM2 token IDs), targets (489-dim multi-hot float list)

prot_seq  →  decode token IDs using ESM2 vocabulary  →  amino acid string
targets   →  torch.tensor([0.0, 1.0, 0.0, ...])  →  stacked into [N, 489]

Saved as: {'train': {'seqs': [...], 'targets': Tensor[N, 489]}, ...}
```

The GO vocabulary (`go_mf_vocab.json`) maps GO IDs (e.g. `GO:0003700`) to
integer indices 0–488, and stores the human-readable term names used as
PubMedBERT input.

## Reproducing on HPRC (TAMU Grace cluster)

```bash
# 1. Clone and create conda environment
git clone git@github.com:Antoninnnn/HypAlign.git $SCRATCH/HypAlign
cd $SCRATCH/HypAlign
conda env create -f environment.yml        # creates 'hypalign' env (~5 min)
conda activate hypalign

# 2. Download data and build CPU-side cache (login node, no GPU, ~15 min)
python experiments/hyp_ssf_probe/scripts/prefetch_data.py

# 3. Compute frozen encoder features (GPU required, ~15 min on A100)
sbatch experiments/hyp_ssf_probe/scripts/submit_prepare_features.sh

# 4. Run geometry comparison experiments (once features are cached)
sbatch experiments/hyp_ssf_probe/scripts/submit_probe_v2.sh           # BCE loss
sbatch experiments/hyp_ssf_probe/scripts/submit_probe_v2_mulsupcon.sh # MulSupCon loss
```

Results land in `experiments/hyp_ssf_probe/results/results_v2_{bce,msc}.json`.
Checkpoints in `experiments/hyp_ssf_probe/checkpoints/v2_{bce,msc}_*.pt`.

## Setup (local)

```bash
conda activate pannot-infer
cd experiments/hyp_ssf_probe
python -u run_experiment_go_v2.py --loss bce
python -u run_experiment_go_v2.py --loss mulsupcon
```

The training scripts auto-compute and cache features on the first run.

## Architecture

```
ESM2-650M (frozen)  →  mean-pool  →  [N, 1280]
                                           │
                                    seq_head MLP
                              Linear(1280→512) → GELU → Linear(512→256)
                                           │
                                    geometry head
                               Euclidean: L2-normalize → [N, 256]
                               Lorentz:   exp_map0(·, κ) → H^256_κ

PubMedBERT (frozen)  →  CLS  →  [489, 768]
                                      │
                               text_head MLP  (same structure, 768→512→256)
                                      │
                                geometry head
```

Learned parameters: seq_head weights, text_head weights, temperature τ,
curvature κ (Lorentz only), scale α (Lorentz only).

## References

- ProtST: Zhang et al. 2023 — protein–text pre-training and GO-MF dataset
- MERU: Desai et al. 2023 — hyperbolic image-text contrastive learning (entailment cone loss)
- MulSupCon: Zhang & Wu, AAAI 2024 — multi-label supervised contrastive loss
- ASL: Ridnik et al. ICCV 2021 — asymmetric loss for multi-label classification
