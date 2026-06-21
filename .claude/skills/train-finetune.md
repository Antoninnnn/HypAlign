---
description: Run ESM-2 fine-tuned baseline (reference upper bound, not geometry comparison)
---

Run the ESM-2 end-to-end fine-tuned baseline using Asymmetric Loss.
This is a **supervised classification** reference, not the geometry comparison
experiment. Use it to establish the performance ceiling.

## Steps

1. Navigate and verify cache:
   ```
   cd experiments/hyp_ssf_probe
   ls cache/protst_go_mf_decoded.pt   # must exist
   ```

2. Edit config at the top of `run_finetune_go.py` if needed:
   - **Local (RTX 4500, 25 GB)**: `ESM_MODEL = "facebook/esm2_t30_150M_UR50D"`,
     `BATCH_SIZE = 8`, `GRAD_ACCUM = 4`
   - **HPRC A100 (80 GB)**: change to `ESM_MODEL = "facebook/esm2_t33_650M_UR50D"`,
     `BATCH_SIZE = 8`, `GRAD_ACCUM = 8`

3. Run:
   ```
   conda run --no-capture-output -n pannot-infer python3 -u run_finetune_go.py
   ```
   Wall time: ~3 h on RTX 4500 (150M), ~6 h on A100 (650M).
   Best checkpoint saved to `checkpoints/esm2_ft_best.pt`.
   Final metrics printed to stdout and logged.

## Architecture

```
ESM-2 → mean-pool over sequence → Linear(esm_dim, 489) → sigmoid
```

Loss: Asymmetric Loss (γ+=0, γ-=4, clip=0.05).
Optimizer: AdamW, backbone LR=5e-5, head LR=3e-4, cosine + warmup.

## Expected results

| Model | Fmax | AUPR |
|-------|------|------|
| ESM-2 150M FT (local) | ~0.35–0.45 | ~0.40–0.50 |
| ESM-2 650M FT (HPRC) | ~0.45–0.50 | ~0.50–0.55 |
| ProtST-ESM-2 650M | ~0.55 | ~0.62 |

## Note on scientific role

This script answers "what is the supervised ceiling?" It is NOT a geometry
comparison. Fine-tuning ESM-2 on GO labels teaches the backbone to discriminate
GO terms directly, which confounds the geometry effect. Use `/train-probe` with
frozen ESM-2 650M for the clean geometry comparison.
