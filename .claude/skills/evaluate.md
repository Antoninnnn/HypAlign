---
description: Evaluate saved probe checkpoints with full metric suite (Fmax, AUPR, wFmax, nDCG, MAP)
---

Evaluate all saved probe checkpoints and print a comparison table.

## Steps

1. Verify checkpoints exist:
   ```
   ls experiments/hyp_ssf_probe/checkpoints/
   ```
   Expected: `probe_Euclidean.pt`, `probe_Hyp.pt`, `probe_HyppMERU_lam05.pt`,
   `probe_HyppMERUpDAG_lam05.pt`

2. Run:
   ```
   cd experiments/hyp_ssf_probe
   conda run --no-capture-output -n pannot-infer python3 -u evaluate_checkpoints.py
   ```

3. Results are printed as a table and saved to `results/results_full_eval.json`.

## Metrics computed

| Metric | Description |
|--------|-------------|
| **Fmax** | Protein-centric max F1 over quantile threshold sweep (CAFA standard) |
| **AUPR** | Area under protein-centric precision-recall curve |
| **wFmax** | Fmax weighted by GO term information content (rare terms count more) |
| **nDCG@10** | Normalized discounted cumulative gain at rank 10 |
| **MAP** | Mean average precision over ranked GO list per protein |
| κ, seq_r, go_r | Curvature and mean Poincaré radii (hyperbolic models only) |

## To add a new checkpoint

Add an entry to the `checkpoints` list in `evaluate_checkpoints.py`:
```python
("my_model.pt", "My Model", "euclidean", 256),  # or "lorentz"
```

## Analysis figures

After evaluation, generate figures:
```
conda run --no-capture-output -n pannot-infer python3 -u analysis.py
```
Figures saved to `figures/`.
