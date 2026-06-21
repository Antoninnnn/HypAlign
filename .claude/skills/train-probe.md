---
description: Run the v1 4-condition geometry probe experiment (frozen ESM-2, InfoNCE)
---

Run the v1 probe experiment that trains four geometry conditions on the
ProtST GO-MF dataset with frozen ESM-2 features.

## Steps

1. Navigate to the experiment directory:
   ```
   cd experiments/hyp_ssf_probe
   ```

2. Check that cache files exist:
   ```
   ls cache/esm2_go_feats.pt cache/go_term_embs.pt cache/protst_go_mf_decoded.pt
   ```
   If missing, the script will download and build them on first run (needs internet
   and ~10 min). On HPRC, rsync cache/ from local first.

3. Run training (all 4 conditions sequentially):
   ```
   conda run --no-capture-output -n pannot-infer python3 -u run_experiment_go.py
   ```
   Wall time: ~30 min on A100, ~2 h on RTX 4500.
   Checkpoints saved to `checkpoints/probe_*.pt`.
   Results saved to `results/results_go.json`.

4. After training, run evaluation for full metric suite:
   ```
   conda run --no-capture-output -n pannot-infer python3 -u evaluate_checkpoints.py
   ```
   Output: `results/results_full_eval.json` + printed table.

5. Generate analysis figures:
   ```
   conda run --no-capture-output -n pannot-infer python3 -u analysis.py
   ```
   Output: `figures/fig*.png` (7 figures).

## Key config (top of run_experiment_go.py)

- `PROJ_DIM = 256` — embedding dimension
- `EPOCHS = 60` — training epochs
- `BATCH_SIZE = 128`
- `TEMPERATURE = 0.07`
- Conditions: Euclidean / Hyp / Hyp+MERU λ=0.5 / Hyp+MERU+DAG λ=0.5

## To run only specific conditions

Pass `--cond` flag (not yet implemented; edit `CONDITIONS` list at bottom of
`run_experiment_go.py` to subset).

## Expected results (ESM-2 8M frozen)

| Model | Fmax | AUPR |
|-------|------|------|
| Euclidean | 0.075 | 0.032 |
| Hyp+MERU | 0.083 | 0.034 |
