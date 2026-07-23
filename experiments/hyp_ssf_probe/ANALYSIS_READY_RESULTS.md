# Analysis-Ready Grace Results

Updated: 2026-07-23.

This file is the researcher-facing handoff for the current HypAlign GO
experiments. Use it to decide which baseline to compare against, which rows are
paper-comparable, and which older runs should be treated as provenance only.

## Primary Baseline

Use this as the main internal baseline for future experiments:

```text
Frozen ESM2-650M protein features
+ frozen NeuML/pubmedbert-base-embeddings GO text embeddings
+ trainable MLP dual-encoder projection heads
+ BCEWithLogitsLoss(pos_weight = min(neg / pos, 100))
```

The old run label `PLIP` means this supervised protein-GO dual encoder. It does
not mean the pathology-image PLIP model.

## Analysis-Ready Subtask Results

These are from successful Grace evaluator job `19181226`.

| GO subtask | Terms | Test proteins | Fmax | AUPR | wFmax | macro-AUPR | Smin |
|---|---:|---:|---:|---:|---:|---:|---:|
| GO-MF | 489 | 3416 | 0.6343 | 0.6320 | 0.6172 | 0.6064 | 20.7825 |
| GO-BP | 1943 | 3416 | 0.4643 | 0.3234 | 0.4353 | 0.2849 | 253.7932 |
| GO-CC | 320 | 3416 | 0.5390 | 0.4214 | 0.5131 | 0.3628 | 40.4545 |

Metric definitions for this table:

| Column | Definition |
|---|---|
| Fmax | CAFA-style max F1 over thresholds |
| AUPR | micro-AUPR over all protein-GO pairs |
| wFmax | information-content weighted Fmax |
| macro-AUPR | term-centric per-GO AUPR averaged across non-empty terms |
| Smin | CAFA remaining-uncertainty / misinformation score; lower is better |

## External ProtST Baseline

Use this table for direct paper-style comparison against ProtST-ESM-2. The
external ProtST values are from the ProtST protein function classification table
for GO-BP, GO-MF, and GO-CC in Table 2 of the ProtST paper:
https://spj.science.org/doi/10.34133/hds.0211

| GO subtask | Our Fmax | ProtST-ESM-2 Fmax | Delta Fmax | Our AUPR | ProtST-ESM-2 AUPR | Delta AUPR |
|---|---:|---:|---:|---:|---:|---:|
| GO-MF | 0.6343 | 0.6680 | -0.0337 | 0.6320 | 0.6470 | -0.0150 |
| GO-BP | 0.4643 | 0.4820 | -0.0177 | 0.3234 | 0.3420 | -0.0186 |
| GO-CC | 0.5390 | 0.4870 | +0.0520 | 0.4214 | 0.3640 | +0.0574 |

Interpretation:

| Subtask | Status |
|---|---|
| GO-MF | Slightly below ProtST-ESM-2, but close on both Fmax and AUPR |
| GO-BP | Slightly below ProtST-ESM-2 |
| GO-CC | Above ProtST-ESM-2 on both Fmax and AUPR |

## Geometry Comparison Baselines

For geometry experiments, do not compare hyperbolic rows directly to the
supervised dual-encoder above. Compare each hyperbolic condition to the
Euclidean row with the same cache, feature encoder, text encoder, loss, and
training budget.

| Loss | Proper Euclidean control | Best hyperbolic variant | Conclusion |
|---|---:|---:|---|
| BCE | 0.1399 Fmax | Hyp+MERU: 0.1545 Fmax | MERU helps BCE, but absolute predictor is weak |
| MulSupCon | 0.4149 Fmax | Hyp+MERU: 0.3556 Fmax | Euclidean is stronger; static hyperbolic geometry does not win |

Research conclusion: static hyperbolic GO item embeddings are not the current
best predictive baseline. Hyperbolic structure is more plausible as a
reasoning/path-consistency module after a strong first-stage predictor.

## Source Files

| Artifact | Path |
|---|---|
| Main report | `experiments/hyp_ssf_probe/report.md` |
| CAFA evaluator | `experiments/hyp_ssf_probe/evaluate_dual_encoder_cafa.py` |
| Successful evaluator log | `experiments/hyp_ssf_probe/logs/hypalign-eval-cafa_19181226.out` |
| Analysis-ready metric JSON | `experiments/hyp_ssf_probe/results/dual_encoder_cafa_metrics.json` |
| GO-MF baseline JSON | `experiments/hyp_ssf_probe/results/plip_esm2_650m_neuml_go_mf_clean_bce_posw_cap100.json` |
| GO-BP baseline JSON | `experiments/hyp_ssf_probe/results/plip_esm2_650m_neuml_go_bp_bce_posw_cap100.json` |
| GO-CC baseline JSON | `experiments/hyp_ssf_probe/results/plip_esm2_650m_neuml_go_cc_bce_posw_cap100.json` |
| Clean v2 BCE geometry JSON | `experiments/hyp_ssf_probe/results/results_v2_bce_cleanmf.json` |
| Clean v2 MulSupCon geometry JSON | `experiments/hyp_ssf_probe/results/results_v2_msc_cleanmf.json` |

Do not use the tracked files under `experiments/hyp_ssf_probe/data/` for the
current baseline. They are legacy GO-MF prototype artifacts renamed with the
source dataset namespace:
`protst_geneontology_mf_vocab_legacy.json` and
`protst_geneontology_mf_pubmedbert_cls_embs_legacy.pt`. The analysis-ready runs
use the namespace-specific `cache/go_terms_protst_go_*_NeuML...pt` files.

## What Not To Use As Main Baseline

| Run family | Reason |
|---|---|
| `plip_esm2_650m_neuml_bce_posw_cap100` | Old ambiguous MF cache; invalidated by clean MF rerun |
| `plip_esm2_650m_neuml_bce_posw_uncapped` | Same old cache regime and weaker result |
| `results_v2.json`, `results_v2_bce.json`, `results_v2_msc.json` | Superseded by clean cache-safe v2 reruns |
| Fine-tuned ESM2-650M ASL/BCE rows | Old cache regime; keep as traceability only until rerun on clean caches |
| v1 ESM2-8M InfoNCE sections in `report.md` | Historical exploratory analysis only |

## Ready-To-Analyze Claim

The clean supervised frozen dual encoder is the current baseline to beat. It is
close to ProtST-ESM-2 on GO-MF and GO-BP and stronger on GO-CC under the
currently recorded paper-style metrics. The clean v2 geometry experiments do
not show a predictive advantage for static hyperbolic GO embeddings, so the next
scientific direction should test graph/path reasoning on top of the strong
dual-encoder scores rather than replacing the baseline with a hyperbolic
projector.
