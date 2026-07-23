# Next Steps: GO Reasoning Experiments

Updated: 2026-07-23.

Use `ANALYSIS_READY_RESULTS.md` as the baseline handoff for all follow-up
experiments. It defines the clean supervised dual-encoder baseline, the
ProtST-ESM-2 comparison rows, and the non-comparable legacy runs.

## Current Decision

The clean Grace runs show that directly placing GO items in hyperbolic space is
not the strongest predictor. The strongest current baseline is the frozen
ESM2-650M + NeuML PubMedBERT dual encoder trained with BCE and capped
`pos_weight`. Static hyperbolic v2 variants are useful diagnostics, but they
should not be the primary model unless a later experiment shows a clear gain.

The next useful question is narrower: can hyperbolic geometry help as a
reasoning or path-consistency module over the GO graph after a strong predictor
has generated candidate terms?

## Proposed Experiment

Use the clean dual encoder as the first-stage candidate generator. For each
protein, keep top-k GO terms from the BCE model, then reason over local GO graph
neighborhoods instead of scoring every GO term independently.

Test these model families:

| Track | Description | Purpose |
|---|---|---|
| Retriever only | Current BCE dual encoder | Strong non-reasoning baseline |
| Euclidean graph diffusion | Smooth/re-rank top-k terms over GO ancestors and children | Tests whether any graph reasoning helps |
| GO GNN reranker | Message passing over induced candidate subgraph | Strong Euclidean graph baseline |
| Hyperbolic path reranker | Score parent-child paths with Lorentz/geodesic transitions | Tests hyperbolic reasoning steps directly |
| LLM reranker | Prompt or lightweight adapter using GO names/definitions and candidates | Tests language-based GO reasoning |
| LLM + hyperbolic path features | LLM reranker receives path depth, IC, ancestor consistency, and hyperbolic path scores | Tests whether geometry helps the reasoning process |

## Hyperbolic Step Formulation

Represent each GO transition as a local step rather than a final static score.
For a candidate path `root -> ancestor -> ... -> term`, score:

```text
score(protein, term, path)
  = protein_term_score
  + lambda_path * sum transition_score(parent, child)
  + lambda_entail * ontology_consistency(parent, child)
  - lambda_violate * inconsistency_penalty(predicted_set)
```

The hyperbolic version should parameterize `transition_score(parent, child)` by
Lorentz distance, radial ordering, and entailment-cone satisfaction. The
Euclidean control should use the same candidate graph and training labels, but
replace Lorentz geometry with Euclidean or cosine transition features.

## Data Work

Keep the namespace-specific clean caches for ProtST comparison, then add a full
GO cache for the reasoning experiments.

Required data artifacts:

| Artifact | Status | Notes |
|---|---|---|
| `go-basic.obo` | Present | Needed for ancestors, depth, namespace, and IC |
| Clean GO-MF/BP/CC TorchDrug caches | Present | Current ProtST-style benchmark |
| Full GO vocabulary cache | Not built yet | Needed for whole-GO reasoning |
| Ancestor propagation tables | Not built yet | Needed for CAFA-consistent training/eval |
| GO path/depth/IC features | Not built yet | Needed for reranker diagnostics |

## Metrics

Report the same paper-compatible metrics for every reasoning run:

| Metric | Why |
|---|---|
| CAFA Fmax | Main benchmark metric |
| micro-AUPR and macro-AUPR | Separates global pair ranking from term-centric behavior |
| weighted Fmax and Smin | Tests high-IC/deep-term utility |
| nDCG@10 and MAP | Tests ranking quality before thresholding |
| Ontology violation rate | Counts child predicted without required ancestors |
| Depth/IC-stratified Fmax | Checks whether graph reasoning helps specific GO terms |

## Grace Execution Order

1. Build full-GO vocabulary, ancestor closure, and propagated target caches.
2. Re-evaluate the current BCE dual encoder with propagated targets to establish
   the graph-consistent baseline.
3. Run Euclidean graph diffusion on saved logits only; no GPU required.
4. Run a small MF hyperbolic path reranker smoke test on A100.
5. If MF improves Smin or high-IC/deep-term Fmax, run BP/CC and then full GO.
6. Add LLM reranking only after the graph-only baselines are strong enough to
   justify the added compute and prompt complexity.

## Go / No-Go Criterion

Continue the hyperbolic reasoning direction only if it improves at least one of
these without materially hurting CAFA Fmax:

| Criterion | Required signal |
|---|---|
| Specific-term quality | Better high-IC or deep-term Fmax |
| Ontology consistency | Lower violation rate after thresholding |
| CAFA error profile | Lower Smin |
| Ranking | Better nDCG@10 or MAP on candidate terms |

If the only outcome is similar Fmax with no Smin, depth, or consistency gain,
then hyperbolic geometry should remain a diagnostic/analysis tool rather than a
main model component.
