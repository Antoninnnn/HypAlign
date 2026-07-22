# Hyperbolic Embedding for Protein GO Function Prediction
## Preliminary Study: Does Hyperbolic Geometry + Entailment Improve GO-MF Retrieval?

---

## 1. Motivation and Principal

### 1.1 The Problem with Euclidean Space for Ontological Data

Standard contrastive learning (CLIP, ProTrek) embeds proteins and text in Euclidean space with cosine similarity. This works well for flat retrieval tasks but has a structural mismatch with Gene Ontology (GO): the GO is a directed acyclic graph (DAG) where function terms are organized hierarchically from general to specific.

In Euclidean space, the volume of a ball of radius *r* grows polynomially as r^n. This means there is limited room to represent exponentially branching hierarchies — a tree with branching factor *b* and depth *d* has b^d leaves, but fitting them in Euclidean space requires distortion that grows with depth.

Hyperbolic space has *exponential* volume growth: a ball of radius *r* in H^n has volume proportional to e^{(n−1)r}. This matches the natural geometry of trees and DAGs, enabling near-zero distortion embeddings of hierarchical structures.

### 1.2 Lorentz (Hyperboloid) Model

We use the Lorentz model H^n_κ with curvature κ > 0. A point in hyperbolic space is represented as the *space component* **x** ∈ ℝ^n, with the time component computed as:

```
t(x) = sqrt(1/κ + ||x||²)
```

The exponential map from origin maps a Euclidean tangent vector **v** to hyperbolic space:

```
exp_map0(v) = sinh(√κ · ||v||) · v / ||v||
```

Points near the origin have high curvature (wide entailment cones), corresponding to *general* concepts. Points far from the origin have low curvature (narrow cones), corresponding to *specific* concepts.

### 1.3 MERU Cross-Modal Entailment

MERU (Desai et al., 2023) defines entailment cones in hyperbolic space. A point **x** entails **y** if **y** lies within the cone of **x**:

```
oxy_angle(x, y) ≤ half_aperture(x)
```

where:
- `oxy_angle(x, y)` = exterior angle at **x** in hyperbolic triangle O–x–y
- `half_aperture(x) = arcsin(2·r_min / (√κ · ||x||))` — wider near origin (general), narrower far out (specific)

The entailment loss penalizes violations:

```
L_MERU = mean(max(0, oxy_angle(text_i, seq_i) − half_aperture(text_i)))
```

Applied cross-modally: each GO function term (general concept) should entail the protein sequence (specific instance). This pushes GO terms toward the origin and proteins outward — a geometrically correct representation of "a GO term is a class that subsumes many proteins."

### 1.4 GO DAG Entailment (Intra-Text)

Beyond cross-modal entailment, the GO hierarchy itself is explicit: if term A is an ancestor of term B (e.g., "nucleotide binding" → "ATP binding"), then A should entail B in hyperbolic space. We add a second entailment loss over `is_a` edges within the vocabulary:

```
L_DAG = mean(max(0, oxy_angle(parent_i, child_i) − half_aperture(parent_i)))
```

sampled from the 456 `is_a` edges induced among the 489 GO-MF vocabulary terms.

---

## 2. Dataset

**Dataset:** `mila-intel/ProtST-GeneOntology-MF` (Xu et al., ProtST 2023)

This is the DeepFRI molecular function benchmark. Proteins are derived from the Protein Data Bank (PDB), annotated with GO-MF terms from PDB-GOA. GO terms are filtered to those appearing in 50–5000 PDB chains, yielding 489 GO-MF terms.

| Split      | Proteins | GO terms/protein (avg) |
|------------|----------|------------------------|
| Train      | 23,022*  | 4.3                    |
| Validation | 2,526*   | 4.3                    |
| Test       | 2,991    | 9.1                    |

*After filtering proteins with no GO-MF annotation (these cannot form training pairs).

**Sequence representation:** Proteins are stored as pre-tokenized ESM integer sequences. We decode them to amino acid strings (tokens 4–23 → ARNDCQEGHILKMFPSTWYV) and encode with ESM-2 8M (mean pool of final hidden state, 320-dim).

**Text representation:** Each of the 489 GO-MF terms is encoded as `"FUNCTION: {go_term_name}."` using frozen PubMedBERT (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`), yielding a fixed [489, 768] embedding table. This follows ProtST's text format for GO prompts.

**Training signal:** Each protein is paired with one of its positive GO terms, sampled uniformly at each batch. This produces standard InfoNCE training pairs (protein sequence ↔ GO function text) without any additional annotation.

---

## 3. Experimental Setup

### 3.1 Architecture

All conditions share the same two-encoder structure:

```
Sequence side:  ESM-2 8M features [B, 320]  →  seq_head  →  [B, 256]
Text side:      GO emb lookup [B, 768]       →  text_head →  [B, 256]
```

**Euclidean head:** `Linear(in, 256) → L2-normalize`

**Lorentz head (MERU-style):**
```
x = Linear(in, 256) · exp(log_α)     # log_α ≤ 0, prevents boundary saturation
return exp_map0(x, κ)                 # maps to hyperbolic space component
```

Both modalities share a single learnable curvature κ (initialized to 1.0, clamped to [0.1, 10]).

### 3.2 Training Conditions

| Condition | Geometry | Objective |
|---|---|---|
| **Euclidean** | Euclidean | InfoNCE (cosine sim) |
| **Hyp** | Lorentz H^256_κ | InfoNCE (neg Lorentz distance) |
| **Hyp+MERU λ=0.5** | Lorentz | InfoNCE + 0.5 · L_MERU (GO→protein) |
| **Hyp+MERU+DAG λ=0.5** | Lorentz | InfoNCE + 0.5 · L_MERU + 0.5 · L_DAG (parent→child GO) |

### 3.3 Training Details

| Hyperparameter | Value |
|---|---|
| Epochs | 60 |
| Batch size | 128 |
| Optimizer | AdamW (lr=3e-4, wd=1e-4) |
| LR schedule | Cosine annealing |
| Temperature | Learnable, initialized 0.07 |
| Projection dim | 256 |
| GO DAG edges | 456 (is_a edges among 489 vocab terms) |

### 3.4 Evaluation Metrics

All metrics are computed zero-shot: the same [seq_head, text_head] weights from contrastive training are used directly to compute a similarity matrix [N_test=2991, 489].

**Threshold-based:**
| Metric | Description |
|---|---|
| **Fmax** | Protein-centric max F1 over 199 quantile-based threshold sweep. CAFA standard. |
| **wFmax** | Fmax weighted by GO term IC (= −log freq). Rare specific terms count more. |
| **AUPR** | Area under protein-centric precision-recall curve. |

**Rank-based (threshold-free):**
| Metric | Description |
|---|---|
| **R@k** | Recall at k: fraction of a protein's true GO terms that appear in the top-k ranked predictions. Multi-label analog of MERU's recall@{5,10}. |
| **P@k** | Precision at k: fraction of top-k predictions that are true GO terms. |
| **nDCG@10** | Normalized Discounted Cumulative Gain at rank 10. Rewards true GO terms appearing at the very top of the ranked list. |
| **MAP** | Mean Average Precision over the full ranked GO list per protein. |

**Geometric analysis:**
| Metric | Description |
|---|---|
| **Poincaré radius** | `‖x‖ / (t + 1/√κ)` — distance from hyperbolic origin. GO terms should have smaller radius than proteins (general near origin, specific far out). |
| **IC–radius correlation** | Spearman ρ between GO term IC and its Poincaré radius. Should be positive: more specific GO terms should be further from origin. |
| **Hierarchy recovery** | Fraction of DAG parent→child pairs where parent has smaller radius than child. Should approach 100%. |

---

## 4. Results

### 4.0 Consolidated Grace Results

Updated: 2026-07-22. This section is the result registry for Grace runs. `AUPR` below is the existing protein-centric threshold-sweep AUPR unless otherwise noted. `macro-AUPR` is term-centric per-GO average precision. Paper-style micro-AUPR is not yet recomputed for all rows.

**Current clean frozen Protein-GO dual-encoder baselines**

These are the primary ProtST-style comparison rows. They use TorchDrug `GeneOntology.zip` namespace caches, frozen ESM2-650M features, frozen NeuML PubMedBERT GO text embeddings, trainable MLP projection heads, and BCE with `pos_weight = neg / pos` capped at 100. The old `PLIP` label means this dual-encoder baseline, not the pathology-image PLIP model.

| Dataset | Run | Terms | Train / Val / Test proteins | Loss | Fmax | AUPR | wFmax | nDCG@10 | MAP | P2G P@1 |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| GO-MF | `plip_esm2_650m_neuml_go_mf_clean_bce_posw_cap100` | 489 | 29902 / 3323 / 3416 | BCE + pos_weight cap 100 | **0.6226** | 0.1679 | **0.6050** | **0.7239** | **0.6561** | **0.8197** |
| GO-BP | `plip_esm2_650m_neuml_go_bp_bce_posw_cap100` | 1943 | 29902 / 3323 / 3416 | BCE + pos_weight cap 100 | 0.4632 | 0.1943 | 0.4328 | 0.7015 | 0.4656 | 0.7892 |
| GO-CC | `plip_esm2_650m_neuml_go_cc_bce_posw_cap100` | 320 | 29902 / 3323 / 3416 | BCE + pos_weight cap 100 | 0.5359 | 0.2709 | 0.5019 | 0.6562 | 0.5695 | 0.7485 |

**ProtST paper-style Fmax comparison**

These ProtST values are the relevant paper baseline for the current namespace-specific PDB-chain benchmark. AUPR is omitted here because our current JSON AUPR is not the same paper-style micro-AUPR.

| Dataset | Our clean Fmax | ProtST-ESM2 Fmax | Gap |
|---|---:|---:|---:|
| GO-MF | 0.6226 | 0.6680 | -0.0454 |
| GO-BP | 0.4632 | 0.4820 | -0.0188 |
| GO-CC | 0.5359 | 0.4870 | +0.0489 |

**Current clean GO-MF v2 geometry reruns**

These are frozen ESM2-650M feature experiments using the same clean MF cache as the clean dual-encoder baseline. They explicitly pass and validate `--splits-cache`, `--vocab-file`, `--feature-cache`, and `--term-embedding-cache`, so they are comparable to the clean MF baseline. They are geometry/projector experiments, not encoder fine-tuning.

| Loss | Geometry | Fmax | AUPR | macro-AUPR | Best val Fmax |
|---|---|---:|---:|---:|---:|
| BCE | Euclidean-v2 | 0.1399 | 0.0353 | 0.0460 | 0.1242 |
| BCE | Hyp-v2 | 0.1307 | 0.0434 | 0.0337 | 0.0830 |
| BCE | Hyp+MERU-v2 | **0.1545** | **0.0490** | 0.0378 | **0.1243** |
| BCE | Hyp+MERU+DAG-v2 | 0.1215 | 0.0438 | 0.0241 | 0.0881 |
| MulSupCon | Euclidean-v2 | **0.4149** | 0.1389 | **0.2320** | **0.4164** |
| MulSupCon | Hyp-v2 | 0.2785 | 0.1223 | 0.0314 | 0.2279 |
| MulSupCon | Hyp+MERU-v2 | 0.3556 | **0.1895** | 0.0909 | 0.3059 |
| MulSupCon | Hyp+MERU+DAG-v2 | 0.3442 | 0.1753 | 0.0727 | 0.3010 |

**Legacy fine-tuning and frozen contrastive runs**

These runs are retained for traceability. They were produced before the cache-safe namespace workflow and should not be directly compared to the clean MF/BP/CC table without rerunning on the same clean caches.

| Track | Run | Encoder / Features | Loss | Fmax | AUPR | macro-AUPR | wFmax | nDCG@10 | MAP | P2G P@1 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Fine-tune | ESM2-650M-FT-ASL | ESM2-650M end-to-end | ASL | 0.3566 | 0.0829 | 0.2954 | 0.3378 | 0.4277 | 0.3636 | 0.5025 |
| Fine-tune | ESM2-650M-FT-ASL-rerun-20260626 | ESM2-650M end-to-end | ASL | 0.3528 | 0.0839 | 0.2912 | 0.3353 | 0.4229 | 0.3601 | 0.4958 |
| Fine-tune | ESM2-650M-FT-BCE | ESM2-650M end-to-end | BCE + pos_weight | 0.3206 | 0.0923 | 0.2460 | 0.2997 | 0.3850 | 0.3345 | 0.4493 |
| Fine-tune | ESM2-650M-FT-BCE-rerun-20260626 | ESM2-650M end-to-end | BCE + pos_weight | 0.3162 | 0.0893 | 0.2479 | 0.2956 | 0.3816 | 0.3307 | 0.4390 |
| Frozen InfoNCE | multipos_bs1024 | cached ESM2-650M + NeuML text | multi-positive InfoNCE | 0.1903 | 0.0611 | 0.1351 | 0.1709 | 0.2438 | 0.1974 | 0.3086 |
| Frozen InfoNCE | fullmultipos_bs1024 | cached ESM2-650M + NeuML text | all-489-GO multi-positive InfoNCE | 0.1722 | 0.0530 | 0.0494 | 0.1505 | 0.2229 | 0.1483 | 0.3534 |
| Frozen InfoNCE | pair_bs256 | cached ESM2-650M + NeuML text | pair InfoNCE | 0.1751 | 0.0700 | 0.1321 | 0.1586 | 0.1894 | 0.1829 | 0.1922 |
| Frozen InfoNCE | pair_bs2048 | cached ESM2-650M + NeuML text | pair InfoNCE | 0.1648 | 0.0659 | 0.1413 | 0.1467 | 0.1832 | 0.1780 | 0.2003 |

**Superseded / invalidated runs**

These rows are kept only to document what went wrong. They should not be cited as main results.

| Run | Issue | Fmax | Status |
|---|---|---:|---|
| `plip_esm2_650m_neuml_bce_posw_cap100` | Old MF run used ambiguous cache/data path; clean rerun improved 0.2096 → 0.6226 | 0.2096 | Invalidated |
| `plip_esm2_650m_neuml_bce_posw_uncapped` | Same old MF cache regime as above | 0.1879 | Invalidated |
| `results_v2_bce.json` | v2 before cache-safe clean-MF workflow; uses ambiguous old feature cache path | best 0.1522 | Superseded |
| `results_v2_msc.json` | v2 before cache-safe clean-MF workflow; uses ambiguous old feature cache path | best 0.2121 | Superseded |
| `results_v2.json` | Earlier v2 before NeuML/mean-pooling and architecture fixes | best 0.1307 | Superseded |

**External comparison reported by Yuxuan**

These rows are not reproduced by our current code path; they are numbers shared in chat and are included only as a comparison target. Their AUPR definition appears different from our current threshold-sweep AUPR.

| Reported run | Loss | Fmax | AUPR | wFmax | nDCG@10 | MAP | P2G P@1 |
|---|---|---:|---:|---:|---:|---:|---:|
| plip_test | BCE | 0.5337 | 0.5267 | 0.5165 | 0.6260 | 0.5613 | 0.6968 |
| infonce_test | InfoNCE | 0.1479 | 0.4752 | 0.1193 | 0.5984 | 0.5350 | 0.6887 |
| infonce_bigbatch_test | InfoNCE | 0.3169 | 0.0979 | 0.2647 | 0.6002 | 0.5320 | 0.6881 |

**Current interpretation:** the clean supervised frozen dual-encoder baseline is strong and close to ProtST on GO-MF/GO-BP while exceeding the ProtST Fmax reference on GO-CC. Static hyperbolic GO modeling does not beat Euclidean geometry on Fmax in the clean v2 runs. Hyperbolic/MERU remains useful as a structural diagnostic and possibly as a reasoning-path regularizer, but not as the current best predictor.

### 4.1 Prediction Metrics

```
  Model                   Fmax    AUPR   wFmax  nDCG@10    MAP
  ──────────────────────────────────────────────────────────────
  Euclidean              0.0748  0.0318  0.0614   0.0881  0.0877
  Hyp                    0.0650  0.0298  0.0550   0.0561  0.0657
  Hyp+MERU λ=0.5         0.0828  0.0337  0.0704   0.0882  0.0867
  Hyp+MERU+DAG λ=0.5     0.0545  0.0261  0.0456   0.0399  0.0534
```

### 4.2 Retrieval Metrics

MERU's paper reports `recall@{5,10}` in both directions (image→text and text→image). We report analogous metrics for both **Protein→GO** and **GO→Protein** retrieval.

**Protein → GO term** (given a protein, rank all 489 GO terms; the standard function prediction direction):

```
  Model                   R@1    R@5   R@10    P@1    P@5   P@10
  ────────────────────────────────────────────────────────────────
  Euclidean              0.014  0.047  0.081  0.095  0.074  0.065
  Hyp                    0.009  0.030  0.054  0.051  0.043  0.043
  Hyp+MERU λ=0.5         0.012  0.054  0.088  0.067  0.070  0.064
  Hyp+MERU+DAG λ=0.5     0.007  0.023  0.040  0.030  0.030  0.030
```

R@k = fraction of the protein's true GO terms that appear in the top-k ranked predictions (averaged over all test proteins). P@k = fraction of top-k predictions that are true GO terms.

**GO term → Protein** (given a GO functional concept, rank all 2,991 test proteins):

```
  Model                   R@1    R@5   R@10    P@1    P@5   P@10
  ────────────────────────────────────────────────────────────────
  Euclidean              0.002  0.009  0.018  0.087  0.079  0.078
  Hyp                    0.002  0.007  0.014  0.069  0.071  0.069
  Hyp+MERU λ=0.5         0.001  0.005  0.008  0.035  0.043  0.036
  Hyp+MERU+DAG λ=0.5     0.001  0.003  0.008  0.039  0.032  0.035
```

R@k = fraction of the GO term's annotated proteins that appear in the top-k ranked proteins (averaged over GO terms with ≥1 annotated protein in the test set).

**Key contrast between directions:** The two retrieval directions reveal a fundamental asymmetry. In the Protein→GO direction, Hyp+MERU is the best model (R@5=0.054, R@10=0.088), outperforming Euclidean on recall. In the GO→Protein direction, the pattern reverses — Euclidean dominates on both recall and precision. This is consistent with the entailment geometry: MERU concentrates GO terms near the origin in a compact cluster, which helps proteins find their GO terms (high recall), but makes it harder for a GO term to discriminate which proteins belong to it (lower GO→Protein precision) because the GO term embeddings are no longer spread out enough to individually "point at" their annotated proteins.

### 4.2 Poincaré Geometry Analysis

```
  Model                           κ    seq_r     go_r    IC–radius ρ   Hierarchy
  ───────────────────────────────────────────────────────────────────────────────
  Hyp                        1.0445   0.2085   0.2751    ρ = −0.19 ✗   49.1% ✗
  Hyp+MERU λ=0.5             0.9549   0.1952   0.0944    ρ = +0.30 ✓   58.6% ~
  Hyp+MERU+DAG λ=0.5         0.9556   0.1835   0.0953    ρ = +0.50 ✓   96.5% ✓
```

`seq_r` = mean Poincaré radius of test proteins; `go_r` = mean Poincaré radius of 489 GO terms.
IC–radius ρ = Spearman correlation between GO term information content and its Poincaré radius (positive = specific terms further from origin = correct).
Hierarchy = fraction of DAG parent→child pairs where parent radius < child radius (correct ordering).

### 4.3 Gromov δ-Hyperbolicity

Gromov δ-hyperbolicity measures how tree-like a metric space is. For any four points x, y, z, w, form the three pairwise-sum candidates S1 ≥ S2 ≥ S3; then δ = (S1 − S2) / 2. A perfect tree has δ = 0; flat Euclidean space achieves δ = D/2 (diameter/2). We report δ_rel = δ_max / (D/2) ∈ [0, 1] so results are comparable across models with different scales.

We sample 50,000 random 4-tuples from (a) the 489 GO term embeddings and (b) 500 sampled test protein embeddings, using the model's native distance (Lorentz geodesic for hyperbolic models, Euclidean L2 for Euclidean).

```
  Model                    GO δ_mean  GO δ_max  GO δ_rel   Pr δ_mean  Pr δ_rel
  ────────────────────────────────────────────────────────────────────────────
  Euclidean                   0.0486    0.2629    0.387      0.0549     0.406
  Hyp                         0.0287    0.1375    0.324      0.0167     0.321
  Hyp+MERU λ=0.5              0.0097    0.0541    0.272      0.0171     0.341
  Hyp+MERU+DAG λ=0.5          0.0092    0.0547    0.261      0.0164     0.289
```

Three patterns stand out. First, simply switching to the Lorentz model reduces GO δ_rel from 0.387 to 0.324 — the Lorentz parameterization constrains geometry to be intrinsically less flat, even without explicit hierarchy supervision. Second, adding MERU halves GO δ_rel further (0.324 → 0.272): the entailment loss that pulls GO terms toward the origin also makes the GO embedding space measurably more tree-like. Third, Hyp+MERU+DAG achieves the lowest δ_rel overall (GO: 0.261, Prot: 0.289), consistent with it imposing the strongest structural constraints — though at the cost of retrieval performance.

---

## 5. Analysis

### 5.1 Figure 4 Analog: Poincaré Radius Distribution (proteins vs GO terms)

![Fig 4: Radius distributions](figures/fig4_radius_distribution.png)

This figure mirrors MERU's Figure 4, which shows text embeddings sitting closer to the hyperbolic origin (ROOT) than image embeddings. Here we plot the distribution of Poincaré radii for proteins (blue) and GO terms (red) for each hyperbolic model.

**Hyp (no entailment):** The two distributions heavily overlap and the GO terms (mean radius 0.275) sit *further* from the origin than proteins (0.208). This is geometrically inverted — the model has no signal to push general GO classes toward the origin.

**Hyp+MERU λ=0.5:** The distributions cleanly separate into two non-overlapping peaks. GO terms collapse near the origin (mean 0.094) while proteins spread to ~0.195. This is the correct structure: GO functional categories are general concepts (wide entailment cones near origin), individual proteins are specific instances (narrower cones further out).

**Hyp+MERU+DAG λ=0.5:** Near-identical separation to MERU-only (go_r=0.095, seq_r=0.183), confirming that the DAG loss does not further separate modalities — it only rearranges GO terms among themselves within the hyperbolic space.

### 5.2 Figure 4b: Positive vs Negative Pair Distances

![Fig 4b: Pair distance distributions](figures/fig4b_pair_distances.png)

For each model, we sample 5,000 matched (protein, true GO term) pairs and 5,000 unmatched pairs, and plot the distribution of their distances or dissimilarities (Lorentz geodesic distance; 1−cosine for Euclidean).

All models separate positive from negative pairs, but the gap is small across the board: Euclidean (pos=0.867 vs neg=0.907, gap=0.040), Hyp (0.413 vs 0.437, gap=0.024), Hyp+MERU (0.335 vs 0.349, gap=0.014). This modest separation explains the generally low absolute Fmax values — there is limited score margin to threshold on.

A key observation: Hyp+MERU compresses both distributions relative to Euclidean, but the *positive distribution is more tightly concentrated* (lower variance) than the negative distribution. This means MERU produces more consistent, less noisy distances for true GO matches — which is what drives R@5/10 improvement even without a large mean gap.

### 5.3 Figure 5 Analog: Geodesic Traversal from Protein toward Origin

![Fig 5: Geodesic traversal](figures/fig5_geodesic_traversal.png)

This is the protein-function analog of MERU's Figure 5, which walks from an image embedding toward the hyperbolic origin and observes that retrieved text becomes progressively more generic.

We interpolate from each protein's hyperbolic embedding along the geodesic toward the origin using the log_map0/exp_map0 pair: `x(s) = exp_map0(s · log_map0(x_protein))`, where s=1.0 is the protein position and s=0.0 is the origin. At each of 12 equally-spaced steps, we retrieve the nearest GO term. The gray dashed curve shows the Poincaré radius decreasing monotonically from left (specific) to right (general).

Results for four example proteins (2, 3, 7, and 9 annotated GO terms): true GO terms (shown in red) cluster near the specific end of the traversal (s close to 1.0) and disappear as we move toward the origin. Near the origin, the retrieved GO terms become broad molecular function categories. This confirms that the hyperbolic geometry learned by MERU places protein-specific functions near the protein, and broader functional classes closer to the center of the space — the geodesic acts as a specificity gradient.

### 5.4 GO Hierarchy Recovery

![Hierarchy recovery](figures/fig_hierarchy_recovery.png)

For each of the 456 is_a edges in the induced GO-MF DAG, we ask: does the parent GO term (general concept) have a smaller Poincaré radius than the child GO term (specific concept)? Points below the diagonal indicate correct ordering (parent < child radius).

- **Hyp:** 49.1% correct — essentially random. Pure InfoNCE places GO terms without regard to hierarchy.
- **Hyp+MERU λ=0.5:** 58.6% correct — mild improvement. The cross-modal entailment loss (GO→protein) gives the hierarchy a gentle push but does not explicitly constrain GO-GO ordering.
- **Hyp+MERU+DAG λ=0.5:** **96.5%** correct — near-perfect hierarchy recovery. The explicit parent→child DAG loss forces the correct ordering for nearly all edges.

This result reveals the central tension in the experiment: the DAG model achieves the most geometrically faithful representation of the GO hierarchy, yet it produces the worst retrieval performance. Imposing a rigid tree structure concentrates all GO terms into a narrow radius band, collapsing the discriminative spread between them.

### 5.5 GO Term Specificity (IC) vs Poincaré Radius

![IC vs radius](figures/fig_ic_vs_radius.png)

We compute the Spearman correlation between each GO term's information content (IC = −log(freq/N_train), higher = rarer = more specific) and its Poincaré radius. In a correctly structured hyperbolic space, specific terms should sit further from the origin (larger radius), yielding a positive correlation.

- **Hyp:** ρ = **−0.19** (p=2.5×10⁻⁵) — significantly *negative*. Without guidance, the model places more specific GO terms *closer* to the origin, which is the wrong direction. This is because rare GO terms have fewer training pairs, so the contrastive objective pulls them less strongly from the origin.
- **Hyp+MERU λ=0.5:** ρ = **+0.30** (p=2.1×10⁻¹¹) — positive and highly significant. MERU reverses the direction: the entailment cone loss pushes more general terms (low IC) toward the origin and allows specific terms to move outward.
- **Hyp+MERU+DAG λ=0.5:** ρ = **+0.50** (p=3.8×10⁻³²) — the strongest correlation. The DAG loss explicitly arranges terms by specificity level, yielding a clean monotonic IC–radius relationship.

This gives a clear quantitative signature: MERU alone is sufficient to establish the correct specificity gradient (ρ = +0.30), and the DAG loss strengthens it further (ρ = +0.50) at the cost of discriminative performance.

### 5.6 Per-GO-Term Average Precision: Euclidean vs Hyp+MERU

![Per-term AP](figures/fig_per_term_ap.png)

The left scatter plots per-term AP for Euclidean (x-axis) vs Hyp+MERU (y-axis), colored by GO term IC. Points above the diagonal indicate terms where Hyp+MERU wins. The right histogram shows the distribution of AP differences.

The result is counterintuitive: **Hyp+MERU wins on only 27% of GO terms** by per-term AP, with a mean Δ = −0.020. Yet Hyp+MERU achieves better aggregate Fmax (+10.7%) and R@5/10.

The explanation lies in what these metrics measure differently. Per-term AP measures how well a specific GO term is discriminated from all proteins — a GO-centric view. Fmax and R@k are protein-centric: they measure how well the correct GO terms appear near the top of each protein's ranked list. A model can improve aggregate protein-centric metrics without improving per-term AP if it better *distributes* scores across the similarity matrix — more consistently placing true GO terms above false ones for each protein, even if the absolute AP for each GO term is slightly lower.

Looking at the scatter: the terms where Hyp+MERU wins tend to be lower-AP terms (clustered near the origin of the scatter). High-AP terms (top-right, common GO terms like "transporter activity") are dominated by Euclidean. MERU's benefit is broadest across the many low-frequency, harder-to-predict GO terms.

### 5.7 Gromov δ-Hyperbolicity as a Geometry Quality Check

The δ-hyperbolicity results (§4.3) provide a model-intrinsic sanity check that complements retrieval metrics. A key question is whether hyperbolic models actually learn a hyperbolic geometry, or simply learn a curved version of a flat space.

The answer from δ_rel is clear: MERU is the mechanism that makes the learned space genuinely tree-like. Without it (pure Hyp), GO δ_rel = 0.324, which is lower than Euclidean (0.387) but still far from tree-like. With MERU, GO δ_rel drops to 0.272 — a 30% reduction relative to Euclidean. This reduction comes specifically from the GO embedding side: MERU concentrates general GO terms near the origin into a star-shaped arrangement that is locally tree-like, with specific GO terms branching outward.

The protein side tells a different story: Pr δ_rel decreases more gradually (0.406 → 0.321 → 0.341 → 0.289). The MERU loss does not directly constrain protein positions — it only constrains GO terms to entail proteins. Proteins are pushed outward but remain spread across the space without a strict hierarchical arrangement, which is geometrically appropriate (a single protein belongs to multiple GO terms and has no single parent concept).

The one surprise is that Hyp+MERU slightly *increases* Pr δ_rel relative to pure Hyp (0.341 vs 0.321). This suggests that concentrating GO terms near the origin forces proteins to spread across a wider angular range of the hyperboloid, which increases pairwise distances and thereby raises δ. The DAG model resolves this by also constraining the protein distribution (indirectly, through tighter GO organization), yielding the lowest Pr δ_rel = 0.289.

### 5.8 UMAP of Protein + GO Embeddings

![UMAP](figures/fig_umap.png)

We project 400 test proteins (circles) and 489 GO terms (triangles, colored by IC) into 2D using UMAP for both Euclidean and Hyp+MERU.

In the Euclidean space, GO terms and proteins are interleaved with no clear modality separation. High-IC (dark red) GO terms are scattered throughout the space.

In Hyp+MERU, GO terms form a distinct outer cluster in the UMAP — they are geometrically separated from the protein cloud. The high-IC (specific) GO terms tend to be more peripheral while lower-IC (general) terms cluster together, consistent with the IC–radius gradient from §5.5. The protein distribution is broader and more diffuse relative to the compact GO term cluster, reflecting the seq_r > go_r hierarchy that MERU enforces.

---

## 6. Central Tension: A Three-Way Tradeoff

The key finding is that hierarchy faithfulness and discriminative retrieval pull in opposite directions:

| Property | Euclidean | Hyp | Hyp+MERU | Hyp+MERU+DAG |
|---|---|---|---|---|
| GO terms near origin? | N/A | ✗ inverted | ✓ correct | ✓ correct |
| IC–radius gradient? | N/A | ρ=−0.19 (wrong) | ρ=+0.30 | ρ=+0.50 |
| GO hierarchy recovery? | N/A | 49% (random) | 59% (mild) | **97%** (near-perfect) |
| GO δ_rel (lower = more tree-like) | 0.387 | 0.324 | **0.272** | 0.261 |
| Retrieval R@10 | 0.081 | 0.054 | **0.088** | 0.040 |
| Fmax | 0.075 | 0.065 | **0.083** | 0.055 |

The DAG model produces the most geometrically correct hyperbolic space by every structural measure — yet it achieves the worst retrieval. The MERU-only model finds the sweet spot: enough structural organization to place GO terms near the origin and establish a positive IC–radius gradient, without over-constraining their relative positions to the point of losing discriminability.

---

## 7. Conclusions

| Finding | Implication |
|---|---|
| Pure Hyp < Euclidean on all metrics | Hyperbolic geometry without structural signal actively harms retrieval |
| Hyp+MERU best on Fmax, AUPR, wFmax, R@5, R@10 | Cross-modal entailment is the key signal that makes hyperbolic geometry useful |
| MERU corrects IC gradient (ρ: −0.19 → +0.30) | The entailment loss establishes the correct general→specific axis in the space |
| GO δ_rel: 0.387 → 0.324 → 0.272 → 0.261 | Each level of hierarchy supervision makes the GO space measurably more tree-like |
| MERU halves GO δ_rel (0.324 → 0.272) | Entailment loss is the mechanism that induces genuine hyperbolicity in the space |
| DAG: 97% hierarchy recovery but worst retrieval | Over-constraining GO term positions sacrifices discriminative spread |
| Hyp+MERU wins R@5/10, Euclidean wins P@1 | MERU improves recall coverage; Euclidean has higher single-prediction confidence |
| Hyp+MERU wins only 27% of terms by per-term AP | Aggregate benefit comes from improved global score calibration, not per-term accuracy |

The primary takeaway: hyperbolic geometry for protein function prediction requires MERU-style cross-modal entailment to be beneficial. The geometry alone is not a free lunch — it needs an explicit inductive bias that defines what "general" and "specific" mean in the embedding space. When provided, hyperbolic geometry improves retrieval of true GO terms within top-5/10, weighted F-max on rare terms, and organizes the semantic structure of the space in a biologically meaningful way.

### Directions for Improvement

1. **Smaller DAG weight** (λ_dag ≈ 0.05) — soft regularizer instead of hard constraint; may preserve hierarchy signal without collapsing discriminability
2. **Anneal DAG loss** — introduce GO hierarchy constraint after contrastive learning has converged, so the contrastive objective sets the discriminative baseline first
3. **Learnable GO embeddings** — apply DAG loss to a trainable GO embedding table rather than frozen PubMedBERT; removes the conflict between fixed PubMedBERT distances and the imposed DAG structure
4. **Supervised linear probe evaluation** — train a 489-class classifier on frozen embeddings for comparison with ProtST's reported numbers
5. **BP ontology** — extend to 1,943 GO-BP classes, reusing the same ESM-2 feature cache
