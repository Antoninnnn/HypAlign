#!/usr/bin/env python3
"""
Comprehensive analysis of hyperbolic vs Euclidean GO-MF probes.

Analyses (matching MERU paper + additional):
  A. Retrieval metrics  — R@1/5/10, P@1/5/10 (multi-label GO retrieval)
  B. Distance distributions (Fig 4 analog)
       B1. Poincaré radius: proteins vs GO terms per model
       B2. Positive vs negative pair distances per model
  C. Geodesic traversal (Fig 5 analog)
       Walk from a protein toward origin; retrieve nearest GO at each step
  D. GO hierarchy recovery
       Do parent GO terms sit closer to origin than their children?
  E. GO term IC vs Poincaré radius scatter
  F. Per-GO-term AP: Euclidean vs Hyp+MERU scatter
  G. 2D UMAP of protein + GO embeddings (colored by GO family)
  H. Positive pair distance improvement: Hyp+MERU vs Euclidean

Run:
    conda activate pannot-infer
    cd /home/yining_yang/NLP/hyperbolic/experiments/hyp_ssf_probe
    python analysis.py
"""

from __future__ import annotations
import json, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

REPO_ROOT   = Path(__file__).resolve().parent
CACHE_DIR   = REPO_ROOT / "cache"
CKPT_DIR    = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"
FIG_DIR     = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

EPS = 1e-8
PALETTE = {
    "Euclidean":           "#4878CF",
    "Hyp":                 "#6ACC65",
    "Hyp+MERU λ=0.5":     "#D65F5F",
    "Hyp+MERU+DAG λ=0.5": "#B47CC7",
}

# ── Lorentz math ──────────────────────────────────────────────────────────────

def _time(x, curv):
    return torch.sqrt(1.0 / curv + (x * x).sum(-1))

def exp_map0(x, curv):
    sqrt_c = curv.sqrt()
    xn = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    return torch.sinh((sqrt_c * xn).clamp(max=math.asinh(2.0**15))) * x / xn

def log_map0(x, curv):
    """Inverse of exp_map0: hyperbolic space component → tangent vector at origin."""
    sqrt_c = curv.sqrt()
    xn = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    return torch.asinh(sqrt_c * xn) / (sqrt_c * xn + EPS) * x

def pairwise_dist(x, y, curv):
    xt = _time(x, curv).unsqueeze(1)
    yt = _time(y, curv).unsqueeze(0)
    inner = x @ y.T - xt * yt
    return torch.acosh((-curv * inner).clamp(min=1.0 + EPS)) / curv.sqrt()

def poincare_radius(x, curv):
    t = _time(x, curv)
    return x.norm(dim=-1) / (t + 1.0 / curv.sqrt())

# ── Model ─────────────────────────────────────────────────────────────────────

class LorentzHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(1.0 / math.sqrt(out_dim))))
    def forward(self, z, curv):
        return exp_map0(self.proj(z) * self.log_alpha.exp(), curv)

class EuclideanHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
    def forward(self, z):
        return F.normalize(self.proj(z), dim=-1)

class GOProbe(nn.Module):
    def __init__(self, seq_dim, text_dim, proj_dim, geometry, learn_curv=True):
        super().__init__()
        self.geometry = geometry
        if geometry == "lorentz":
            self.log_curv = nn.Parameter(torch.tensor(0.0)) if learn_curv \
                else self.register_buffer("log_curv", torch.tensor(0.0))
            self.seq_head  = LorentzHead(seq_dim,  proj_dim)
            self.text_head = LorentzHead(text_dim, proj_dim)
        else:
            self.seq_head  = EuclideanHead(seq_dim,  proj_dim)
            self.text_head = EuclideanHead(text_dim, proj_dim)
        self.log_temp = nn.Parameter(torch.tensor(math.log(0.07)))

    @property
    def curv(self):
        return self.log_curv.exp()

    def encode_seq(self, x):
        return self.seq_head(x, self.curv) if self.geometry == "lorentz" else self.seq_head(x)

    def encode_text(self, x):
        return self.text_head(x, self.curv) if self.geometry == "lorentz" else self.text_head(x)

# ── Data loading ──────────────────────────────────────────────────────────────

def load_all(device):
    splits   = torch.load(CACHE_DIR / "protst_go_mf_decoded.pt", map_location="cpu", weights_only=False)
    esm_feat = torch.load(CACHE_DIR / "esm2_go_feats.pt",        map_location="cpu", weights_only=True)
    go_embs  = torch.load(CACHE_DIR / "go_term_embs.pt",         map_location="cpu", weights_only=True)
    vocab    = json.load(open(CACHE_DIR / "go_mf_vocab.json"))

    seq_test = esm_feat["test"]
    tgt_test = splits["test"]["targets"]  # [2991, 489]

    # GO term IC from training labels
    tgt_train  = splits["train"]["targets"]
    term_freq  = (tgt_train > 0.5).float().sum(0)
    ic         = -torch.log((term_freq / len(tgt_train)).clamp(min=1e-6))  # [489]

    # GO DAG edges
    obo_path = CACHE_DIR / "go-basic.obo"
    id_to_idx = vocab["id_to_idx"]
    dag_edges = []
    if obo_path.exists():
        current_id, in_term = None, False
        with open(obo_path) as fh:
            for line in fh:
                line = line.strip()
                if line == "[Term]":
                    in_term, current_id = True, None
                elif line.startswith("id:") and in_term:
                    current_id = line.split("id:", 1)[1].strip().split()[0]
                elif line.startswith("is_a:") and in_term and current_id:
                    pid = line.split("is_a:", 1)[1].strip().split()[0]
                    if pid in id_to_idx and current_id in id_to_idx:
                        dag_edges.append((id_to_idx[pid], id_to_idx[current_id]))
                elif line == "" and in_term:
                    in_term = False

    return seq_test, tgt_test, go_embs, ic, vocab, dag_edges


def load_probe(fname, geometry, proj_dim, device):
    ckpt  = torch.load(CKPT_DIR / fname, map_location="cpu", weights_only=False)
    probe = GOProbe(320, 768, proj_dim, geometry, learn_curv=True)
    probe.load_state_dict(ckpt["state_dict"])
    return probe.to(device).eval()


@torch.no_grad()
def get_embeddings(probe, seq_test, go_embs, device):
    """Returns (seq_embs [N,D], go_embs_proj [489,D]) in embedding space."""
    go_r = probe.encode_text(go_embs.to(device)).cpu()
    rows = []
    for i in range(0, len(seq_test), 256):
        rows.append(probe.encode_seq(seq_test[i:i+256].to(device)).cpu())
    return torch.cat(rows), go_r


@torch.no_grad()
def get_sim(seq_r, go_r, probe):
    if probe.geometry == "lorentz":
        return -pairwise_dist(seq_r, go_r, probe.curv.detach().cpu())
    return seq_r @ go_r.T


# ══════════════════════════════════════════════════════════════════════════════
# A. RETRIEVAL METRICS
# ══════════════════════════════════════════════════════════════════════════════

def retrieval_metrics(sim_np, tgt_np, ks=(1, 5, 10)):
    """Retrieval metrics for both directions.

    Protein→GO (prot2go): given a protein, rank GO terms.
      R@k: fraction of the protein's true GO terms in top-k.
      P@k: fraction of top-k predictions that are true GO terms.

    GO→Protein (go2prot): given a GO term, rank proteins.
      R@k: fraction of the GO term's annotated proteins in top-k.
      P@k: fraction of top-k retrieved proteins that are truly annotated.
    """
    # Protein → GO  (sim: [N_prot, N_go])
    N = len(sim_np)
    order_p2g = np.argsort(-sim_np, axis=1)
    metrics = {}
    for k in ks:
        topk = order_p2g[:, :k]
        tp_k = np.array([tgt_np[i, topk[i]].sum() for i in range(N)])
        n_pos = tgt_np.sum(1)
        metrics[f"prot2go_R@{k}"] = float((tp_k / np.maximum(n_pos, 1)).mean())
        metrics[f"prot2go_P@{k}"] = float((tp_k / k).mean())

    # GO → Protein  (transpose sim: [N_go, N_prot])
    sim_T = sim_np.T          # [489, N_prot]
    tgt_T = tgt_np.T          # [489, N_prot]
    order_g2p = np.argsort(-sim_T, axis=1)
    G = len(sim_T)
    for k in ks:
        topk = order_g2p[:, :k]
        tp_k = np.array([tgt_T[j, topk[j]].sum() for j in range(G)])
        n_pos = tgt_T.sum(1)
        # only average over GO terms that have at least one annotated protein
        mask = n_pos > 0
        metrics[f"go2prot_R@{k}"] = float((tp_k[mask] / n_pos[mask]).mean())
        metrics[f"go2prot_P@{k}"] = float((tp_k[mask] / k).mean())

    return metrics


def fmax_aupr_wfmax(sim_np, tgt_np, ic_np):
    qs = np.linspace(1, 99, 199)
    thresholds = np.percentile(sim_np, qs)
    best_f, best_wf = 0.0, 0.0
    precs, recs = [], []
    for t in thresholds:
        preds = (sim_np > t).astype(np.float32)
        tp    = (preds * tgt_np).sum(1)
        prec  = float((tp / np.maximum(preds.sum(1), 1)).mean())
        rec   = float((tp / np.maximum(tgt_np.sum(1), 1)).mean())
        precs.append(prec); recs.append(rec)
        f1 = 2*prec*rec/(prec+rec+1e-8)
        best_f = max(best_f, f1)
        # wFmax
        tp_w   = (preds * tgt_np * ic_np).sum(1)
        pred_w = (preds * ic_np).sum(1)
        true_w = (tgt_np * ic_np).sum(1)
        wp = float((tp_w / np.maximum(pred_w, 1e-8)).mean())
        wr = float((tp_w / np.maximum(true_w, 1e-8)).mean())
        best_wf = max(best_wf, 2*wp*wr/(wp+wr+1e-8))
    order = np.argsort(recs)
    aupr  = float(np.trapz(np.array(precs)[order], np.array(recs)[order]))
    ndcg_scores, ap_scores = [], []
    full_order = np.argsort(-sim_np, axis=1)
    for i in range(len(sim_np)):
        true_idx = set(np.where(tgt_np[i] > 0.5)[0])
        if not true_idx: continue
        o = full_order[i]
        dcg  = sum(1/math.log2(r+2) for r,j in enumerate(o[:10]) if j in true_idx)
        idcg = sum(1/math.log2(r+2) for r in range(min(len(true_idx),10)))
        ndcg_scores.append(dcg/idcg if idcg > 0 else 0.0)
        hits, ap = 0, 0.0
        for r,j in enumerate(o):
            if j in true_idx:
                hits += 1; ap += hits/(r+1)
        ap_scores.append(ap/len(true_idx))
    return best_f, aupr, best_wf, float(np.mean(ndcg_scores)), float(np.mean(ap_scores))


# ══════════════════════════════════════════════════════════════════════════════
# B. DISTANCE DISTRIBUTIONS (MERU Fig 4)
# ══════════════════════════════════════════════════════════════════════════════

def plot_distance_distributions(all_data, save_path):
    """B1: Poincaré radius distribution proteins vs GO terms per model."""
    models = [k for k in all_data if k != "Euclidean"]
    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 4), sharey=False)
    if len(models) == 1: axes = [axes]

    for ax, label in zip(axes, models):
        d = all_data[label]
        if "seq_r" not in d: continue
        seq_r = d["seq_r"].numpy()
        go_r  = d["go_r"].numpy()
        bins = np.linspace(0, max(seq_r.max(), go_r.max()) * 1.05, 50)
        ax.hist(go_r,  bins=bins, alpha=0.7, color="#D65F5F", label="GO terms", density=True)
        ax.hist(seq_r, bins=bins, alpha=0.7, color="#4878CF", label="Proteins", density=True)
        ax.axvline(go_r.mean(),  color="#D65F5F", lw=2, ls="--")
        ax.axvline(seq_r.mean(), color="#4878CF", lw=2, ls="--")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Poincaré radius  d(z) = ‖x‖/(t + 1/√κ)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    plt.suptitle("Fig 4 analog: Embedding distance from origin (ROOT)\n"
                 "Text (GO terms) should sit closer to origin than sequences (proteins)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pair_distances(all_data, save_path):
    """B2: Distribution of positive vs negative pair distances."""
    labels = list(all_data.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(5*len(labels), 4), sharey=False)
    if len(labels) == 1: axes = [axes]

    for ax, label in zip(axes, labels):
        d = all_data[label]
        pos_d = d["pos_dist"].numpy()
        neg_d = d["neg_dist"].numpy()
        all_vals = np.concatenate([pos_d, neg_d])
        bins = np.linspace(all_vals.min(), np.percentile(all_vals, 99), 60)
        ax.hist(neg_d, bins=bins, alpha=0.6, color="#888888", label="Negatives", density=True)
        ax.hist(pos_d, bins=bins, alpha=0.8, color=PALETTE.get(label, "orange"),
                label="Positives", density=True)
        ax.axvline(pos_d.mean(), color=PALETTE.get(label, "orange"), lw=2, ls="--")
        ax.axvline(neg_d.mean(), color="#444444", lw=1.5, ls="--")
        ax.set_title(f"{label}\npos={pos_d.mean():.3f}  neg={neg_d.mean():.3f}", fontsize=9)
        ax.set_xlabel("Distance / dissimilarity")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.suptitle("Positive vs Negative pair distances\n"
                 "(larger gap = better separation)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# C. GEODESIC TRAVERSAL (MERU Fig 5)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def geodesic_traversal(probe, seq_r_single, go_r_all, go_names, n_steps=12):
    """Walk from a protein embedding toward origin along geodesic.
    At each step retrieve the nearest GO term.
    Returns list of (step, poincare_radius, go_term_name, go_term_idx).
    """
    curv = probe.curv.detach().cpu()
    x = seq_r_single.unsqueeze(0)   # [1, D]

    # log_map0: map to tangent space
    v = log_map0(x, curv)   # [1, D]

    results = []
    for s in np.linspace(1.0, 0.0, n_steps, endpoint=False):
        x_interp = exp_map0(s * v, curv)   # [1, D]
        r = poincare_radius(x_interp, curv).item()

        # Nearest GO term
        dists = pairwise_dist(x_interp, go_r_all, curv)[0]  # [489]
        nearest_idx = dists.argmin().item()

        results.append({
            "step":   round(s, 3),
            "radius": round(r, 4),
            "go_idx": nearest_idx,
            "go_name": go_names[nearest_idx],
            "dist":   dists[nearest_idx].item(),
        })
    return results


def plot_geodesic_traversals(probe, seq_r, go_r, tgt_test, vocab, save_path,
                             n_proteins=4):
    """Plot geodesic traversal for several example proteins."""
    go_names = vocab["go_names"]

    # Pick proteins with diverse number of GO terms
    n_pos = (tgt_test > 0.5).sum(1).numpy()
    chosen = []
    for target_n in [2, 4, 6, 10]:
        candidates = np.where(np.abs(n_pos - target_n) <= 1)[0]
        if len(candidates) > 0:
            chosen.append(int(candidates[len(candidates)//2]))
    chosen = chosen[:n_proteins]

    fig, axes = plt.subplots(n_proteins, 1, figsize=(14, 3.5*n_proteins))
    if n_proteins == 1: axes = [axes]

    for ax, prot_idx in zip(axes, chosen):
        trav = geodesic_traversal(probe, seq_r[prot_idx], go_r, go_names)
        true_go = set((tgt_test[prot_idx] > 0.5).nonzero(as_tuple=True)[0].tolist())

        steps   = [t["step"] for t in trav]
        radii   = [t["radius"] for t in trav]
        names   = [t["go_name"][:35] for t in trav]
        is_true = [t["go_idx"] in true_go for t in trav]

        # Plot radius curve
        ax2 = ax.twinx()
        ax2.plot(range(len(trav)), radii, color="#888888", lw=1.5, ls="--", alpha=0.6)
        ax2.set_ylabel("Poincaré radius", color="#888888", fontsize=8)
        ax2.tick_params(axis='y', colors='#888888', labelsize=7)

        # Show GO term labels
        for i, (name, hit) in enumerate(zip(names, is_true)):
            color = "#D65F5F" if hit else "#333333"
            weight = "bold" if hit else "normal"
            ax.text(i, 0.5, name, ha="center", va="center", fontsize=7,
                    color=color, fontweight=weight, rotation=35)

        ax.set_xlim(-0.5, len(trav)-0.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks(range(len(trav)))
        ax.set_xticklabels([f"s={t['step']:.2f}" for t in trav], fontsize=7, rotation=30)
        n_go = int(n_pos[prot_idx])
        ax.set_title(f"Protein #{prot_idx}  ({n_go} GO terms)   "
                     f"— red = protein's true GO term", fontsize=9)
        ax.set_xlabel("← specific (protein)                                       general (origin) →",
                      fontsize=8)

    plt.suptitle("Fig 5 analog: Geodesic traversal from protein toward origin\n"
                 "s=1.0 = protein position,  s=0.0 = origin (ROOT)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# D. GO HIERARCHY RECOVERY
# ══════════════════════════════════════════════════════════════════════════════

def plot_hierarchy_recovery(all_data, dag_edges, save_path):
    """For each model: do parent GO terms have lower Poincaré radius than children?"""
    models_with_r = {k: v for k, v in all_data.items() if "go_r" in v}
    if not dag_edges or not models_with_r:
        return

    parent_r_all, child_r_all, labels_all = [], [], []
    for label, d in models_with_r.items():
        go_r = d["go_r"].numpy()
        for p, c in dag_edges:
            parent_r_all.append(go_r[p])
            child_r_all.append(go_r[c])
            labels_all.append(label)

    parent_r_all = np.array(parent_r_all)
    child_r_all  = np.array(child_r_all)
    labels_all   = np.array(labels_all)

    fig, axes = plt.subplots(1, len(models_with_r), figsize=(5*len(models_with_r), 4))
    if len(models_with_r) == 1: axes = [axes]

    for ax, label in zip(axes, models_with_r):
        mask = labels_all == label
        pr   = parent_r_all[mask]
        cr   = child_r_all[mask]
        pct_correct = (pr < cr).mean() * 100  # parent should be closer to origin

        ax.scatter(pr, cr, alpha=0.15, s=8, color=PALETTE.get(label, "gray"))
        lim = max(pr.max(), cr.max()) * 1.05
        ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5)
        ax.set_xlabel("Parent radius (general)")
        ax.set_ylabel("Child radius (specific)")
        ax.set_title(f"{label}\n{pct_correct:.1f}% parent < child ✓", fontsize=9)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)

    plt.suptitle("D. GO hierarchy recovery\n"
                 "Correct: parent (general) should have smaller Poincaré radius than child (specific)",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# E. GO TERM IC vs POINCARÉ RADIUS
# ══════════════════════════════════════════════════════════════════════════════

def plot_ic_vs_radius(all_data, ic, save_path):
    models_with_r = {k: v for k, v in all_data.items() if "go_r" in v}
    if not models_with_r: return

    fig, axes = plt.subplots(1, len(models_with_r), figsize=(5*len(models_with_r), 4))
    if len(models_with_r) == 1: axes = [axes]
    ic_np = ic.numpy()

    for ax, label in zip(axes, models_with_r):
        go_r = all_data[label]["go_r"].numpy()
        from scipy.stats import spearmanr
        rho, p = spearmanr(ic_np, go_r)
        ax.scatter(ic_np, go_r, alpha=0.5, s=15, color=PALETTE.get(label, "gray"))
        ax.set_xlabel("GO term IC  (−log freq)  →  more specific")
        ax.set_ylabel("Poincaré radius  →  further from origin")
        ax.set_title(f"{label}\nSpearman ρ = {rho:.3f}  p = {p:.3e}", fontsize=9)

    plt.suptitle("E. GO term specificity (IC) vs Poincaré radius\n"
                 "Positive correlation expected: specific terms further from origin",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# F. PER-GO-TERM AP: Euclidean vs Hyp+MERU
# ══════════════════════════════════════════════════════════════════════════════

def per_term_ap(sim_np, tgt_np):
    """Average precision per GO term (GO-centric view)."""
    from sklearn.metrics import average_precision_score
    aps = []
    for j in range(tgt_np.shape[1]):
        y_true = tgt_np[:, j]
        if y_true.sum() == 0:
            aps.append(float("nan"))
        else:
            try:
                aps.append(average_precision_score(y_true, sim_np[:, j]))
            except Exception:
                aps.append(float("nan"))
    return np.array(aps)


def plot_per_term_comparison(all_data, ic, vocab, save_path):
    """Scatter: Euclidean AP vs Hyp+MERU AP per GO term."""
    if "Euclidean" not in all_data or "Hyp+MERU λ=0.5" not in all_data:
        return
    ap_euc  = all_data["Euclidean"]["per_ap"]
    ap_hyp  = all_data["Hyp+MERU λ=0.5"]["per_ap"]
    ic_np   = ic.numpy()
    go_names = vocab["go_names"]

    valid = ~(np.isnan(ap_euc) | np.isnan(ap_hyp))
    ap_e  = ap_euc[valid]; ap_h = ap_hyp[valid]
    ic_v  = ic_np[valid]
    diff  = ap_h - ap_e

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: scatter colored by IC
    sc = axes[0].scatter(ap_e, ap_h, c=ic_v, cmap="RdYlGn", alpha=0.6, s=20,
                         vmin=ic_v.min(), vmax=ic_v.max())
    lim = max(ap_e.max(), ap_h.max()) * 1.05
    axes[0].plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5)
    axes[0].set_xlabel("AP — Euclidean")
    axes[0].set_ylabel("AP — Hyp+MERU λ=0.5")
    axes[0].set_title("Per-GO-term Average Precision\n(above diagonal = Hyp+MERU wins)")
    plt.colorbar(sc, ax=axes[0], label="IC (more specific →)")

    # Annotate top wins / losses
    idx_sorted = np.argsort(diff)
    go_idx_valid = np.where(valid)[0]
    for rank in list(range(3)) + list(range(-3, 0)):
        i = idx_sorted[rank]
        axes[0].annotate(go_names[go_idx_valid[i]][:25],
                         (ap_e[i], ap_h[i]), fontsize=5.5, alpha=0.8,
                         xytext=(4, 0), textcoords="offset points")

    # Right: histogram of AP differences
    axes[1].hist(diff, bins=40, color="#D65F5F", alpha=0.8, edgecolor="white")
    axes[1].axvline(0, color="black", lw=1.5)
    axes[1].axvline(diff.mean(), color="#D65F5F", lw=2, ls="--",
                    label=f"mean Δ = {diff.mean():.4f}")
    pct_win = (diff > 0).mean() * 100
    axes[1].set_xlabel("AP(Hyp+MERU) − AP(Euclidean)")
    axes[1].set_ylabel("# GO terms")
    axes[1].set_title(f"Hyp+MERU wins on {pct_win:.1f}% of GO terms")
    axes[1].legend()

    plt.suptitle("F. Per-GO-term AP: Euclidean vs Hyp+MERU", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# G. 2D UMAP EMBEDDING VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_umap_embeddings(all_data, tgt_test, vocab, ic, save_path, n_proteins=300):
    """2D UMAP of protein + GO embeddings for Euclidean and Hyp+MERU."""
    try:
        from umap import UMAP
    except ImportError:
        print("  [skip UMAP] umap-learn not installed")
        return

    keys = ["Euclidean", "Hyp+MERU λ=0.5"]
    keys = [k for k in keys if k in all_data]
    if not keys: return

    go_names = vocab["go_names"]
    ic_np = ic.numpy()

    fig, axes = plt.subplots(1, len(keys), figsize=(8*len(keys), 7))
    if len(keys) == 1: axes = [axes]

    # Sample proteins
    np.random.seed(42)
    prot_idx = np.random.choice(len(all_data[keys[0]]["seq_emb"]), n_proteins, replace=False)
    n_pos = (tgt_test > 0.5).sum(1).numpy()

    for ax, label in zip(axes, keys):
        seq_emb = all_data[label]["seq_emb"][prot_idx]
        go_emb  = all_data[label]["go_emb"]

        combined = np.vstack([go_emb, seq_emb])
        reducer  = UMAP(n_components=2, random_state=42, min_dist=0.1, n_neighbors=20)
        emb2d    = reducer.fit_transform(combined)

        go_2d  = emb2d[:489]
        seq_2d = emb2d[489:]

        # Plot GO terms colored by IC
        sc = ax.scatter(go_2d[:, 0], go_2d[:, 1], c=ic_np, cmap="YlOrRd",
                        s=40, alpha=0.85, marker="^", zorder=3,
                        vmin=ic_np.min(), vmax=ic_np.max(),
                        label="GO terms (▲, color=IC)")

        # Plot proteins colored by #GO terms
        ax.scatter(seq_2d[:, 0], seq_2d[:, 1], c=n_pos[prot_idx], cmap="Blues",
                   s=12, alpha=0.4, marker="o", zorder=2,
                   vmin=1, vmax=n_pos.max(), label="Proteins (●)")

        plt.colorbar(sc, ax=ax, label="GO term IC (specificity)")
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    plt.suptitle("G. UMAP of protein + GO embeddings\n"
                 "GO terms (▲) colored by IC; proteins (●) colored by #GO annotations",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# H. SUMMARY METRICS TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_full_table(results):
    W = 24
    cols = ["Fmax", "AUPR", "wFmax", "nDCG@10", "MAP", "R@1", "R@5", "R@10", "P@1", "P@5", "P@10"]
    sep = "=" * (W + 2 + len(cols) * 9)
    print(sep)
    print(f"  {'Model':<{W}}" + "".join(f"  {c:>7}" for c in cols))
    print("-" * (W + 2 + len(cols) * 9))
    for label, r in results.items():
        row = f"  {label:<{W}}"
        for c in cols:
            v = r.get(c.lower().replace("@","").replace("+",""), r.get(c, None))
            # Try alternate key formats
            if v is None:
                key = c.lower()
                v = r.get(key, None)
            if v is None:
                row += "        -"
            else:
                row += f"  {v:>7.4f}"
        print(row)
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

CHECKPOINTS = [
    ("probe_Euclidean.pt",          "Euclidean",          "euclidean", 256),
    ("probe_Hyp.pt",                "Hyp",                "lorentz",   256),
    ("probe_HyppMERU_lam05.pt",     "Hyp+MERU λ=0.5",    "lorentz",   256),
    ("probe_HyppMERUpDAG_lam05.pt", "Hyp+MERU+DAG λ=0.5","lorentz",   256),
]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Loading data ...")
    seq_test, tgt_test, go_embs, ic, vocab, dag_edges = load_all(device)
    tgt_np = tgt_test.numpy().astype(np.float32)
    ic_np  = ic.numpy().astype(np.float32)
    print(f"  test={seq_test.shape}, GO={go_embs.shape}, DAG edges={len(dag_edges)}\n")

    # ── Compute embeddings + all metrics for each checkpoint ─────────────────
    all_data = {}
    results  = {}

    for fname, label, geometry, proj_dim in CHECKPOINTS:
        if not (CKPT_DIR / fname).exists():
            print(f"  [skip] {fname}")
            continue
        print(f"── {label} ──────────────────────────────")

        probe  = load_probe(fname, geometry, proj_dim, device)
        seq_r, go_r = get_embeddings(probe, seq_test, go_embs, device)
        sim    = get_sim(seq_r, go_r, probe)
        sim_np = sim.numpy().astype(np.float32)

        # Store raw embeddings for UMAP (all models)
        data = {"sim": sim_np, "seq_emb": seq_r.numpy(), "go_emb": go_r.numpy()}

        # Distance from origin (for B1 / Fig 4 analog) — hyperbolic only
        if geometry == "lorentz":
            curv = probe.curv.detach().cpu()
            data["seq_r"] = poincare_radius(seq_r, curv)
            data["go_r"]  = poincare_radius(go_r,  curv)

        # Positive and negative pair distances (for B2)
        pos_mask = (tgt_np > 0.5)   # [N, 489]
        # Sample 5000 pairs to keep memory manageable
        rng = np.random.default_rng(42)
        pos_i, pos_j = np.where(pos_mask)
        neg_i, neg_j = np.where(~pos_mask)
        s = min(5000, len(pos_i))
        pi = rng.choice(len(pos_i), s, replace=False)
        ni = rng.choice(len(neg_i), s, replace=False)
        if geometry == "lorentz":
            # distance = negative similarity for our negdist sim
            data["pos_dist"] = torch.tensor(-sim_np[pos_i[pi], pos_j[pi]])
            data["neg_dist"] = torch.tensor(-sim_np[neg_i[ni], neg_j[ni]])
        else:
            # cosine: distance = 1 - sim
            data["pos_dist"] = torch.tensor(1 - sim_np[pos_i[pi], pos_j[pi]])
            data["neg_dist"] = torch.tensor(1 - sim_np[neg_i[ni], neg_j[ni]])

        # Per-GO-term AP (for F)
        data["per_ap"] = per_term_ap(sim_np, tgt_np)

        all_data[label] = data

        # Metrics
        fmax, aupr, wfmax, ndcg10, map_s = fmax_aupr_wfmax(sim_np, tgt_np, ic_np)
        ret = retrieval_metrics(sim_np, tgt_np)
        kappa = probe.curv.item() if geometry == "lorentz" else None

        row = {
            "fmax": round(fmax, 4), "aupr": round(aupr, 4),
            "wfmax": round(wfmax, 4), "ndcg10": round(ndcg10, 4),
            "map": round(map_s, 4),
            **{k.lower().replace("@", ""): round(v, 4) for k, v in ret.items()},
        }
        if kappa:
            row["kappa"] = round(kappa, 4)
            row["mean_seq_r"] = round(data["seq_r"].mean().item(), 4)
            row["mean_go_r"]  = round(data["go_r"].mean().item(), 4)
        results[label] = row

        print(f"  Fmax={fmax:.4f}  AUPR={aupr:.4f}  wFmax={wfmax:.4f}  "
              f"nDCG@10={ndcg10:.4f}  MAP={map_s:.4f}")
        p2g = {k: v for k, v in ret.items() if k.startswith("prot2go")}
        g2p = {k: v for k, v in ret.items() if k.startswith("go2prot")}
        print(f"  Prot→GO:  " + "  ".join(f"{k}={v:.4f}" for k, v in p2g.items()))
        print(f"  GO→Prot:  " + "  ".join(f"{k}={v:.4f}" for k, v in g2p.items()))
        if kappa:
            print(f"  κ={kappa:.4f}  seq_r={data['seq_r'].mean():.4f}  "
                  f"go_r={data['go_r'].mean():.4f}")
        print()

    # ── Full numeric tables ───────────────────────────────────────────────────
    print("\n" + "─"*90)
    print("PREDICTION METRICS")
    print("─"*90)
    W = 24
    pred_cols = ["Fmax", "AUPR", "wFmax", "nDCG@10", "MAP"]
    pred_keys = {"Fmax":"fmax","AUPR":"aupr","wFmax":"wfmax","nDCG@10":"ndcg10","MAP":"map"}
    print(f"  {'Model':<{W}}" + "".join(f"  {c:>9}" for c in pred_cols))
    print("─" * (W + 2 + len(pred_cols) * 11))
    for label, r in results.items():
        row = f"  {label:<{W}}"
        for c in pred_cols:
            v = r.get(pred_keys[c])
            row += f"  {v:>9.4f}" if v is not None else "          -"
        print(row)
    print("=" * (W + 2 + len(pred_cols) * 11))

    print("\n" + "─"*90)
    print("RETRIEVAL METRICS — Protein → GO term  (given a protein, rank 489 GO terms)")
    print("─"*90)
    # stored keys are lowercased: prot2go_r1, prot2go_r5, ...
    ret_cols = ["prot2go_r1","prot2go_r5","prot2go_r10","prot2go_p1","prot2go_p5","prot2go_p10"]
    hdr      = ["R@1","R@5","R@10","P@1","P@5","P@10"]
    print(f"  {'Model':<{W}}" + "".join(f"  {h:>7}" for h in hdr))
    print("─" * (W + 2 + len(hdr) * 9))
    for label, r in results.items():
        row = f"  {label:<{W}}"
        for c in ret_cols:
            v = r.get(c)
            row += f"  {v:>7.4f}" if v is not None else "        -"
        print(row)
    print("=" * (W + 2 + len(hdr) * 9))

    print("\n" + "─"*90)
    print("RETRIEVAL METRICS — GO term → Protein  (given a GO term, rank 2991 proteins)")
    print("─"*90)
    ret_cols2 = ["go2prot_r1","go2prot_r5","go2prot_r10","go2prot_p1","go2prot_p5","go2prot_p10"]
    print(f"  {'Model':<{W}}" + "".join(f"  {h:>7}" for h in hdr))
    print("─" * (W + 2 + len(hdr) * 9))
    for label, r in results.items():
        row = f"  {label:<{W}}"
        for c in ret_cols2:
            v = r.get(c)
            row += f"  {v:>7.4f}" if v is not None else "        -"
        print(row)
    print("=" * (W + 2 + len(hdr) * 9))

    if any("kappa" in r for r in results.values()):
        print(f"\n  {'Model':<{W}}  {'κ':>7}  {'seq_r':>7}  {'go_r':>7}")
        print("─" * 52)
        for label, r in results.items():
            if "kappa" in r:
                print(f"  {label:<{W}}  {r['kappa']:>7.4f}  "
                      f"{r['mean_seq_r']:>7.4f}  {r['mean_go_r']:>7.4f}")

    # ── Save numerical results ────────────────────────────────────────────────
    out = RESULTS_DIR / "results_complete.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")

    # ── Visualizations ────────────────────────────────────────────────────────
    print("\nGenerating figures ...")

    plot_distance_distributions(all_data, FIG_DIR / "fig4_radius_distribution.png")
    plot_pair_distances(all_data, FIG_DIR / "fig4b_pair_distances.png")

    # Fig 5 analog: geodesic traversal (only for Hyp+MERU)
    if "Hyp+MERU λ=0.5" in all_data:
        probe_meru = load_probe("probe_HyppMERU_lam05.pt", "lorentz", 256, device)
        seq_r_meru, go_r_meru = get_embeddings(probe_meru, seq_test, go_embs, device)
        plot_geodesic_traversals(probe_meru, seq_r_meru, go_r_meru,
                                 tgt_test, vocab,
                                 FIG_DIR / "fig5_geodesic_traversal.png", n_proteins=4)

    plot_hierarchy_recovery(all_data, dag_edges, FIG_DIR / "fig_hierarchy_recovery.png")
    plot_ic_vs_radius(all_data, ic, FIG_DIR / "fig_ic_vs_radius.png")
    plot_per_term_comparison(all_data, ic, vocab, FIG_DIR / "fig_per_term_ap.png")
    plot_umap_embeddings(all_data, tgt_test, vocab, ic,
                         FIG_DIR / "fig_umap.png", n_proteins=400)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
