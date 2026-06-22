#!/usr/bin/env python3
"""
Gromov δ-hyperbolicity of learned embedding spaces.

For four points x,y,z,w in a metric space, define three pairwise-sum candidates:
    S1 = d(x,y) + d(z,w)
    S2 = d(x,z) + d(y,w)
    S3 = d(x,w) + d(y,z)
Sort descending: δ = (S1 - S2) / 2   (the gap between the two largest sums).

A tree has δ = 0.  Euclidean R^n has δ = D/2 (diameter/2) — maximally non-hyperbolic.
We report δ_rel = δ / (D/2) ∈ [0, 1] so results are comparable across models.

We compute this on:
  (a) GO term embeddings (489 terms — the hierarchical side)
  (b) Protein test embeddings (2991 — sampled 500 for speed)

Run:
    conda activate pannot-infer
    cd /home/yining_yang/NLP/hyperbolic/experiments/hyp_ssf_probe
    python compute_hyperbolicity.py
"""

from __future__ import annotations
import json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT   = Path(__file__).resolve().parent
CACHE_DIR   = REPO_ROOT / "cache"
CKPT_DIR    = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"

N_SAMPLES   = 50_000   # number of random 4-tuples to sample
SEED        = 42
EPS         = 1e-8

# ── Lorentz math (same as training) ──────────────────────────────────────────

def _time(x, curv):
    return torch.sqrt(1.0 / curv + (x * x).sum(-1))

def exp_map0(x, curv):
    sqrt_c = curv.sqrt()
    xn = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    return torch.sinh((sqrt_c * xn).clamp(max=math.asinh(2.0**15))) * x / xn

def lorentz_dist(x, y, curv):
    """Pairwise Lorentz distance [N,D] × [M,D] → [N,M]."""
    xt = _time(x, curv).unsqueeze(1)   # [N,1]
    yt = _time(y, curv).unsqueeze(0)   # [1,M]
    inner = x @ y.T - xt * yt          # [N,M]
    return torch.acosh((-curv * inner).clamp(min=1.0 + EPS)) / curv.sqrt()

def euclidean_dist(x, y):
    return torch.cdist(x, y)

# ── Model (same as training) ──────────────────────────────────────────────────

class LorentzHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj      = nn.Linear(in_dim, out_dim, bias=False)
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
    def __init__(self, seq_dim, text_dim, proj_dim, geometry):
        super().__init__()
        self.geometry = geometry
        if geometry == "lorentz":
            self.log_curv  = nn.Parameter(torch.tensor(0.0))
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

# ── Hyperbolicity ─────────────────────────────────────────────────────────────

def gromov_delta(dist_np: np.ndarray, n_samples: int = N_SAMPLES, seed: int = SEED):
    """
    Sample random 4-tuples and compute Gromov δ.

    Returns:
        delta_mean  — mean δ over sampled 4-tuples
        delta_max   — max δ over sampled 4-tuples (worst case)
        diameter    — max pairwise distance
        delta_rel   — delta_max / (diameter / 2)  ∈ [0,1]
    """
    rng = np.random.default_rng(seed)
    n   = dist_np.shape[0]
    deltas = []

    for _ in range(n_samples):
        i, j, k, l = rng.choice(n, size=4, replace=False)
        s1 = dist_np[i, j] + dist_np[k, l]
        s2 = dist_np[i, k] + dist_np[j, l]
        s3 = dist_np[i, l] + dist_np[j, k]
        top2 = sorted([s1, s2, s3], reverse=True)[:2]
        deltas.append((top2[0] - top2[1]) / 2.0)

    deltas   = np.array(deltas)
    diameter = dist_np.max()
    d_mean   = float(deltas.mean())
    d_max    = float(deltas.max())
    d_rel    = d_max / (diameter / 2.0 + 1e-8)
    return d_mean, d_max, diameter, d_rel


@torch.no_grad()
def get_embeddings(probe, seq_feats, go_embs, device, n_prot=500):
    probe.eval()
    probe.to(device)

    go_emb = probe.encode_text(go_embs.to(device)).cpu()   # [489, D]

    # sample proteins for speed
    idx    = torch.randperm(len(seq_feats))[:n_prot]
    prot_emb = probe.encode_seq(seq_feats[idx].to(device)).cpu()   # [n_prot, D]

    return prot_emb, go_emb


def compute_dists(emb, geometry, curv_val):
    if geometry == "lorentz":
        curv = torch.tensor(curv_val)
        D = lorentz_dist(emb, emb, curv).numpy()
    else:
        D = euclidean_dist(emb, emb).numpy()
    return D.astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load data
    esm_feat = torch.load(CACHE_DIR / "esm2_go_feats.pt", map_location="cpu", weights_only=True)
    go_embs  = torch.load(CACHE_DIR / "go_term_embs.pt",  map_location="cpu", weights_only=True)
    seq_test = esm_feat["test"]   # [2991, 320]

    checkpoints = [
        ("probe_Euclidean.pt",          "Euclidean",          "euclidean", 256),
        ("probe_Hyp.pt",                "Hyp",                "lorentz",   256),
        ("probe_HyppMERU_lam05.pt",     "Hyp+MERU λ=0.5",    "lorentz",   256),
        ("probe_HyppMERUpDAG_lam05.pt", "Hyp+MERU+DAG λ=0.5","lorentz",   256),
    ]

    results = {}
    for fname, label, geometry, proj_dim in checkpoints:
        ckpt_path = CKPT_DIR / fname
        if not ckpt_path.exists():
            print(f"  [skip] {fname} not found")
            continue

        print(f"── {label} ──────────────────────────────────────────────────")
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        probe = GOProbe(320, 768, proj_dim, geometry)
        probe.load_state_dict(ckpt["state_dict"])
        curv_val = probe.curv.item() if geometry == "lorentz" else None

        prot_emb, go_emb = get_embeddings(probe, seq_test, go_embs, device)

        # GO term hyperbolicity
        t0 = time.time()
        D_go = compute_dists(go_emb, geometry, curv_val)
        go_mean, go_max, go_diam, go_rel = gromov_delta(D_go)
        print(f"  GO  embeddings (n=489):  δ_mean={go_mean:.4f}  δ_max={go_max:.4f}"
              f"  diam={go_diam:.4f}  δ_rel={go_rel:.4f}  ({time.time()-t0:.1f}s)")

        # Protein hyperbolicity
        t0 = time.time()
        D_prot = compute_dists(prot_emb, geometry, curv_val)
        pr_mean, pr_max, pr_diam, pr_rel = gromov_delta(D_prot)
        print(f"  Prot embeddings (n=500): δ_mean={pr_mean:.4f}  δ_max={pr_max:.4f}"
              f"  diam={pr_diam:.4f}  δ_rel={pr_rel:.4f}  ({time.time()-t0:.1f}s)")

        if curv_val:
            print(f"  κ = {curv_val:.4f}")
        print()

        results[label] = {
            "go":   {"delta_mean": round(go_mean,4), "delta_max": round(go_max,4),
                     "diameter":   round(float(go_diam),4), "delta_rel": round(go_rel,4)},
            "prot": {"delta_mean": round(pr_mean,4), "delta_max": round(pr_max,4),
                     "diameter":   round(float(pr_diam),4), "delta_rel": round(pr_rel,4)},
        }
        if curv_val:
            results[label]["kappa"] = round(curv_val, 4)

    # ── Summary table ────────────────────────────────────────────────────────
    sep = "=" * 88
    print(sep)
    print(f"  {'Model':<26}  {'GO δ_mean':>10}  {'GO δ_max':>9}  {'GO δ_rel':>9}"
          f"  {'Pr δ_mean':>10}  {'Pr δ_rel':>9}")
    print("-" * 88)
    for label, r in results.items():
        g, p = r["go"], r["prot"]
        print(f"  {label:<26}  {g['delta_mean']:>10.4f}  {g['delta_max']:>9.4f}"
              f"  {g['delta_rel']:>9.4f}  {p['delta_mean']:>10.4f}  {p['delta_rel']:>9.4f}")
    print(sep)
    print("\nInterpretation: δ_rel ∈ [0,1].  0 = tree-like.  1 = flat Euclidean.")

    out = RESULTS_DIR / "hyperbolicity.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
