#!/usr/bin/env python3
"""
Evaluate saved checkpoints with full metric suite.

Metrics:
  Fmax      — protein-centric max F1 over threshold sweep (CAFA standard)
  AUPR      — area under protein-centric precision-recall curve
  nDCG@k    — rank quality: are true GO terms near top of ranked list?
  wFmax     — Fmax weighted by GO term information content (rare terms count more)
  MAP       — mean average precision over ranked GO terms per protein

Run:
    conda activate pannot-infer
    cd /home/yining_yang/NLP/hyperbolic/experiments/hyp_ssf_probe
    python evaluate_checkpoints.py
"""

from __future__ import annotations
import json, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT   = Path(__file__).resolve().parent
CACHE_DIR   = REPO_ROOT / "cache"
CKPT_DIR    = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"

EPS = 1e-8

# ── Lorentz math (identical to training script) ───────────────────────────────

def _time(x, curv):
    return torch.sqrt(1.0 / curv + (x * x).sum(-1))

def exp_map0(x, curv):
    sqrt_c = curv.sqrt()
    xn = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    return torch.sinh((sqrt_c * xn).clamp(max=math.asinh(2.0**15))) * x / xn

def pairwise_dist(x, y, curv):
    xt = _time(x, curv).unsqueeze(1)
    yt = _time(y, curv).unsqueeze(0)
    inner = x @ y.T - xt * yt
    return torch.acosh((-curv * inner).clamp(min=1.0 + EPS)) / curv.sqrt()

def poincare_radius(x, curv):
    t = _time(x, curv)
    return x.norm(dim=-1) / (t + 1.0 / curv.sqrt())

# ── Model (identical to training script) ─────────────────────────────────────

class LorentzHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(1.0 / math.sqrt(out_dim))))

    def forward(self, z, curv):
        return exp_map0(self.proj(z) * self.log_alpha.exp(), curv)

    def clamp(self):
        self.log_alpha.data.clamp_(max=0.0)


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
            if learn_curv:
                self.log_curv = nn.Parameter(torch.tensor(0.0))
            else:
                self.register_buffer("log_curv", torch.tensor(0.0))
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


# ── Load data and embeddings ──────────────────────────────────────────────────

def load_eval_data(device):
    splits   = torch.load(CACHE_DIR / "protst_go_mf_decoded.pt",  map_location="cpu", weights_only=False)
    esm_feat = torch.load(CACHE_DIR / "esm2_go_feats.pt",         map_location="cpu", weights_only=True)
    go_embs  = torch.load(CACHE_DIR / "go_term_embs.pt",          map_location="cpu", weights_only=True)

    seq_test = esm_feat["test"]               # [2991, 320]
    tgt_test = splits["test"]["targets"]      # [2991, 489]

    # GO term IC from training labels (IC = -log(freq / N_proteins))
    tgt_train = splits["train"]["targets"]
    term_freq  = (tgt_train > 0.5).float().sum(0)   # [489]
    N_train    = len(tgt_train)
    ic = -torch.log((term_freq / N_train).clamp(min=1e-6))  # [489]

    return seq_test, tgt_test, go_embs, ic


@torch.no_grad()
def get_sim_matrix(probe, seq_test, go_embs, device):
    probe.eval()
    go_r = probe.encode_text(go_embs.to(device)).cpu()   # [489, D]
    rows = []
    for i in range(0, len(seq_test), 256):
        sr = probe.encode_seq(seq_test[i:i+256].to(device)).cpu()
        if probe.geometry == "lorentz":
            rows.append(-pairwise_dist(sr, go_r, probe.curv.detach().cpu()))
        else:
            rows.append(sr @ go_r.T)
    return torch.cat(rows)   # [N_test, 489]


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_fmax_aupr(sim_np, tgt_np):
    qs = np.linspace(1, 99, 199)
    thresholds = np.percentile(sim_np, qs)
    best_f, best_t = 0.0, thresholds[0]
    precs, recs = [], []
    for t in thresholds:
        preds = (sim_np > t).astype(np.float32)
        tp    = (preds * tgt_np).sum(1)
        prec  = float((tp / np.maximum(preds.sum(1), 1)).mean())
        rec   = float((tp / np.maximum(tgt_np.sum(1), 1)).mean())
        f1    = 2.0 * prec * rec / (prec + rec + 1e-8)
        precs.append(prec); recs.append(rec)
        if f1 > best_f:
            best_f, best_t = f1, float(t)
    order = np.argsort(recs)
    aupr  = float(np.trapz(np.array(precs)[order], np.array(recs)[order]))
    return best_f, float(best_t), aupr


def compute_wfmax(sim_np, tgt_np, ic_np):
    """Weighted Fmax: each GO term weighted by information content."""
    qs = np.linspace(1, 99, 199)
    thresholds = np.percentile(sim_np, qs)
    best_wf = 0.0
    for t in thresholds:
        preds = (sim_np > t).astype(np.float32)
        tp_w  = (preds * tgt_np * ic_np).sum(1)
        pred_w = (preds * ic_np).sum(1)
        true_w = (tgt_np * ic_np).sum(1)
        prec = float((tp_w / np.maximum(pred_w, 1e-8)).mean())
        rec  = float((tp_w / np.maximum(true_w, 1e-8)).mean())
        wf1  = 2.0 * prec * rec / (prec + rec + 1e-8)
        best_wf = max(best_wf, wf1)
    return best_wf


def compute_ndcg_map(sim_np, tgt_np, k=10):
    """nDCG@k and MAP over ranked GO terms per protein."""
    ndcg_scores, ap_scores = [], []
    for i in range(len(sim_np)):
        true_idx = set(np.where(tgt_np[i] > 0.5)[0])
        if not true_idx:
            continue
        order = np.argsort(sim_np[i])[::-1]   # descending

        # nDCG@k
        dcg  = sum(1.0 / math.log2(r + 2) for r, j in enumerate(order[:k]) if j in true_idx)
        idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(true_idx), k)))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

        # MAP: average precision over full ranked list
        hits, ap = 0, 0.0
        for r, j in enumerate(order):
            if j in true_idx:
                hits += 1
                ap += hits / (r + 1)
        ap_scores.append(ap / len(true_idx))

    return float(np.mean(ndcg_scores)), float(np.mean(ap_scores))


def poincare_stats(probe, seq_test, go_embs, tgt_test, device):
    if probe.geometry != "lorentz":
        return {}
    probe.eval()
    with torch.no_grad():
        curv = probe.curv.detach().cpu()
        sr   = probe.encode_seq(seq_test[:500].to(device)).cpu()
        go_r = probe.encode_text(go_embs.to(device)).cpu()
    seq_r = poincare_radius(sr, curv)
    go_r_ = poincare_radius(go_r, curv)
    return {
        "kappa":          round(curv.item(), 4),
        "mean_seq_r":     round(seq_r.mean().item(), 4),
        "mean_go_r":      round(go_r_.mean().item(), 4),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Loading data and embeddings ...")
    seq_test, tgt_test, go_embs, ic = load_eval_data(device)
    tgt_np = tgt_test.numpy().astype(np.float32)
    ic_np  = ic.numpy().astype(np.float32)
    print(f"  test: {seq_test.shape}, GO terms: {go_embs.shape}, IC range: [{ic.min():.2f}, {ic.max():.2f}]\n")

    # Map checkpoint file → (label, geometry, proj_dim)
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

        print(f"── {label} ──────────────────────────────────────────────")
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        probe = GOProbe(320, 768, proj_dim, geometry, learn_curv=True)
        probe.load_state_dict(ckpt["state_dict"])
        probe.to(device).eval()

        sim    = get_sim_matrix(probe, seq_test, go_embs, device)
        sim_np = sim.numpy().astype(np.float32)

        fmax, best_t, aupr  = compute_fmax_aupr(sim_np, tgt_np)
        wfmax               = compute_wfmax(sim_np, tgt_np, ic_np)
        ndcg, map_score     = compute_ndcg_map(sim_np, tgt_np, k=10)
        pstats              = poincare_stats(probe, seq_test, go_embs, tgt_test, device)

        row = {
            "fmax":  round(fmax,       4),
            "aupr":  round(aupr,       4),
            "wfmax": round(wfmax,      4),
            "ndcg10":round(ndcg,       4),
            "map":   round(map_score,  4),
        }
        row.update(pstats)
        results[label] = row

        print(f"  Fmax={fmax:.4f}  AUPR={aupr:.4f}  wFmax={wfmax:.4f}  "
              f"nDCG@10={ndcg:.4f}  MAP={map_score:.4f}")
        if pstats:
            print(f"  κ={pstats['kappa']:.4f}  "
                  f"seq_r={pstats['mean_seq_r']:.4f}  go_r={pstats['mean_go_r']:.4f}")
        print()

    # ── Table ─────────────────────────────────────────────────────────────────
    W = 24
    sep = "=" * 82
    print(sep)
    print(f"  {'Model':<{W}}  {'Fmax':>7}  {'AUPR':>7}  {'wFmax':>7}  {'nDCG@10':>8}  {'MAP':>7}")
    print("-" * 82)
    for label, r in results.items():
        print(f"  {label:<{W}}  {r['fmax']:>7.4f}  {r['aupr']:>7.4f}  "
              f"{r['wfmax']:>7.4f}  {r['ndcg10']:>8.4f}  {r['map']:>7.4f}")
    print(sep)

    if any("kappa" in r for r in results.values()):
        print(f"\n  {'Model':<{W}}  {'κ':>7}  {'seq_r':>7}  {'go_r':>7}")
        print("-" * 52)
        for label, r in results.items():
            if "kappa" in r:
                print(f"  {label:<{W}}  {r['kappa']:>7.4f}  "
                      f"{r['mean_seq_r']:>7.4f}  {r['mean_go_r']:>7.4f}")

    out = RESULTS_DIR / "results_full_eval.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
