#!/usr/bin/env python3
"""
Hyperbolic SSF Probe v2 — Clean geometry comparison
====================================================
Fixes three confounds from v1:
  1. Encoder:   ESM2-650M frozen (vs 8M) — stronger protein features
  2. Projector: 2-layer MLP (vs linear) — nonlinear capacity before exp_map0
  3. Loss:      BCE over all 489 GO terms (vs InfoNCE) — no false negatives

All four geometry conditions are otherwise identical, so the only variable
is the embedding space geometry.

Conditions:
  1. Euclidean      — MLP → L2-norm, cosine sim, BCE
  2. Hyp            — MLP → exp_map0, Lorentz dist, BCE
  3. Hyp+MERU λ=0.5 — + entailment cone loss (GO→protein)
  4. Hyp+MERU+DAG   — + DAG is_a edges (parent→child GO)

Run:
    conda activate hypalign
    cd experiments/hyp_ssf_probe
    python -u run_experiment_go_v2.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT   = Path(__file__).resolve().parent
CACHE_DIR   = REPO_ROOT / "cache"
RESULTS_DIR = REPO_ROOT / "results"
CKPT_DIR    = REPO_ROOT / "checkpoints"
for d in [CACHE_DIR, RESULTS_DIR, CKPT_DIR]:
    d.mkdir(exist_ok=True)

ESM2_HF       = "facebook/esm2_t33_650M_UR50D"
PUBMEDBERT_HF = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

GO_TERMS    = 489
ESM_DIM     = 1280       # ESM2-650M hidden dim
BERT_DIM    = 768
PROJ_HIDDEN = 512        # MLP hidden layer
PROJ_DIM    = 256        # output embedding dim
BATCH_SIZE  = 128
LR          = 3e-4
EPOCHS      = 60
TEMPERATURE = 0.07
CURV_INIT   = 1.0
LAMBDA_MERU = 0.5
LAMBDA_DAG  = 0.5
EPS         = 1e-8

# ── Lorentz math ──────────────────────────────────────────────────────────────

def _time(x, curv):
    return torch.sqrt(1.0 / curv + (x * x).sum(-1))

def exp_map0(x, curv):
    sqrt_c = curv.sqrt()
    xn = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    return torch.sinh((sqrt_c * xn).clamp(max=math.asinh(2.0**15))) * x / xn

def log_map0(x, curv):
    sqrt_c = curv.sqrt()
    xn = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    return (torch.asinh(sqrt_c * xn) / (sqrt_c * xn + EPS)) * x

def pairwise_dist(x, y, curv):
    xt = _time(x, curv).unsqueeze(1)
    yt = _time(y, curv).unsqueeze(0)
    inner = x @ y.T - xt * yt
    return torch.acosh((-curv * inner).clamp(min=1.0 + EPS)) / curv.sqrt()

def poincare_radius(x, curv):
    return x.norm(dim=-1) / (_time(x, curv) + 1.0 / curv.sqrt())

# ── Projector heads (MLP) ─────────────────────────────────────────────────────

class MLPLorentzHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )
        self.log_alpha = nn.Parameter(torch.tensor(math.log(1.0 / math.sqrt(out_dim))))

    def forward(self, z, curv):
        return exp_map0(self.net(z) * self.log_alpha.exp(), curv)

    def clamp(self):
        self.log_alpha.data.clamp_(max=0.0)


class MLPEuclideanHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, z):
        return F.normalize(self.net(z), dim=-1)


# ── Probe model ───────────────────────────────────────────────────────────────

class GOProbeV2(nn.Module):
    def __init__(self, seq_dim, text_dim, hidden_dim, proj_dim, geometry,
                 learn_curv=True):
        super().__init__()
        self.geometry = geometry
        if geometry == "lorentz":
            self.log_curv  = nn.Parameter(torch.tensor(math.log(CURV_INIT))) \
                             if learn_curv else \
                             torch.tensor(math.log(CURV_INIT))
            self.seq_head  = MLPLorentzHead(seq_dim,  hidden_dim, proj_dim)
            self.text_head = MLPLorentzHead(text_dim, hidden_dim, proj_dim)
        else:
            self.seq_head  = MLPEuclideanHead(seq_dim,  hidden_dim, proj_dim)
            self.text_head = MLPEuclideanHead(text_dim, hidden_dim, proj_dim)
        self.log_temp = nn.Parameter(torch.tensor(math.log(TEMPERATURE)))

    @property
    def curv(self):
        return self.log_curv.exp().clamp(0.1, 10.0)

    def encode_seq(self, x):
        return self.seq_head(x, self.curv) if self.geometry == "lorentz" \
               else self.seq_head(x)

    def encode_text(self, x):
        return self.text_head(x, self.curv) if self.geometry == "lorentz" \
               else self.text_head(x)

    def similarity(self, seq_emb, go_emb):
        """[B, D] × [N_go, D] → [B, N_go] similarity logits."""
        if self.geometry == "lorentz":
            return -pairwise_dist(seq_emb, go_emb, self.curv) \
                   * self.log_temp.exp()
        else:
            return seq_emb @ go_emb.T * self.log_temp.exp()


# ── Losses ────────────────────────────────────────────────────────────────────

def bce_loss(logits, targets):
    """Dense multi-label BCE over all 489 GO terms — no false negatives."""
    return F.binary_cross_entropy_with_logits(logits, targets.float())


def meru_loss(probe, seq_emb, go_emb):
    """
    Entailment cone: GO term (general) should entail protein (specific).
    oxy_angle(go, seq) ≤ half_aperture(go)
    """
    curv     = probe.curv
    sqrt_c   = curv.sqrt()
    r_min    = 1.0 / sqrt_c   # minimum Poincaré radius for cone formula

    go_norm  = go_emb.norm(dim=-1).clamp(min=EPS)
    seq_norm = seq_emb.norm(dim=-1).clamp(min=EPS)

    # Half-aperture of go cone
    aperture = torch.asin((2 * r_min / go_norm).clamp(max=1.0 - EPS))

    # Exterior angle at go in O-go-seq triangle
    go_t  = _time(go_emb,  curv)
    seq_t = _time(seq_emb, curv)
    cos_num = go_t * curv * (go_emb * seq_emb).sum(-1) - seq_t
    cos_den = (go_t ** 2 * curv - 1.0).clamp(min=EPS).sqrt() \
              * (go_norm * curv).clamp(min=EPS)
    oxy_angle = torch.acos((cos_num / cos_den).clamp(-1 + EPS, 1 - EPS))

    return torch.relu(oxy_angle - aperture).mean()


def dag_loss(probe, go_emb, dag_edges):
    """Parent GO term should entail child GO term."""
    if not dag_edges:
        return go_emb.new_tensor(0.0)
    parents = go_emb[[e[0] for e in dag_edges]]
    children = go_emb[[e[1] for e in dag_edges]]
    return meru_loss(probe, children, parents)


# ── Data ──────────────────────────────────────────────────────────────────────

def compute_esm2_features(splits, device):
    model_tag  = ESM2_HF.split("/")[-1]
    cache_path = CACHE_DIR / f"esm2_{model_tag}_go_feats.pt"
    if cache_path.exists():
        print(f"  [esm2] Using cache: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    print(f"  [esm2] Computing {ESM2_HF} features ...")
    from transformers import EsmTokenizer, EsmModel
    tok   = EsmTokenizer.from_pretrained(ESM2_HF)
    model = EsmModel.from_pretrained(ESM2_HF).to(device).eval()

    feats = {}
    for split, data in splits.items():
        seqs  = data["seqs"]
        embs  = []
        with torch.no_grad():
            for i in range(0, len(seqs), 16):
                batch = seqs[i:i+16]
                inp   = tok(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=512).to(device)
                out   = model(**inp)
                mask  = inp["attention_mask"].float().unsqueeze(-1)
                emb   = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
                embs.append(emb.cpu())
        feats[split] = torch.cat(embs)
        print(f"    {split}: {feats[split].shape}")
    del model; torch.cuda.empty_cache()
    torch.save(feats, cache_path)
    return feats


def compute_go_embeddings(vocab, device):
    cache_path = CACHE_DIR / "go_term_embs.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    print(f"  [pubmedbert] Computing GO term embeddings ...")
    from transformers import AutoTokenizer, AutoModel
    tok   = AutoTokenizer.from_pretrained(PUBMEDBERT_HF)
    model = AutoModel.from_pretrained(PUBMEDBERT_HF).to(device).eval()
    names = [f"FUNCTION: {vocab['idx_to_name'][str(i)]}." for i in range(GO_TERMS)]
    embs  = []
    with torch.no_grad():
        for i in range(0, len(names), 64):
            inp = tok(names[i:i+64], return_tensors="pt", padding=True,
                      truncation=True, max_length=128).to(device)
            out = model(**inp)
            embs.append(out.last_hidden_state[:, 0].cpu())
    go_embs = torch.cat(embs)
    del model; torch.cuda.empty_cache()
    torch.save(go_embs, cache_path)
    return go_embs


def load_dag_edges(vocab):
    id2idx   = vocab["id_to_idx"]
    obo_path = CACHE_DIR / "go-basic.obo"
    edges, current_id = [], None
    with open(obo_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("id: GO:"):
                current_id = line.split("id: ")[1]
            elif line.startswith("is_a:") and current_id:
                parent_id = line.split()[1]
                if current_id in id2idx and parent_id in id2idx:
                    edges.append((id2idx[parent_id], id2idx[current_id]))
    return edges


# ── Evaluation ────────────────────────────────────────────────────────────────

def compute_fmax_aupr(sim_np, tgt_np):
    qs  = np.linspace(1, 99, 199)
    thr = np.percentile(sim_np, qs)
    best_f, precs, recs = 0.0, [], []
    for t in thr:
        preds = (sim_np > t).astype(np.float32)
        tp    = (preds * tgt_np).sum(1)
        p     = float((tp / np.maximum(preds.sum(1), 1)).mean())
        r     = float((tp / np.maximum(tgt_np.sum(1), 1)).mean())
        precs.append(p); recs.append(r)
        best_f = max(best_f, 2*p*r/(p+r+1e-8))
    order = np.argsort(recs)
    aupr  = float(np.trapz(np.array(precs)[order], np.array(recs)[order]))
    return best_f, aupr


@torch.no_grad()
def evaluate(probe, seq_feats, go_emb_table, tgt, device):
    probe.eval()
    go_r = probe.encode_text(go_emb_table.to(device)).cpu()
    rows = []
    for i in range(0, len(seq_feats), 256):
        sr  = probe.encode_seq(seq_feats[i:i+256].to(device)).cpu()
        if probe.geometry == "lorentz":
            rows.append(-pairwise_dist(sr, go_r, probe.curv.detach().cpu()))
        else:
            rows.append(sr @ go_r.T)
    sim    = torch.cat(rows).numpy().astype(np.float32)
    tgt_np = tgt.numpy().astype(np.float32)
    return compute_fmax_aupr(sim, tgt_np)


# ── Training ──────────────────────────────────────────────────────────────────

def train_one(label, geometry, lam_meru, lam_dag,
              esm_feats, go_emb_raw, splits, dag_edges, device,
              n_epochs=EPOCHS, proj_dim=PROJ_DIM):

    seq_train = esm_feats["train"]
    tgt_train = splits["train"]["targets"].float()
    seq_val   = esm_feats["validation"]
    tgt_val   = splits["validation"]["targets"].float()

    # Filter empty annotations in training
    valid = tgt_train.sum(1) > 0
    seq_train, tgt_train = seq_train[valid], tgt_train[valid]

    loader = DataLoader(
        TensorDataset(seq_train, tgt_train),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    probe = GOProbeV2(ESM_DIM, BERT_DIM, PROJ_HIDDEN, proj_dim, geometry).to(device)
    opt   = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=1e-4)

    # Pre-encode all 489 GO terms once and keep on device (small, 489×768)
    go_emb_raw_dev = go_emb_raw.to(device)

    best_fmax, best_state = 0.0, None
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        probe.train()
        epoch_loss = 0.0

        # Re-encode GO terms each epoch (text_head changes)
        go_emb = probe.encode_text(go_emb_raw_dev).detach()   # [489, D]

        for seq_b, tgt_b in loader:
            seq_b, tgt_b = seq_b.to(device), tgt_b.to(device)

            seq_emb = probe.encode_seq(seq_b)              # [B, D]
            logits  = probe.similarity(seq_emb, go_emb)   # [B, 489]
            loss    = bce_loss(logits, tgt_b)

            if lam_meru > 0:
                # Sample one positive GO term per protein for MERU
                pos_idx = torch.multinomial(tgt_b.float().clamp(min=1e-6), 1).squeeze(1)
                go_pos  = go_emb[pos_idx]                  # [B, D]
                loss    = loss + lam_meru * meru_loss(probe, seq_emb, go_pos)

            if lam_dag > 0 and dag_edges:
                loss = loss + lam_dag * dag_loss(probe, go_emb, dag_edges)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            opt.step()
            if probe.geometry == "lorentz":
                probe.seq_head.clamp()
                probe.text_head.clamp()

            epoch_loss += loss.item()

        if epoch % 5 == 0 or epoch == n_epochs:
            fmax, aupr = evaluate(probe, seq_val, go_emb_raw, tgt_val, device)
            elapsed = time.time() - t0
            kstr = f"  κ={probe.curv.item():.3f}" if geometry == "lorentz" else ""
            print(f"  [{label}] ep {epoch:3d}/{n_epochs}  "
                  f"loss={epoch_loss/len(loader):.4f}  "
                  f"val_Fmax={fmax:.4f}  val_AUPR={aupr:.4f}{kstr}  "
                  f"({elapsed:.0f}s)", flush=True)
            if fmax > best_fmax:
                best_fmax  = fmax
                best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

    return best_state, best_fmax


# ── Main ──────────────────────────────────────────────────────────────────────

CONDITIONS = [
    # (label,              geometry,   lam_meru, lam_dag, ckpt_name)
    ("Euclidean-v2",       "euclidean", 0.0,      0.0,    "v2_Euclidean.pt"),
    ("Hyp-v2",             "lorentz",   0.0,      0.0,    "v2_Hyp.pt"),
    ("Hyp+MERU-v2",        "lorentz",   0.5,      0.0,    "v2_HyppMERU.pt"),
    ("Hyp+MERU+DAG-v2",    "lorentz",   0.5,      0.5,    "v2_HyppMERUpDAG.pt"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int, default=EPOCHS)
    parser.add_argument("--proj-dim", type=int, default=PROJ_DIM)
    parser.add_argument("--cond",     type=str, default=None,
                        help="Run only this condition label (substring match)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  epochs={args.epochs}  proj_dim={args.proj_dim}  "
          f"batch={BATCH_SIZE}  encoder={ESM2_HF}\n", flush=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print("[1/4] Loading dataset ...")
    splits = torch.load(CACHE_DIR / "protst_go_mf_decoded.pt",
                        map_location="cpu", weights_only=False)
    vocab  = json.load(open(CACHE_DIR / "go_mf_vocab.json"))
    print(f"  train={len(splits['train']['seqs'])}  "
          f"val={len(splits['validation']['seqs'])}  "
          f"test={len(splits['test']['seqs'])}", flush=True)

    print("\n[2/4] Computing ESM2-650M features ...")
    esm_feats = compute_esm2_features(splits, device)

    print("\n[3/4] Computing GO term embeddings ...")
    go_emb_raw = compute_go_embeddings(vocab, device)   # [489, 768]
    print(f"  GO embeddings: {go_emb_raw.shape}")

    print("\n[4/4] Loading GO DAG edges ...")
    dag_edges = load_dag_edges(vocab)
    print(f"  is_a edges within vocab: {len(dag_edges)}\n")

    tgt_test = splits["test"]["targets"]
    seq_test = esm_feats["test"]

    # ── Train all conditions ──────────────────────────────────────────────────
    results = {}
    for label, geometry, lam_meru, lam_dag, ckpt_name in CONDITIONS:
        if args.cond and args.cond not in label:
            continue

        print(f"{'='*60}\n{label}\n{'='*60}", flush=True)
        best_state, best_val_fmax = train_one(
            label, geometry, lam_meru, lam_dag,
            esm_feats, go_emb_raw, splits, dag_edges, device,
            n_epochs=args.epochs, proj_dim=args.proj_dim,
        )

        # Save checkpoint
        ckpt_path = CKPT_DIR / ckpt_name
        torch.save({"state_dict": best_state,
                    "label": label, "geometry": geometry,
                    "proj_dim": args.proj_dim,
                    "proj_hidden": PROJ_HIDDEN,
                    "esm_model": ESM2_HF}, ckpt_path)

        # Test evaluation
        probe = GOProbeV2(ESM_DIM, BERT_DIM, PROJ_HIDDEN, args.proj_dim, geometry)
        probe.load_state_dict(best_state)
        probe.to(device)
        fmax, aupr = evaluate(probe, seq_test, go_emb_raw, tgt_test, device)
        print(f"\n  Best val Fmax={best_val_fmax:.4f}  "
              f"Test Fmax={fmax:.4f}  Test AUPR={aupr:.4f}\n", flush=True)
        results[label] = {"fmax": round(fmax, 4), "aupr": round(aupr, 4)}

    # ── Summary ───────────────────────────────────────────────────────────────
    if results:
        sep = "=" * 52
        print(f"\n{sep}")
        print(f"  {'Model':<24}  {'Fmax':>7}  {'AUPR':>7}")
        print("-" * 52)

        # v1 reference
        print(f"  {'--- v1 reference (InfoNCE, 8M) ---':<24}")
        v1 = {"Euclidean": (0.0748, 0.0318), "Hyp+MERU λ=0.5": (0.0828, 0.0337)}
        for lbl, (f, a) in v1.items():
            print(f"  {lbl:<24}  {f:>7.4f}  {a:>7.4f}")
        print()

        for lbl, r in results.items():
            print(f"  {lbl:<24}  {r['fmax']:>7.4f}  {r['aupr']:>7.4f}")
        print(sep)

    out = RESULTS_DIR / "results_v2.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
