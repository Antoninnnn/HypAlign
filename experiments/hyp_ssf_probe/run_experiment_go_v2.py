#!/usr/bin/env python3
"""
Hyperbolic SSF Probe v2 — Clean geometry comparison
====================================================
Fixes three confounds from v1:
  1. Encoder:   ESM2-650M frozen (vs 8M) — stronger protein features
  2. Projector: 2-layer MLP (vs linear) — nonlinear capacity before exp_map0
  3. Loss:      BCE or MulSupCon over all 489 GO terms (vs InfoNCE)

Four geometry conditions are otherwise identical, so the only variable
is the embedding space geometry.

Conditions (run twice, once per loss):
  1. Euclidean      — MLP → L2-norm, cosine sim
  2. Hyp            — MLP → exp_map0, Lorentz dist
  3. Hyp+MERU λ=0.5 — + entailment cone loss (GO→protein)
  4. Hyp+MERU+DAG   — + DAG is_a edges (parent→child GO)

Loss options:
  --loss bce         Binary cross-entropy over all 489 labels (default)
  --loss mulsupcon   MulSupCon (Zhang & Wu, AAAI 2024): per-label softmax
                     contrastive loss — directly optimises the retrieval ranking

Run:
    conda activate hypalign
    cd experiments/hyp_ssf_probe
    python -u run_experiment_go_v2.py --loss bce
    python -u run_experiment_go_v2.py --loss mulsupcon
"""

from __future__ import annotations

import argparse
import json
import math
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
NEUML_HF      = "NeuML/pubmedbert-base-embeddings"   # sentence-embedding model; use --pooling mean

GO_TERMS    = 489
ESM_DIM     = 1280
BERT_DIM    = 768
PROJ_HIDDEN = 512
PROJ_DIM    = 256
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

def pairwise_dist(x, y, curv):
    xt = _time(x, curv).unsqueeze(1)
    yt = _time(y, curv).unsqueeze(0)
    inner = x @ y.T - xt * yt
    return torch.acosh((-curv * inner).clamp(min=1.0 + EPS)) / curv.sqrt()

# ── Projector heads (MLP) ─────────────────────────────────────────────────────

class MLPLorentzHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),   # bias: lets head shift distribution freely
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
            nn.Linear(in_dim, hidden_dim, bias=True),   # bias: symmetric with LorentzHead
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
            self.log_curv   = nn.Parameter(torch.tensor(math.log(CURV_INIT))) \
                              if learn_curv else \
                              torch.tensor(math.log(CURV_INIT))
            self.logit_bias = nn.Parameter(torch.zeros(1))  # fixes always-negative BCE logit
            self.seq_head   = MLPLorentzHead(seq_dim,  hidden_dim, proj_dim)
            self.text_head  = MLPLorentzHead(text_dim, hidden_dim, proj_dim)
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
            # logit_bias shifts logits so positives can be > 0 and negatives < 0.
            # Without it, -distance * scale ≤ 0 always, breaking BCE calibration.
            return self.logit_bias - pairwise_dist(seq_emb, go_emb, self.curv) \
                   * self.log_temp.exp()
        else:
            return seq_emb @ go_emb.T * self.log_temp.exp()


# ── Losses ────────────────────────────────────────────────────────────────────

def bce_loss(logits, targets):
    """Dense multi-label BCE over all 489 GO terms."""
    return F.binary_cross_entropy_with_logits(logits, targets.float())


def mulsupcon_loss(logits, targets):
    """
    MulSupCon: Zhang & Wu, AAAI 2024.

    For every positive (protein_i, GO_j) pair, compute the softmax cross-entropy
    of ranking GO_j first among all 489 GO terms.  Normalise by the total number
    of positive pairs in the batch (Eq. 4/5 in the paper).  The 1/|y^(i)|
    per-sample weight is intentionally omitted — ablation Table 8 shows it hurts.

    Directly optimises the retrieval ranking metric (Fmax/AUPR), unlike BCE which
    optimises per-label calibration independently.
    """
    log_softmax = F.log_softmax(logits, dim=1)          # [B, 489]
    n_pos = targets.float().sum().clamp(min=1)
    return -(log_softmax * targets.float()).sum() / n_pos


def meru_loss(probe, seq_emb, go_emb):
    """Entailment cone: GO term (general) should entail protein (specific)."""
    curv   = probe.curv
    sqrt_c = curv.sqrt()
    r_min  = 1.0 / sqrt_c

    go_norm = go_emb.norm(dim=-1).clamp(min=EPS)
    aperture = torch.asin((2 * r_min / go_norm).clamp(max=1.0 - EPS))

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
    parents  = go_emb[[e[0] for e in dag_edges]]
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
        seqs = data["seqs"]
        embs = []
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


def compute_go_embeddings(vocab, device, text_model_hf=PUBMEDBERT_HF, pooling="cls"):
    model_tag  = text_model_hf.replace("/", "_").replace("-", "_")
    cache_path = CACHE_DIR / f"go_term_embs_{model_tag}_{pooling}.pt"
    if cache_path.exists():
        t = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"  [text] Using cache: {cache_path.name}  {tuple(t.shape)}")
        return t

    print(f"  [text] Computing GO embeddings via {text_model_hf}  pooling={pooling} ...")
    from transformers import AutoTokenizer, AutoModel
    tok   = AutoTokenizer.from_pretrained(text_model_hf)
    model = AutoModel.from_pretrained(text_model_hf).to(device).eval()
    names = [f"FUNCTION: {vocab['idx_to_name'][str(i)]}." for i in range(GO_TERMS)]
    embs  = []
    with torch.no_grad():
        for i in range(0, len(names), 64):
            inp = tok(names[i:i+64], return_tensors="pt", padding=True,
                      truncation=True, max_length=128).to(device)
            out = model(**inp)
            if pooling == "mean":
                mask = inp["attention_mask"].float().unsqueeze(-1)
                vec  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            else:
                vec = out.last_hidden_state[:, 0]
            embs.append(vec.cpu())
    go_embs = torch.cat(embs)
    del model; torch.cuda.empty_cache()
    torch.save(go_embs, cache_path)
    print(f"  Saved {cache_path.name}  {tuple(go_embs.shape)}")
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
        sr = probe.encode_seq(seq_feats[i:i+256].to(device)).cpu()
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
              loss_type="bce", n_epochs=EPOCHS, proj_dim=PROJ_DIM):

    seq_train = esm_feats["train"]
    tgt_train = splits["train"]["targets"].float()
    seq_val   = esm_feats["validation"]
    tgt_val   = splits["validation"]["targets"].float()

    valid = tgt_train.sum(1) > 0
    seq_train, tgt_train = seq_train[valid], tgt_train[valid]

    loader = DataLoader(
        TensorDataset(seq_train, tgt_train),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    probe = GOProbeV2(ESM_DIM, BERT_DIM, PROJ_HIDDEN, proj_dim, geometry).to(device)
    opt   = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=1e-4)

    go_emb_raw_dev = go_emb_raw.to(device)   # [489, 768] frozen PubMedBERT outputs

    best_fmax, best_state = 0.0, None
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        probe.train()
        epoch_loss = 0.0

        for seq_b, tgt_b in loader:
            seq_b, tgt_b = seq_b.to(device), tgt_b.to(device)

            # Encode GO terms inside the step so text_head receives gradients.
            # 489 × MLP(768→512→256) is fast relative to seq processing.
            go_emb  = probe.encode_text(go_emb_raw_dev)    # [489, D]
            seq_emb = probe.encode_seq(seq_b)               # [B, D]
            logits  = probe.similarity(seq_emb, go_emb)     # [B, 489]

            if loss_type == "mulsupcon":
                loss = mulsupcon_loss(logits, tgt_b)
            else:
                loss = bce_loss(logits, tgt_b)

            if lam_meru > 0:
                pos_idx = torch.multinomial(tgt_b.float().clamp(min=1e-6), 1).squeeze(1)
                go_pos  = go_emb[pos_idx]
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
            if geometry == "lorentz":
                kstr = (f"  κ={probe.curv.item():.3f}"
                        f"  bias={probe.logit_bias.item():.3f}"
                        f"  seq_α={probe.seq_head.log_alpha.exp().item():.4f}"
                        f"  txt_α={probe.text_head.log_alpha.exp().item():.4f}")
            else:
                kstr = ""
            print(f"  [{label}] ep {epoch:3d}/{n_epochs}  "
                  f"loss={epoch_loss/len(loader):.4f}  "
                  f"val_Fmax={fmax:.4f}  val_AUPR={aupr:.4f}{kstr}  "
                  f"({elapsed:.0f}s)", flush=True)
            if fmax > best_fmax:
                best_fmax  = fmax
                best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

    return best_state, best_fmax


# ── Conditions ────────────────────────────────────────────────────────────────

CONDITIONS = [
    # (label_suffix,    geometry,    lam_meru, lam_dag)
    ("Euclidean",       "euclidean", 0.0,      0.0),
    ("Hyp",             "lorentz",   0.0,      0.0),
    ("Hyp+MERU",        "lorentz",   0.5,      0.0),
    ("Hyp+MERU+DAG",    "lorentz",   0.5,      0.5),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss",       type=str, default="bce",
                        choices=["bce", "mulsupcon"],
                        help="bce: binary cross-entropy | mulsupcon: Zhang & Wu AAAI-24")
    parser.add_argument("--text-model", type=str, default=PUBMEDBERT_HF,
                        help="HuggingFace text encoder for GO term embeddings. "
                             "Use NeuML/pubmedbert-base-embeddings with --pooling mean.")
    parser.add_argument("--pooling",    type=str, default="cls", choices=["cls", "mean"],
                        help="Pooling for text encoder: cls (default) or mean")
    parser.add_argument("--epochs",     type=int, default=EPOCHS)
    parser.add_argument("--proj-dim",   type=int, default=PROJ_DIM)
    parser.add_argument("--cond",       type=str, default=None,
                        help="Run only conditions whose label contains this substring")
    parser.add_argument("--run-tag",    type=str, default="",
                        help="Suffix appended to result/checkpoint filenames, "
                             "e.g. '_bias' → results_v2_bce_bias.json")
    args = parser.parse_args()

    pfx = ("msc" if args.loss == "mulsupcon" else "bce") + args.run_tag

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  loss={args.loss}  epochs={args.epochs}  "
          f"proj_dim={args.proj_dim}  batch={BATCH_SIZE}\n"
          f"  seq_encoder:  {ESM2_HF}\n"
          f"  text_encoder: {args.text_model}  pooling={args.pooling}\n",
          flush=True)

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
    go_emb_raw = compute_go_embeddings(vocab, device, args.text_model, args.pooling)
    print(f"  GO embeddings: {go_emb_raw.shape}")

    print("\n[4/4] Loading GO DAG edges ...")
    dag_edges = load_dag_edges(vocab)
    print(f"  is_a edges within vocab: {len(dag_edges)}\n")

    tgt_test = splits["test"]["targets"]
    seq_test = esm_feats["test"]

    # ── Train all conditions ──────────────────────────────────────────────────
    results = {}
    for label_suffix, geometry, lam_meru, lam_dag in CONDITIONS:
        if args.cond and args.cond not in label_suffix:
            continue

        label     = f"{label_suffix}-v2-{pfx}"
        ckpt_name = f"v2_{pfx}_{label_suffix.replace('+', 'p').replace(' ', '_')}.pt"

        print(f"{'='*60}\n{label}\n{'='*60}", flush=True)
        best_state, best_val_fmax = train_one(
            label, geometry, lam_meru, lam_dag,
            esm_feats, go_emb_raw, splits, dag_edges, device,
            loss_type=args.loss,
            n_epochs=args.epochs, proj_dim=args.proj_dim,
        )

        ckpt_path = CKPT_DIR / ckpt_name
        torch.save({"state_dict": best_state,
                    "label": label, "geometry": geometry,
                    "loss_type": args.loss,
                    "proj_dim": args.proj_dim,
                    "proj_hidden": PROJ_HIDDEN,
                    "esm_model": ESM2_HF}, ckpt_path)

        probe = GOProbeV2(ESM_DIM, BERT_DIM, PROJ_HIDDEN, args.proj_dim, geometry)
        probe.load_state_dict(best_state)
        probe.to(device)
        fmax, aupr = evaluate(probe, seq_test, go_emb_raw, tgt_test, device)
        print(f"\n  Best val Fmax={best_val_fmax:.4f}  "
              f"Test Fmax={fmax:.4f}  Test AUPR={aupr:.4f}\n", flush=True)
        results[label] = {"fmax": round(fmax, 4), "aupr": round(aupr, 4)}

    # ── Summary ───────────────────────────────────────────────────────────────
    if results:
        sep = "=" * 56
        print(f"\n{sep}")
        print(f"  {'Model':<28}  {'Fmax':>7}  {'AUPR':>7}")
        print("-" * 56)
        print(f"  {'--- v1 reference (InfoNCE, ESM2-8M) ---'}")
        for lbl, (f, a) in [("Euclidean", (0.0748, 0.0318)),
                              ("Hyp+MERU",  (0.0828, 0.0337))]:
            print(f"  {lbl:<28}  {f:>7.4f}  {a:>7.4f}")
        print()
        for lbl, r in results.items():
            print(f"  {lbl:<28}  {r['fmax']:>7.4f}  {r['aupr']:>7.4f}")
        print(sep)

    out = RESULTS_DIR / f"results_v2_{pfx}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
