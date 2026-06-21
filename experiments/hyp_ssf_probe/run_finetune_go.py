#!/usr/bin/env python3
"""
Strong Euclidean baseline: fine-tuned ESM-2 for GO-MF function prediction.

Key improvements over the frozen probe (v1):
  1. ESM-2 150M fine-tuned end-to-end (vs ESM-2 8M frozen)
  2. Direct multi-label BCE loss (vs InfoNCE)
     → no false-negative problem from multi-label sampling
     → gradients flow to ESM-2 directly from GO supervision
  3. Asymmetric loss (ASL) for long-tailed GO class distribution
  4. Positive class reweighting for severe imbalance
  5. Separate LR: backbone 5e-5, head 3e-4  with warmup + cosine schedule

Architecture:
  ESM-2 150M → mean-pool → Linear(640, 489) → sigmoid → BCE

Evaluation: same Fmax + AUPR as v1 (protein-centric, quantile thresholds).

Run:
    conda activate pannot-infer
    cd /home/yining_yang/NLP/hyperbolic/experiments/hyp_ssf_probe
    python run_finetune_go.py
"""

from __future__ import annotations
import json, math, os, sys, time
from pathlib import Path

# Avoid CUDA memory allocator assertion failures on some driver versions
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import EsmTokenizer, EsmModel
from transformers import get_cosine_schedule_with_warmup

REPO_ROOT  = Path(__file__).resolve().parent
CACHE_DIR  = REPO_ROOT / "cache"
CKPT_DIR   = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
ESM_MODEL   = "facebook/esm2_t30_150M_UR50D"   # 150M params, 640-dim
N_GO        = 489
PROJ_DIM    = 640        # same as ESM-2 150M hidden dim (no bottleneck)
N_EPOCHS    = 50
BATCH_SIZE   = 8           # micro-batch; effective batch = 8 * GRAD_ACCUM
GRAD_ACCUM   = 4           # effective batch = 32
BACKBONE_LR  = 5e-5
HEAD_LR      = 3e-4
WEIGHT_DECAY = 1e-4
MAX_LEN      = 512
WARMUP_FRAC = 0.05       # 5% of total steps
GRAD_CLIP   = 1.0
POS_WEIGHT_CAP = 10.0    # cap inverse-frequency weights

# Asymmetric Loss hyperparameters (ASL)
ASL_GAMMA_POS = 0        # focusing on positives (0 = no focusing)
ASL_GAMMA_NEG = 4        # hard down-weighting of easy negatives
ASL_CLIP      = 0.05     # probability margin shift for negatives


# ── Asymmetric Loss ───────────────────────────────────────────────────────────

def asymmetric_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """ASL for long-tailed multi-label classification.

    Decouples the focusing exponents for positive (γ+) and negative (γ-)
    samples, and applies a probability shift (clip) on negatives to avoid
    the easy-negative domination problem in severe class imbalance.

    Zamir et al., "Asymmetric Loss For Multi-Label Classification" (ICCV 2021).
    """
    prob = torch.sigmoid(logits)

    # Positive branch: standard focal weighting
    loss_pos = (1 - prob).pow(ASL_GAMMA_POS) * targets \
               * torch.log(prob.clamp(min=1e-7))

    # Negative branch: shifted probability + asymmetric focus
    prob_neg = (prob - ASL_CLIP).clamp(min=0)   # shift negatives down
    loss_neg = prob_neg.pow(ASL_GAMMA_NEG) * (1 - targets) \
               * torch.log((1 - prob_neg).clamp(min=1e-7))

    return -(loss_pos + loss_neg).mean()


# ── Dataset ───────────────────────────────────────────────────────────────────

class GODataset(Dataset):
    def __init__(self, seqs: list[str], targets: torch.Tensor,
                 filter_empty: bool = True):
        if filter_empty:
            keep = (targets.sum(1) > 0).nonzero(as_tuple=True)[0]
            seqs    = [seqs[i] for i in keep.tolist()]
            targets = targets[keep]
        self.seqs    = seqs
        self.targets = targets.float()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], self.targets[i]


def make_loader(split_data: dict, batch_size: int, shuffle: bool,
                filter_empty: bool = True) -> DataLoader:
    ds = GODataset(split_data["seqs"], split_data["targets"], filter_empty)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False,
                      collate_fn=lambda b: (
                          [x[0] for x in b],
                          torch.stack([x[1] for x in b])
                      ))


# ── Model ─────────────────────────────────────────────────────────────────────

class GOClassifier(nn.Module):
    """ESM-2 backbone + linear classification head for 489 GO-MF terms."""

    def __init__(self, backbone: EsmModel, esm_dim: int = 640):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(esm_dim, N_GO)
        nn.init.normal_(self.head.weight, std=esm_dim**-0.5)
        nn.init.zeros_(self.head.bias)

    def encode(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids,
                            attention_mask=attention_mask)
        mask = attention_mask.float().unsqueeze(-1)   # [B, L, 1]
        emb  = (out.last_hidden_state * mask).sum(1) \
               / mask.sum(1).clamp(min=1)              # [B, esm_dim]
        return emb

    def forward(self, input_ids, attention_mask):
        return self.head(self.encode(input_ids, attention_mask))  # [B, 489]


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model: GOClassifier, loader: DataLoader,
            tokenizer: EsmTokenizer, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_targets = [], []
    for seqs, targets in loader:
        enc = tokenizer(seqs, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LEN)
        logits = model(
            input_ids      = enc["input_ids"].to(device),
            attention_mask = enc["attention_mask"].to(device),
        )
        all_logits.append(logits.cpu())
        all_targets.append(targets)
    return (torch.cat(all_logits).numpy(),
            torch.cat(all_targets).numpy())


def fmax_aupr(logits_np: np.ndarray, tgt_np: np.ndarray):
    """Protein-centric Fmax and AUPR over 199 quantile thresholds."""
    qs = np.linspace(1, 99, 199)
    thresholds = np.percentile(logits_np, qs)
    best_f = 0.0
    precs, recs = [], []
    for t in thresholds:
        preds = (logits_np > t).astype(np.float32)
        tp    = (preds * tgt_np).sum(1)
        prec  = float((tp / np.maximum(preds.sum(1), 1)).mean())
        rec   = float((tp / np.maximum(tgt_np.sum(1), 1)).mean())
        precs.append(prec)
        recs.append(rec)
        best_f = max(best_f, 2 * prec * rec / (prec + rec + 1e-8))
    order = np.argsort(recs)
    aupr  = float(np.trapz(np.array(precs)[order], np.array(recs)[order]))
    return best_f, aupr


def retrieval_metrics(logits_np, tgt_np, ks=(1, 5, 10)):
    """Protein→GO recall@k and precision@k."""
    order = np.argsort(-logits_np, axis=1)
    out = {}
    for k in ks:
        topk = order[:, :k]
        tp_k = np.array([tgt_np[i, topk[i]].sum() for i in range(len(tgt_np))])
        n_pos = tgt_np.sum(1)
        out[f"R@{k}"] = float((tp_k / np.maximum(n_pos, 1)).mean())
        out[f"P@{k}"] = float((tp_k / k).mean())
    return out


# ── Training ──────────────────────────────────────────────────────────────────

def train(model: GOClassifier, tokenizer: EsmTokenizer,
          train_loader: DataLoader, val_loader: DataLoader,
          pos_weight: torch.Tensor, device, label: str):

    # Separate learning rates for backbone and classification head
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
        {"params": model.head.parameters(),     "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    # n_steps counts optimizer steps (after gradient accumulation)
    n_opt_steps = math.ceil(len(train_loader) / GRAD_ACCUM) * N_EPOCHS
    n_warmup    = int(WARMUP_FRAC * n_opt_steps)
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=n_warmup, num_training_steps=n_opt_steps
    )

    best_val_fmax, best_state = 0.0, None
    t0 = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        epoch_loss  = 0.0
        optimizer.zero_grad()

        for step, (seqs, targets) in enumerate(train_loader):
            targets = targets.to(device)

            enc = tokenizer(seqs, return_tensors="pt", padding=True,
                            truncation=True, max_length=MAX_LEN)
            logits = model(
                input_ids      = enc["input_ids"].to(device),
                attention_mask = enc["attention_mask"].to(device),
            )                                   # [B, 489]

            loss = asymmetric_loss(logits, targets) / GRAD_ACCUM
            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Validation Fmax every 5 epochs
        if epoch % 5 == 0 or epoch == N_EPOCHS:
            val_logits, val_tgt = predict(model, val_loader, tokenizer, device)
            val_fmax, val_aupr  = fmax_aupr(val_logits, val_tgt)

            if val_fmax > best_val_fmax:
                best_val_fmax = val_fmax
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}

            elapsed = time.time() - t0
            print(f"  [{label}] ep {epoch:3d}/{N_EPOCHS}  "
                  f"loss={epoch_loss/len(train_loader):.4f}  "
                  f"val_Fmax={val_fmax:.4f}  val_AUPR={val_aupr:.4f}  "
                  f"({elapsed:.0f}s)")

    return best_state, best_val_fmax


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  ESM-2: {ESM_MODEL}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
              flush=True)
    print()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data ...")
    splits = torch.load(CACHE_DIR / "protst_go_mf_decoded.pt",
                        map_location="cpu", weights_only=False)

    train_loader = make_loader(splits["train"],      BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(splits["validation"], BATCH_SIZE, shuffle=False,
                               filter_empty=False)
    test_loader  = make_loader(splits["test"],       BATCH_SIZE, shuffle=False,
                               filter_empty=False)

    # Positive class weights from training labels (for reference / fallback)
    tgt_train = splits["train"]["targets"].float()
    pos_freq  = (tgt_train > 0.5).float().mean(0)              # [489]
    pos_weight = ((1 - pos_freq) / pos_freq.clamp(min=1e-6)).clamp(max=POS_WEIGHT_CAP)

    n_train = sum(1 for s in splits["train"]["seqs"]
                  if splits["train"]["targets"][
                      splits["train"]["seqs"].index(s)].sum() > 0) \
              if False else len(train_loader.dataset)
    print(f"  train={len(train_loader.dataset)}  "
          f"val={len(val_loader.dataset)}  "
          f"test={len(test_loader.dataset)}")
    print(f"  GO terms: {N_GO}  "
          f"avg pos/protein (train): "
          f"{(tgt_train > 0.5).float().sum(1).mean():.1f}\n")

    # ── Load ESM-2 ─────────────────────────────────────────────────────────────
    print(f"Loading {ESM_MODEL} ...")
    tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL)
    backbone  = EsmModel.from_pretrained(ESM_MODEL)
    backbone.gradient_checkpointing_enable()   # saves ~50% activation memory
    esm_dim   = backbone.config.hidden_size    # 640 for 150M
    print(f"  ESM-2 hidden dim: {esm_dim}  "
          f"params: {sum(p.numel() for p in backbone.parameters())/1e6:.0f}M")
    print(f"  Gradient checkpointing: enabled")
    print(f"  Effective batch size: {BATCH_SIZE} × {GRAD_ACCUM} = "
          f"{BATCH_SIZE * GRAD_ACCUM}\n")

    # ── Single condition: strong Euclidean baseline ────────────────────────────
    label = f"ESM2-150M-FT"
    print(f"── {label} ──────────────────────────────")
    print(f"  Loss: Asymmetric (γ+={ASL_GAMMA_POS}, γ-={ASL_GAMMA_NEG}, "
          f"clip={ASL_CLIP})")
    print(f"  LR: backbone={BACKBONE_LR}  head={HEAD_LR}  "
          f"epochs={N_EPOCHS}  batch={BATCH_SIZE}\n")

    model = GOClassifier(backbone, esm_dim).to(device)

    best_state, best_val_fmax = train(
        model, tokenizer, train_loader, val_loader, pos_weight, device, label
    )

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\n  Best val Fmax = {best_val_fmax:.4f}")
    print("  Running test evaluation ...")

    model.load_state_dict(best_state)
    model.to(device)

    test_logits, test_tgt = predict(model, test_loader, tokenizer, device)
    test_fmax, test_aupr  = fmax_aupr(test_logits, test_tgt)
    ret = retrieval_metrics(test_logits, test_tgt)

    # IC for wFmax
    tgt_np   = test_tgt
    ic_np    = -np.log((pos_freq.numpy() + 1e-6))
    qs       = np.linspace(1, 99, 199)
    tholds   = np.percentile(test_logits, qs)
    best_wf  = 0.0
    for t in tholds:
        preds  = (test_logits > t).astype(np.float32)
        tp_w   = (preds * tgt_np * ic_np).sum(1)
        pred_w = (preds * ic_np).sum(1)
        true_w = (tgt_np * ic_np).sum(1)
        wp = float((tp_w / np.maximum(pred_w, 1e-8)).mean())
        wr = float((tp_w / np.maximum(true_w, 1e-8)).mean())
        best_wf = max(best_wf, 2 * wp * wr / (wp + wr + 1e-8))

    ndcg_scores, ap_scores = [], []
    full_order = np.argsort(-test_logits, axis=1)
    for i in range(len(test_logits)):
        true_idx = set(np.where(tgt_np[i] > 0.5)[0])
        if not true_idx: continue
        o = full_order[i]
        dcg  = sum(1/math.log2(r+2) for r, j in enumerate(o[:10]) if j in true_idx)
        idcg = sum(1/math.log2(r+2) for r in range(min(len(true_idx), 10)))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
        hits, ap = 0, 0.0
        for r, j in enumerate(o):
            if j in true_idx:
                hits += 1; ap += hits / (r + 1)
        ap_scores.append(ap / len(true_idx))

    results = {
        "fmax":   round(test_fmax, 4),
        "aupr":   round(test_aupr, 4),
        "wfmax":  round(best_wf,   4),
        "ndcg10": round(float(np.mean(ndcg_scores)), 4),
        "map":    round(float(np.mean(ap_scores)),   4),
        **{k: round(v, 4) for k, v in ret.items()},
    }

    # ── Print table ──────────────────────────────────────────────────────────
    sep = "=" * 78
    print(f"\n{sep}")
    print(f"  {'Model':<22}  Fmax    AUPR   wFmax  nDCG@10    MAP")
    print("-" * 78)
    print(f"  {'ESM2-8M frozen (v1)':<22}  0.0748  0.0318  0.0614   0.0881  0.0877"
          "  ← previous")
    print(f"  {label:<22}  "
          f"{results['fmax']:.4f}  {results['aupr']:.4f}  "
          f"{results['wfmax']:.4f}   {results['ndcg10']:.4f}  "
          f"{results['map']:.4f}  ← fine-tuned")
    print(sep)
    print(f"\n  Protein→GO retrieval:")
    for k in [1, 5, 10]:
        print(f"    R@{k}={results[f'R@{k}']:.4f}  P@{k}={results[f'P@{k}']:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    ckpt_path = CKPT_DIR / f"probe_{label.replace(' ', '_')}.pt"
    torch.save({
        "state_dict": best_state,
        "esm_model":  ESM_MODEL,
        "results":    results,
    }, ckpt_path)
    print(f"\n  Checkpoint saved: {ckpt_path}")

    out_path = RESULTS_DIR / f"results_{label.replace(' ', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved:    {out_path}")


if __name__ == "__main__":
    main()
