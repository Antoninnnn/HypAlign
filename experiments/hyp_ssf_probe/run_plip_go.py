#!/usr/bin/env python3
"""
PLIP-style supervised baseline for ProtST GO.

This matches the stronger baseline discussed with Yuxuan:
  frozen ESM-2 protein embeddings + frozen PubMedBERT sentence embeddings
  -> normalized projection heads -> all protein x GO logits
  -> BCEWithLogitsLoss(pos_weight=...).

The script intentionally trains only the projection heads. It assumes the label
text order matches the target-vector order.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parent
CACHE_DIR = REPO_ROOT / "cache"
CKPT_DIR = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"
CKPT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_ESM_MODEL = "facebook/esm2_t33_650M_UR50D"
DEFAULT_TEXT_MODEL = "NeuML/pubmedbert-base-embeddings"
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        out = self.net(x)
        return F.normalize(out, dim=-1)


class ProteinGoPLIP(nn.Module):
    def __init__(self, protein_dim: int, text_dim: int,
                 proj_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.protein_projector = ProjectionHead(protein_dim, hidden_dim, proj_dim)
        self.term_projector = ProjectionHead(text_dim, hidden_dim, proj_dim)
        self.log_logit_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))

    def forward(self, protein_embeddings, term_embeddings):
        protein_proj = self.protein_projector(protein_embeddings)
        term_proj = self.term_projector(term_embeddings)
        return protein_proj @ term_proj.T * self.log_logit_scale.exp()


def model_tag(model_name: str) -> str:
    return model_name.split("/")[-1].replace("/", "_")


def slugify(label: str) -> str:
    out = []
    for ch in label:
        out.append(ch if ch.isalnum() or ch in "-_." else "_")
    return "".join(out).strip("_")


def load_splits(splits_cache: str):
    path = Path(splits_cache)
    if not path.is_absolute():
        path = CACHE_DIR / splits_cache
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it first with scripts/build_go_namespace_cache.py."
        )
    return torch.load(path, map_location="cpu", weights_only=False), path


def load_term_texts(term_text_file: str | None, vocab_file: str, expected_terms: int):
    if term_text_file:
        texts = [line.strip() for line in Path(term_text_file).read_text().splitlines()
                 if line.strip()]
    else:
        vocab_path = Path(vocab_file)
        if not vocab_path.is_absolute():
            vocab_path = CACHE_DIR / vocab_file
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"{vocab_path} not found. Pass --term-text-file or --vocab-file."
            )
        vocab = json.loads(vocab_path.read_text())
        texts = [f"FUNCTION: {name}." for name in vocab["go_names"]]
    if len(texts) != expected_terms:
        raise ValueError(f"Expected {expected_terms} GO term texts, found {len(texts)}")
    return texts


@torch.no_grad()
def mean_pool_text(model, enc):
    out = model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    return (out * mask).sum(1) / mask.sum(1).clamp(min=1)


def compute_term_embeddings(texts, text_model: str, device, batch_size: int,
                            cache_name: str):
    cache_path = Path(cache_name)
    if not cache_path.is_absolute():
        cache_path = CACHE_DIR / cache_name
    if cache_path.exists():
        print(f"  [text] using cache: {cache_path}")
        embs = torch.load(cache_path, map_location="cpu", weights_only=True)
        if embs.shape[0] != len(texts):
            raise ValueError(
                f"{cache_path} has {embs.shape[0]} rows but {len(texts)} texts were provided"
            )
        return embs

    print(f"  [text] encoding {len(texts)} terms with {text_model}")
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(text_model)
    model = AutoModel.from_pretrained(text_model).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=128).to(device)
            out.append(mean_pool_text(model, enc).cpu())
    embs = F.normalize(torch.cat(out), dim=-1)
    torch.save(embs, cache_path)
    print(f"  [text] saved: {cache_path} {tuple(embs.shape)}")
    return embs


@torch.no_grad()
def compute_protein_embeddings(splits, esm_model: str, device, batch_size: int,
                               max_len: int, cache_name: str):
    cache_path = Path(cache_name)
    if not cache_path.is_absolute():
        cache_path = CACHE_DIR / cache_name
    if cache_path.exists():
        print(f"  [esm] using cache: {cache_path}")
        feats = torch.load(cache_path, map_location="cpu", weights_only=True)
        for split, data in splits.items():
            if split not in feats:
                raise ValueError(f"{cache_path} is missing split {split}")
            if feats[split].shape[0] != len(data["seqs"]):
                raise ValueError(
                    f"{cache_path}:{split} has {feats[split].shape[0]} rows "
                    f"but split has {len(data['seqs'])} sequences"
                )
        return feats

    print(f"  [esm] computing frozen embeddings with {esm_model}")
    from transformers import EsmModel, EsmTokenizer

    tok = EsmTokenizer.from_pretrained(esm_model)
    model = EsmModel.from_pretrained(esm_model).to(device).eval()
    feats = {}
    for split, data in splits.items():
        seqs = data["seqs"]
        rows = []
        for i in range(0, len(seqs), batch_size):
            enc = tok(seqs[i:i + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len).to(device)
            out = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            rows.append(pooled.cpu())
            if i % max(batch_size * 10, 1) == 0:
                print(f"    {split}: {i}/{len(seqs)}", flush=True)
        feats[split] = torch.cat(rows)
        print(f"    {split}: {tuple(feats[split].shape)}")
    torch.save(feats, cache_path)
    print(f"  [esm] saved: {cache_path}")
    return feats


def fmax_aupr(scores, targets):
    qs = np.linspace(1, 99, 199)
    thresholds = np.percentile(scores, qs)
    best_f = 0.0
    precs, recs = [], []
    for t in thresholds:
        pred = (scores > t).astype(np.float32)
        tp = (pred * targets).sum(1)
        prec = float((tp / np.maximum(pred.sum(1), 1)).mean())
        rec = float((tp / np.maximum(targets.sum(1), 1)).mean())
        precs.append(prec)
        recs.append(rec)
        best_f = max(best_f, 2 * prec * rec / (prec + rec + 1e-8))
    order = np.argsort(recs)
    aupr = float(np.trapz(np.asarray(precs)[order], np.asarray(recs)[order]))
    return best_f, aupr


def wfmax(scores, targets, train_targets):
    pos_freq = (train_targets > 0.5).mean(0)
    ic = -np.log(np.clip(pos_freq, 1e-6, 1.0))
    best = 0.0
    for t in np.percentile(scores, np.linspace(1, 99, 199)):
        pred = (scores > t).astype(np.float32)
        tp_w = (pred * targets * ic).sum(1)
        pred_w = (pred * ic).sum(1)
        true_w = (targets * ic).sum(1)
        prec = float((tp_w / np.maximum(pred_w, 1e-8)).mean())
        rec = float((tp_w / np.maximum(true_w, 1e-8)).mean())
        best = max(best, 2 * prec * rec / (prec + rec + 1e-8))
    return best


def ndcg_map(scores, targets, k=10):
    order = np.argsort(-scores, axis=1)
    ndcg_scores, ap_scores = [], []
    for i in range(len(scores)):
        true_idx = set(np.where(targets[i] > 0.5)[0])
        if not true_idx:
            continue
        row = order[i]
        dcg = sum(1.0 / math.log2(r + 2) for r, j in enumerate(row[:k]) if j in true_idx)
        idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(true_idx), k)))
        ndcg_scores.append(dcg / idcg if idcg else 0.0)
        hits, ap = 0, 0.0
        for r, j in enumerate(row):
            if j in true_idx:
                hits += 1
                ap += hits / (r + 1)
        ap_scores.append(ap / len(true_idx))
    return float(np.mean(ndcg_scores)), float(np.mean(ap_scores))


def protein_to_go(scores, targets, ks=(1, 5, 10)):
    order = np.argsort(-scores, axis=1)
    out = {}
    for k in ks:
        topk = order[:, :k]
        tp = np.asarray([targets[i, topk[i]].sum() for i in range(len(targets))])
        out[f"P2G_R@{k}"] = float((tp / np.maximum(targets.sum(1), 1)).mean())
        out[f"P2G_P@{k}"] = float((tp / k).mean())
    return out


def go_to_protein(scores, targets, ks=(1, 5, 10)):
    order = np.argsort(-scores, axis=0)
    out = {}
    valid_terms = np.where(targets.sum(0) > 0)[0]
    for k in ks:
        recalls, precs = [], []
        for j in valid_terms:
            topk = order[:k, j]
            tp = targets[topk, j].sum()
            recalls.append(tp / max(targets[:, j].sum(), 1))
            precs.append(tp / k)
        out[f"G2P_R@{k}"] = float(np.mean(recalls))
        out[f"G2P_P@{k}"] = float(np.mean(precs))
    return out


def evaluate_scores(scores, targets, train_targets):
    fmax, aupr = fmax_aupr(scores, targets)
    ndcg10, map_score = ndcg_map(scores, targets, k=10)
    out = {
        "Fmax": fmax,
        "AUPR": aupr,
        "wFmax": wfmax(scores, targets, train_targets),
        "nDCG@10": ndcg10,
        "MAP": map_score,
    }
    out.update(protein_to_go(scores, targets))
    out.update(go_to_protein(scores, targets))
    return out


@torch.no_grad()
def predict_scores(model, x, term_embeddings, device, batch_size):
    model.eval()
    rows = []
    term_embeddings = term_embeddings.to(device)
    for i in range(0, len(x), batch_size):
        rows.append(model(x[i:i + batch_size].to(device), term_embeddings).cpu())
    return torch.cat(rows).numpy().astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esm-model", default=DEFAULT_ESM_MODEL)
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--splits-cache", default="protst_go_mf_decoded.pt")
    parser.add_argument("--vocab-file", default="go_mf_vocab.json")
    parser.add_argument("--feature-cache", default=None)
    parser.add_argument("--term-embedding-cache", default=None)
    parser.add_argument("--term-text-file", default=None)
    parser.add_argument("--label", default="plip_esm2_650m_neuml_bce")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embed-batch-size", type=int, default=16)
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--pos-weight-cap", type=float, default=100.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--compute-protein-feats", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Label:  {args.label}")

    splits, splits_path = load_splits(args.splits_cache)
    num_terms = int(splits["train"]["targets"].shape[1])
    dataset_tag = slugify(splits_path.stem.replace("_decoded", ""))
    text_cache = args.term_embedding_cache
    if text_cache is None:
        text_cache = f"go_terms_{dataset_tag}_{model_tag(args.text_model)}.pt"
    feature_cache = args.feature_cache
    if feature_cache is None:
        feature_cache = f"esm2_{model_tag(args.esm_model)}_{dataset_tag}_feats.pt"
    print(f"Splits: {splits_path} ({num_terms} GO terms)")
    print(f"Feature cache: {feature_cache}")
    print(f"Text cache:    {text_cache}")

    texts = load_term_texts(args.term_text_file, args.vocab_file, num_terms)
    term_embeddings = compute_term_embeddings(
        texts, args.text_model, device, args.text_batch_size, text_cache
    )

    feats_path = Path(feature_cache)
    if not feats_path.is_absolute():
        feats_path = CACHE_DIR / feature_cache
    if not feats_path.exists() and not args.compute_protein_feats:
        raise FileNotFoundError(
            f"{feats_path} not found. Re-run with --compute-protein-feats on a GPU node."
        )
    feats = compute_protein_embeddings(
        splits, args.esm_model, device, args.embed_batch_size, args.max_len,
        feature_cache
    )

    xtr = feats["train"].float()
    xva = feats["validation"].float()
    xte = feats["test"].float()
    ytr = splits["train"]["targets"].float()
    yva = splits["validation"]["targets"].float()
    yte = splits["test"]["targets"].float()

    keep = ytr.sum(1) > 0
    xtr, ytr = xtr[keep], ytr[keep]
    train_np = ytr.numpy().astype(np.float32)

    pos = ytr.sum(0).clamp(min=1)
    neg = ytr.shape[0] - ytr.sum(0)
    pos_weight = (neg / pos).clamp(max=args.pos_weight_cap).to(device)

    ds = TensorDataset(xtr, ytr)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model = ProteinGoPLIP(
        protein_dim=xtr.shape[1],
        text_dim=term_embeddings.shape[1],
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    term_device = term_embeddings.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_metric, best_state = -1.0, None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, term_device)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        total_loss /= len(loader.dataset)

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            val_scores = predict_scores(model, xva, term_embeddings, device, args.batch_size)
            val_metrics = evaluate_scores(
                val_scores, yva.numpy().astype(np.float32), train_np
            )
            if val_metrics["Fmax"] > best_metric:
                best_metric = val_metrics["Fmax"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ep {epoch:3d}/{args.epochs}  loss={total_loss:.4f}  "
                  f"val_Fmax={val_metrics['Fmax']:.4f}  "
                  f"val_AUPR={val_metrics['AUPR']:.4f}  "
                  f"scale={model.log_logit_scale.exp().item():.2f}  "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)

    model.load_state_dict(best_state)
    test_scores = predict_scores(model, xte, term_embeddings, device, args.batch_size)
    test_metrics = evaluate_scores(
        test_scores, yte.numpy().astype(np.float32), train_np
    )
    rounded = {k: round(float(v), 6) for k, v in test_metrics.items()}

    print("\nTest metrics")
    print(json.dumps(rounded, indent=2))

    slug = slugify(args.label)
    torch.save({
        "state_dict": model.state_dict(),
        "args": vars(args),
        "results": rounded,
    }, CKPT_DIR / f"{slug}.pt")
    with open(RESULTS_DIR / f"{slug}.json", "w") as f:
        json.dump(rounded, f, indent=2)
    print(f"\nSaved checkpoint/results with prefix: {slug}")


if __name__ == "__main__":
    main()
