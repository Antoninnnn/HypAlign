#!/usr/bin/env python3
"""
Frozen ESM2-650M + NeuML PubMedBERT InfoNCE baselines for ProtST GO-MF.

This script mirrors run_plip_go.py for data, frozen embeddings, model shape, and
metrics, but swaps the supervised BCE objective for contrastive objectives:

  pair_infonce
      Sample one positive GO term per protein and apply symmetric CLIP/InfoNCE
      over the B x B protein/text batch.

  multipos_infonce
      Same sampled text batch, but any protein/text pair is positive if the
      protein has that GO term. This keeps the InfoNCE denominator while avoiding
      many false negatives caused by shared GO labels in a batch.

  full_multipos_infonce
      Score every protein in the batch against all 489 GO terms and maximize the
      probability mass assigned to all true labels. This is still a softmax/NCE
      objective, but uses the full multi-hot supervision.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
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
N_GO = 489
EPS = 1e-8


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class ProteinGoContrastive(nn.Module):
    def __init__(self, protein_dim: int, text_dim: int,
                 proj_dim: int = 256, hidden_dim: int = 512,
                 init_logit_scale: float = 10.0):
        super().__init__()
        self.protein_projector = ProjectionHead(protein_dim, hidden_dim, proj_dim)
        self.term_projector = ProjectionHead(text_dim, hidden_dim, proj_dim)
        self.log_logit_scale = nn.Parameter(
            torch.tensor(np.log(init_logit_scale), dtype=torch.float32)
        )

    def forward(self, protein_embeddings: torch.Tensor,
                term_embeddings: torch.Tensor) -> torch.Tensor:
        protein_proj = self.protein_projector(protein_embeddings)
        term_proj = self.term_projector(term_embeddings)
        return protein_proj @ term_proj.T * self.log_logit_scale.exp()

    def clamp_scale(self, max_scale: float) -> None:
        if max_scale > 0:
            self.log_logit_scale.data.clamp_(max=math.log(max_scale))


def model_tag(model_name: str) -> str:
    return model_name.split("/")[-1].replace("/", "_")


def slugify(label: str) -> str:
    out = []
    for ch in label:
        out.append(ch if ch.isalnum() or ch in "-_." else "_")
    return "".join(out).strip("_")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_splits():
    path = CACHE_DIR / "protst_go_mf_decoded.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/prefetch_data.py first."
        )
    return torch.load(path, map_location="cpu", weights_only=False)


def load_term_texts(term_text_file: str | None):
    if term_text_file:
        texts = [
            line.strip()
            for line in Path(term_text_file).read_text().splitlines()
            if line.strip()
        ]
    else:
        vocab_path = CACHE_DIR / "go_mf_vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"{vocab_path} not found. Pass --term-text-file with 489 ordered prompts."
            )
        vocab = json.loads(vocab_path.read_text())
        texts = [f"FUNCTION: {name}." for name in vocab["go_names"]]
    if len(texts) != N_GO:
        raise ValueError(f"Expected {N_GO} GO term texts, found {len(texts)}")
    return texts


@torch.no_grad()
def mean_pool_text(model, enc):
    out = model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    return (out * mask).sum(1) / mask.sum(1).clamp(min=1)


def compute_term_embeddings(texts, text_model: str, device, batch_size: int):
    cache_path = CACHE_DIR / f"go_terms_{model_tag(text_model)}.pt"
    if cache_path.exists():
        print(f"  [text] using cache: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

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
                               max_len: int):
    tag = model_tag(esm_model)
    cache_path = CACHE_DIR / f"esm2_{tag}_go_feats.pt"
    if cache_path.exists():
        print(f"  [esm] using cache: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

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
        best_f = max(best_f, 2 * prec * rec / (prec + rec + EPS))
    order = np.argsort(recs)
    aupr = float(np.trapz(np.asarray(precs)[order], np.asarray(recs)[order]))
    return best_f, aupr


def macro_aupr(scores, targets):
    try:
        from sklearn.metrics import average_precision_score
    except Exception:
        return None
    valid = np.where(targets.sum(0) > 0)[0]
    vals = []
    for j in valid:
        vals.append(average_precision_score(targets[:, j], scores[:, j]))
    return float(np.mean(vals)) if vals else None


def wfmax(scores, targets, train_targets):
    pos_freq = (train_targets > 0.5).mean(0)
    ic = -np.log(np.clip(pos_freq, 1e-6, 1.0))
    best = 0.0
    for t in np.percentile(scores, np.linspace(1, 99, 199)):
        pred = (scores > t).astype(np.float32)
        tp_w = (pred * targets * ic).sum(1)
        pred_w = (pred * ic).sum(1)
        true_w = (targets * ic).sum(1)
        prec = float((tp_w / np.maximum(pred_w, EPS)).mean())
        rec = float((tp_w / np.maximum(true_w, EPS)).mean())
        best = max(best, 2 * prec * rec / (prec + rec + EPS))
    return best


def ndcg_map(scores, targets, k=10):
    order = np.argsort(-scores, axis=1)
    ndcg_scores, ap_scores = [], []
    for i in range(len(scores)):
        true_idx = set(np.where(targets[i] > 0.5)[0])
        if not true_idx:
            continue
        row = order[i]
        dcg = sum(1.0 / math.log2(r + 2) for r, j in enumerate(row[:k])
                  if j in true_idx)
        idcg = sum(1.0 / math.log2(r + 2)
                   for r in range(min(len(true_idx), k)))
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
    m_aupr = macro_aupr(scores, targets)
    if m_aupr is not None:
        out["macro_AUPR"] = m_aupr
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


def sample_positive_terms(yb: torch.Tensor) -> torch.Tensor:
    idxs = []
    for row in yb:
        pos = (row > 0.5).nonzero(as_tuple=True)[0]
        draw = torch.randint(len(pos), (1,)).item()
        idxs.append(pos[draw])
    return torch.stack(idxs).long()


def cross_entropy_pair_loss(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(logits.size(0), device=logits.device)
    return (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.T, labels)
    ) / 2.0


def multi_positive_nce(logits: torch.Tensor, pos_mask: torch.Tensor,
                       dim: int = 1) -> torch.Tensor:
    if dim == 0:
        logits = logits.T
        pos_mask = pos_mask.T
    valid = pos_mask.any(dim=1)
    logits = logits[valid]
    pos_mask = pos_mask[valid]
    if logits.numel() == 0:
        return logits.sum() * 0.0
    neg_inf = torch.finfo(logits.dtype).min
    log_num = logits.masked_fill(~pos_mask, neg_inf).logsumexp(dim=1)
    log_den = logits.logsumexp(dim=1)
    return -(log_num - log_den).mean()


def contrastive_loss(model: ProteinGoContrastive, xb: torch.Tensor,
                     yb: torch.Tensor, term_embeddings: torch.Tensor,
                     loss_name: str) -> torch.Tensor:
    if loss_name == "full_multipos_infonce":
        logits = model(xb, term_embeddings)
        return multi_positive_nce(logits, yb.to(logits.device).bool(), dim=1)

    go_idx = sample_positive_terms(yb).to(xb.device)
    text_b = term_embeddings[go_idx]
    logits = model(xb, text_b)
    if loss_name == "pair_infonce":
        return cross_entropy_pair_loss(logits)
    if loss_name == "multipos_infonce":
        mask = yb[:, go_idx.cpu()].to(logits.device).bool()
        row_loss = multi_positive_nce(logits, mask, dim=1)
        col_loss = multi_positive_nce(logits, mask, dim=0)
        return (row_loss + col_loss) / 2.0
    raise ValueError(f"Unknown loss: {loss_name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esm-model", default=DEFAULT_ESM_MODEL)
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--term-text-file", default=None)
    parser.add_argument("--label", default="infonce_esm2_650m_neuml")
    parser.add_argument("--loss", choices=[
        "pair_infonce",
        "multipos_infonce",
        "full_multipos_infonce",
    ], default="pair_infonce")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embed-batch-size", type=int, default=16)
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--compute-protein-feats", action="store_true")
    parser.add_argument("--init-logit-scale", type=float, default=10.0)
    parser.add_argument("--max-logit-scale", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Label:  {args.label}")
    print(f"Loss:   {args.loss}")
    print(f"Seed:   {args.seed}")

    splits = load_splits()
    texts = load_term_texts(args.term_text_file)
    term_embeddings = compute_term_embeddings(
        texts, args.text_model, device, args.text_batch_size
    ).float()

    feats_path = CACHE_DIR / f"esm2_{model_tag(args.esm_model)}_go_feats.pt"
    if not feats_path.exists() and not args.compute_protein_feats:
        raise FileNotFoundError(
            f"{feats_path} not found. Re-run with --compute-protein-feats on a GPU node."
        )
    feats = compute_protein_embeddings(
        splits, args.esm_model, device, args.embed_batch_size, args.max_len
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

    ds = TensorDataset(xtr, ytr)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model = ProteinGoContrastive(
        protein_dim=xtr.shape[1],
        text_dim=term_embeddings.shape[1],
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        init_logit_scale=args.init_logit_scale,
    ).to(device)
    term_device = term_embeddings.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    best_metric, best_state = -1.0, None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = contrastive_loss(model, xb, yb, term_device, args.loss)
            loss.backward()
            optimizer.step()
            model.clamp_scale(args.max_logit_scale)
            total_loss += loss.item() * len(xb)
        total_loss /= len(loader.dataset)

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            val_scores = predict_scores(model, xva, term_embeddings, device,
                                        args.batch_size)
            val_metrics = evaluate_scores(
                val_scores, yva.numpy().astype(np.float32), train_np
            )
            if val_metrics["Fmax"] > best_metric:
                best_metric = val_metrics["Fmax"]
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
            macro = val_metrics.get("macro_AUPR")
            macro_s = f"  val_macro_AUPR={macro:.4f}" if macro is not None else ""
            print(f"  ep {epoch:3d}/{args.epochs}  loss={total_loss:.4f}  "
                  f"val_Fmax={val_metrics['Fmax']:.4f}  "
                  f"val_AUPR={val_metrics['AUPR']:.4f}{macro_s}  "
                  f"scale={model.log_logit_scale.exp().item():.2f}  "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)

    model.load_state_dict(best_state)
    test_scores = predict_scores(model, xte, term_embeddings, device,
                                 args.batch_size)
    test_metrics = evaluate_scores(
        test_scores, yte.numpy().astype(np.float32), train_np
    )
    rounded = {k: round(float(v), 6) for k, v in test_metrics.items()}
    rounded["best_val_Fmax"] = round(float(best_metric), 6)
    rounded["loss"] = args.loss
    rounded["batch_size"] = args.batch_size
    rounded["seed"] = args.seed

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
