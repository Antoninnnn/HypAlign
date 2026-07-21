#!/usr/bin/env python3
"""Evaluate a saved fine-tuned ESM2 GO-MF checkpoint with both AUPR definitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import EsmModel, EsmTokenizer

import run_finetune_go as ft

REPO_ROOT = Path(__file__).resolve().parent
CACHE_DIR = REPO_ROOT / "cache"
RESULTS_DIR = REPO_ROOT / "results"


def wfmax(logits_np: np.ndarray, tgt_np: np.ndarray, train_tgt_np: np.ndarray) -> float:
    pos_freq = (train_tgt_np > 0.5).mean(0)
    ic = -np.log(np.clip(pos_freq, 1e-6, 1.0))
    best = 0.0
    for t in np.percentile(logits_np, np.linspace(1, 99, 199)):
        preds = (logits_np > t).astype(np.float32)
        tp_w = (preds * tgt_np * ic).sum(1)
        pred_w = (preds * ic).sum(1)
        true_w = (tgt_np * ic).sum(1)
        p = float((tp_w / np.maximum(pred_w, 1e-8)).mean())
        r = float((tp_w / np.maximum(true_w, 1e-8)).mean())
        best = max(best, 2 * p * r / (p + r + 1e-8))
    return best


def ndcg_map(logits_np: np.ndarray, tgt_np: np.ndarray, k: int = 10) -> tuple[float, float]:
    order = np.argsort(-logits_np, axis=1)
    ndcg_scores, ap_scores = [], []
    for i in range(len(logits_np)):
        true_idx = set(np.where(tgt_np[i] > 0.5)[0])
        if not true_idx:
            continue
        row = order[i]
        dcg = sum(1.0 / np.log2(r + 2) for r, j in enumerate(row[:k]) if j in true_idx)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(true_idx), k)))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
        hits, ap = 0, 0.0
        for r, j in enumerate(row):
            if j in true_idx:
                hits += 1
                ap += hits / (r + 1)
        ap_scores.append(ap / len(true_idx))
    return float(np.mean(ndcg_scores)), float(np.mean(ap_scores))


def metrics_for_split(logits_np: np.ndarray, tgt_np: np.ndarray,
                      train_tgt_np: np.ndarray) -> dict[str, float]:
    fmax, protein_aupr = ft.fmax_aupr(logits_np, tgt_np)
    macro_aupr = ft.macro_aupr(logits_np, tgt_np)
    ndcg10, map_score = ndcg_map(logits_np, tgt_np, k=10)
    out = {
        "Fmax": fmax,
        "protein_AUPR": protein_aupr,
        "macro_AUPR": macro_aupr,
        "wFmax": wfmax(logits_np, tgt_np, train_tgt_np),
        "nDCG@10": ndcg10,
        "MAP": map_score,
    }
    out.update({f"P2G_{k}": v for k, v in ft.retrieval_metrics(logits_np, tgt_np).items()})
    return {k: round(float(v), 6) for k, v in out.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "best_ESM2-650M-FT-BCE.pt",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--split", choices=["validation", "test", "both"], default="both")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})

    ft.ESM_MODEL = ckpt.get("esm_model", ckpt_args.get("esm_model", ft.ESM_MODEL))
    ft.MAX_LEN = args.max_len or ckpt_args.get("max_len", ft.MAX_LEN)
    ft.USE_AMP = args.amp
    ft.NUM_WORKERS = args.num_workers

    print(f"Checkpoint: {args.checkpoint}")
    print(f"ESM model:  {ft.ESM_MODEL}")
    print(f"Device:     {device}")
    print(f"Max len:    {ft.MAX_LEN}")

    splits = torch.load(CACHE_DIR / "protst_go_mf_decoded.pt",
                        map_location="cpu", weights_only=False)
    train_tgt_np = splits["train"]["targets"].float().numpy().astype(np.float32)

    tokenizer = EsmTokenizer.from_pretrained(ft.ESM_MODEL)
    backbone = EsmModel.from_pretrained(ft.ESM_MODEL)
    ft.freeze_unused_esm_parameters(backbone)
    model = ft.GOClassifier(backbone, backbone.config.hidden_size)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    selected = ["validation", "test"] if args.split == "both" else [args.split]
    results = {
        "checkpoint": str(args.checkpoint),
        "esm_model": ft.ESM_MODEL,
        "label": ckpt.get("label"),
        "loss": ckpt.get("loss"),
    }
    for split in selected:
        ds = ft.GODataset(splits[split]["seqs"], splits[split]["targets"],
                          filter_empty=False)
        loader = ft.make_loader(
            ds, args.batch_size, shuffle=False,
            pin_memory=device.type == "cuda",
        )
        logits_np, tgt_np = ft.predict(model, loader, tokenizer, device)
        tgt_np = tgt_np.astype(np.float32)
        results[split] = metrics_for_split(logits_np, tgt_np, train_tgt_np)
        print(f"\n{split}")
        print(json.dumps(results[split], indent=2))

    out = args.out or RESULTS_DIR / f"eval_{args.checkpoint.stem}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
