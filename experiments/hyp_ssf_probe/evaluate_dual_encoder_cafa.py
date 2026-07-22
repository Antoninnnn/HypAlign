#!/usr/bin/env python3
"""Evaluate clean Protein-GO dual-encoder checkpoints with CAFA-style metrics.

The training script's JSON reports a protein-centric threshold-sweep AUPR that
is useful for internal tracking but is not the same as the micro-AUPR usually
reported in ProtST-style tables. This evaluator keeps the old metrics for
traceability and adds:

  - CAFA-style Fmax: precision averaged over proteins with predictions, recall
    averaged over benchmark proteins.
  - weighted Fmax: same protocol, weighted by GO information content.
  - Smin: sqrt(remaining_uncertainty^2 + misinformation^2).
  - micro-AUPR: average precision over all protein-GO pairs.
  - macro-AUPR: average precision per GO term, averaged over non-empty terms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

import run_plip_go as plip


REPO_ROOT = Path(__file__).resolve().parent
CACHE_DIR = REPO_ROOT / "cache"
CKPT_DIR = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"


DEFAULT_RUNS = {
    "mf": {
        "checkpoint": "plip_esm2_650m_neuml_go_mf_clean_bce_posw_cap100.pt",
        "splits": "protst_go_mf_decoded.pt",
        "features": "esm2_esm2_t33_650M_UR50D_protst_go_mf_feats.pt",
        "text": "go_terms_protst_go_mf_NeuML_pubmedbert-base-embeddings.pt",
    },
    "bp": {
        "checkpoint": "plip_esm2_650m_neuml_go_bp_bce_posw_cap100.pt",
        "splits": "protst_go_bp_decoded.pt",
        "features": "esm2_esm2_t33_650M_UR50D_protst_go_bp_feats.pt",
        "text": "go_terms_protst_go_bp_NeuML_pubmedbert-base-embeddings.pt",
    },
    "cc": {
        "checkpoint": "plip_esm2_650m_neuml_go_cc_bce_posw_cap100.pt",
        "splits": "protst_go_cc_decoded.pt",
        "features": "esm2_esm2_t33_650M_UR50D_protst_go_cc_feats.pt",
        "text": "go_terms_protst_go_cc_NeuML_pubmedbert-base-embeddings.pt",
    },
}


def resolve(base: Path, name: str | Path) -> Path:
    path = Path(name)
    return path if path.is_absolute() else base / path


def thresholds_from_scores(scores: np.ndarray, n: int = 1001) -> np.ndarray:
    values = np.unique(np.linspace(0.0, 1.0, n, dtype=np.float32))
    return values


def information_content(train_targets: np.ndarray) -> np.ndarray:
    freq = (train_targets > 0.5).mean(0)
    return -np.log(np.clip(freq, 1e-6, 1.0)).astype(np.float32)


def legacy_fmax_aupr(scores: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
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


def cafa_fmax(scores: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    best = {"fmax": 0.0, "threshold": 0.0, "precision": 0.0, "recall": 0.0,
            "coverage": 0.0}
    true_count = targets.sum(1)
    valid_true = true_count > 0
    for t in thresholds_from_scores(scores):
        pred = (scores >= t).astype(np.float32)
        pred_count = pred.sum(1)
        has_pred = pred_count > 0
        tp = (pred * targets).sum(1)
        precision = float((tp[has_pred] / pred_count[has_pred]).mean()) \
            if has_pred.any() else 0.0
        recall = float((tp[valid_true] / true_count[valid_true]).mean()) \
            if valid_true.any() else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if f1 > best["fmax"]:
            best = {
                "fmax": float(f1),
                "threshold": float(t),
                "precision": precision,
                "recall": recall,
                "coverage": float(has_pred.mean()),
            }
    return best


def weighted_fmax(scores: np.ndarray, targets: np.ndarray, ic: np.ndarray) -> dict[str, float]:
    best = {"wfmax": 0.0, "threshold": 0.0, "precision": 0.0, "recall": 0.0,
            "coverage": 0.0}
    true_weight = (targets * ic).sum(1)
    valid_true = true_weight > 0
    for t in thresholds_from_scores(scores):
        pred = (scores >= t).astype(np.float32)
        pred_weight = (pred * ic).sum(1)
        has_pred = pred_weight > 0
        tp_weight = (pred * targets * ic).sum(1)
        precision = float((tp_weight[has_pred] / pred_weight[has_pred]).mean()) \
            if has_pred.any() else 0.0
        recall = float((tp_weight[valid_true] / true_weight[valid_true]).mean()) \
            if valid_true.any() else 0.0
        wf1 = 2 * precision * recall / (precision + recall + 1e-8)
        if wf1 > best["wfmax"]:
            best = {
                "wfmax": float(wf1),
                "threshold": float(t),
                "precision": precision,
                "recall": recall,
                "coverage": float(has_pred.mean()),
            }
    return best


def smin(scores: np.ndarray, targets: np.ndarray, ic: np.ndarray) -> dict[str, float]:
    best = {"smin": float("inf"), "threshold": 0.0, "ru": 0.0, "mi": 0.0}
    for t in thresholds_from_scores(scores):
        pred = (scores >= t).astype(np.float32)
        ru = float(((1.0 - pred) * targets * ic).sum(1).mean())
        mi = float((pred * (1.0 - targets) * ic).sum(1).mean())
        s = float(np.sqrt(ru * ru + mi * mi))
        if s < best["smin"]:
            best = {"smin": s, "threshold": float(t), "ru": ru, "mi": mi}
    return best


def macro_aupr(scores: np.ndarray, targets: np.ndarray) -> float:
    vals = []
    for j in np.where(targets.sum(0) > 0)[0]:
        if np.unique(targets[:, j]).size < 2:
            continue
        vals.append(average_precision_score(targets[:, j], scores[:, j]))
    return float(np.mean(vals)) if vals else 0.0


def rank_metrics(scores: np.ndarray, targets: np.ndarray, k: int = 10) -> tuple[float, float]:
    order = np.argsort(-scores, axis=1)
    ndcg_scores, ap_scores = [], []
    for i in range(len(scores)):
        true_idx = set(np.where(targets[i] > 0.5)[0])
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


def p2g_precision_at_ks(scores: np.ndarray, targets: np.ndarray, ks=(1, 5, 10)) -> dict[str, float]:
    order = np.argsort(-scores, axis=1)
    out = {}
    for k in ks:
        topk = order[:, :k]
        tp = np.asarray([targets[i, topk[i]].sum() for i in range(len(targets))])
        out[f"P2G_P@{k}"] = float((tp / k).mean())
        out[f"P2G_R@{k}"] = float((tp / np.maximum(targets.sum(1), 1)).mean())
    return out


def load_scores(run_key: str, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = DEFAULT_RUNS[run_key]
    splits = torch.load(resolve(CACHE_DIR, cfg["splits"]), map_location="cpu",
                        weights_only=False)
    features = torch.load(resolve(CACHE_DIR, cfg["features"]), map_location="cpu",
                          weights_only=True)
    text = torch.load(resolve(CACHE_DIR, cfg["text"]), map_location="cpu",
                      weights_only=True)
    ckpt = torch.load(resolve(CKPT_DIR, cfg["checkpoint"]), map_location="cpu",
                      weights_only=False)

    test_x = features["test"].float()
    test_y = splits["test"]["targets"].float().numpy().astype(np.float32)
    train_y = splits["train"]["targets"].float().numpy().astype(np.float32)
    if test_x.shape[0] != test_y.shape[0]:
        raise ValueError(f"{run_key}: feature rows do not match test targets")
    if text.shape[0] != test_y.shape[1]:
        raise ValueError(f"{run_key}: text rows do not match target columns")

    args = ckpt["args"]
    model = plip.ProteinGoPLIP(
        protein_dim=test_x.shape[1],
        text_dim=text.shape[1],
        proj_dim=int(args["proj_dim"]),
        hidden_dim=int(args["hidden_dim"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    logits = plip.predict_scores(model, test_x, text.float(), device, batch_size)
    scores = 1.0 / (1.0 + np.exp(-logits))
    return scores.astype(np.float32), test_y, train_y


def evaluate_run(run_key: str, device: torch.device, batch_size: int) -> dict:
    scores, targets, train_targets = load_scores(run_key, device, batch_size)
    ic = information_content(train_targets)
    legacy_f, legacy_aupr = legacy_fmax_aupr(scores, targets)
    cafa = cafa_fmax(scores, targets)
    wf = weighted_fmax(scores, targets, ic)
    s = smin(scores, targets, ic)
    ndcg10, map_score = rank_metrics(scores, targets, k=10)
    out = {
        "num_proteins": int(targets.shape[0]),
        "num_terms": int(targets.shape[1]),
        "avg_labels_per_protein": round(float(targets.sum(1).mean()), 6),
        "legacy_protein_Fmax": round(float(legacy_f), 6),
        "legacy_protein_AUPR": round(float(legacy_aupr), 6),
        "cafa_Fmax": round(float(cafa["fmax"]), 6),
        "cafa_threshold": round(float(cafa["threshold"]), 6),
        "cafa_precision": round(float(cafa["precision"]), 6),
        "cafa_recall": round(float(cafa["recall"]), 6),
        "cafa_coverage": round(float(cafa["coverage"]), 6),
        "weighted_Fmax": round(float(wf["wfmax"]), 6),
        "weighted_threshold": round(float(wf["threshold"]), 6),
        "weighted_precision": round(float(wf["precision"]), 6),
        "weighted_recall": round(float(wf["recall"]), 6),
        "Smin": round(float(s["smin"]), 6),
        "Smin_threshold": round(float(s["threshold"]), 6),
        "remaining_uncertainty": round(float(s["ru"]), 6),
        "misinformation": round(float(s["mi"]), 6),
        "micro_AUPR": round(float(average_precision_score(targets.ravel(), scores.ravel())), 6),
        "macro_AUPR": round(float(macro_aupr(scores, targets)), 6),
        "nDCG@10": round(float(ndcg10), 6),
        "MAP": round(float(map_score), 6),
    }
    out.update({k: round(float(v), 6) for k, v in p2g_precision_at_ks(scores, targets).items()})
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["mf", "bp", "cc"],
                        choices=sorted(DEFAULT_RUNS))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out", type=Path,
                        default=RESULTS_DIR / "dual_encoder_cafa_metrics.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    results = {}
    for run_key in args.runs:
        print(f"Evaluating {run_key.upper()} on {device} ...", flush=True)
        results[run_key] = evaluate_run(run_key, device, args.batch_size)
        print(json.dumps(results[run_key], indent=2), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
