#!/usr/bin/env python3
"""
Data preparation — Step 2: GPU required, run on compute node.

Computes and caches the frozen encoder embeddings used by all probe experiments.
Run this ONCE after prefetch_data.py; all training scripts reuse the cache.

Usage (HPRC compute node with A100):
    conda activate hypalign
    python experiments/hyp_ssf_probe/scripts/prepare_features.py

What this produces in experiments/hyp_ssf_probe/cache/:
    esm2_esm2_t33_650M_UR50D_go_feats.pt  — mean-pool ESM2-650M features
                                             train [27496, 1280]
                                             validation [3053, 1280]
                                             test [2991, 1280]
                                             ~164 MB, ~15 min on A100

    go_term_embs.pt                         — PubMedBERT CLS embeddings
                                             [489, 768], ~1.5 MB, <1 min
                                             (already in git; regenerated if absent)
"""
import json
import time
from pathlib import Path

import torch

REPO_ROOT     = Path(__file__).resolve().parents[3]
CACHE_DIR     = REPO_ROOT / "experiments" / "hyp_ssf_probe" / "cache"
ESM2_HF       = "facebook/esm2_t33_650M_UR50D"
PUBMEDBERT_HF = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
GO_TERMS      = 489

def step(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}", flush=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cpu":
    print("WARNING: running on CPU will be very slow (~hours). Use a GPU node.")


# ── ESM2-650M protein features ────────────────────────────────────────────────

step("1/2  ESM2-650M protein features")
feat_path = CACHE_DIR / f"esm2_{ESM2_HF.split('/')[-1]}_go_feats.pt"

if feat_path.exists():
    d = torch.load(feat_path, map_location="cpu", weights_only=True)
    print(f"  Already cached: {feat_path.name}")
    for split, t in d.items():
        print(f"    {split}: {tuple(t.shape)}")
else:
    decoded_path = CACHE_DIR / "protst_go_mf_decoded.pt"
    if not decoded_path.exists():
        print("ERROR: protst_go_mf_decoded.pt not found.")
        print("       Run prefetch_data.py first on the login node.")
        raise SystemExit(1)

    splits = torch.load(decoded_path, map_location="cpu", weights_only=False)

    from transformers import EsmTokenizer, EsmModel
    tok   = EsmTokenizer.from_pretrained(ESM2_HF)
    model = EsmModel.from_pretrained(ESM2_HF).to(device).eval()

    feats = {}
    t0 = time.time()
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
                if i % 1600 == 0:
                    elapsed = time.time() - t0
                    print(f"  {split}: {i}/{len(seqs)}  ({elapsed:.0f}s)", flush=True)
        feats[split] = torch.cat(embs)
        print(f"  {split}: {tuple(feats[split].shape)}")

    del model; torch.cuda.empty_cache()
    torch.save(feats, feat_path)
    print(f"\n  Saved {feat_path.name} "
          f"({feat_path.stat().st_size // (1024*1024)} MB)  "
          f"total {time.time()-t0:.0f}s")


# ── PubMedBERT GO-term embeddings ────────────────────────────────────────────

step("2/2  PubMedBERT GO-term embeddings")
go_emb_path = CACHE_DIR / "go_term_embs.pt"

if go_emb_path.exists():
    t = torch.load(go_emb_path, map_location="cpu", weights_only=True)
    print(f"  Already cached: {go_emb_path.name}  {tuple(t.shape)}")
else:
    vocab_path = CACHE_DIR / "go_mf_vocab.json"
    if not vocab_path.exists():
        print("ERROR: go_mf_vocab.json not found.")
        print("       It should be in the repo under experiments/hyp_ssf_probe/data/")
        raise SystemExit(1)

    vocab = json.load(open(vocab_path))
    names = [f"FUNCTION: {vocab['idx_to_name'][str(i)]}." for i in range(GO_TERMS)]

    from transformers import AutoTokenizer, AutoModel
    tok   = AutoTokenizer.from_pretrained(PUBMEDBERT_HF)
    model = AutoModel.from_pretrained(PUBMEDBERT_HF).to(device).eval()
    embs  = []
    with torch.no_grad():
        for i in range(0, len(names), 64):
            inp = tok(names[i:i+64], return_tensors="pt", padding=True,
                      truncation=True, max_length=128).to(device)
            out = model(**inp)
            embs.append(out.last_hidden_state[:, 0].cpu())
    go_embs = torch.cat(embs)
    del model; torch.cuda.empty_cache()
    torch.save(go_embs, go_emb_path)
    print(f"  Saved {go_emb_path.name}  {tuple(go_embs.shape)}")

print("\nFeature preparation complete. You can now submit training jobs.")
