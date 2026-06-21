#!/usr/bin/env python3
"""
Hyperbolic SSF Probe — ProtST-GO-MF Dataset
=============================================
Dataset:    mila-intel/ProtST-GeneOntology-MF (33k proteins, 489 GO-MF classes)
Text:       489 GO term descriptions "FUNCTION: {name}." pre-embedded by PubMedBERT
Seq:        ESM-2 8M features (mean-pooled), pre-computed and cached
Training:   Sample 1 positive GO term per protein per batch → standard InfoNCE
Eval:       Fmax (protein-centric, threshold-sweep F1 — standard GO metric)

Conditions:
  1. Euclidean probe       — linear → L2-normalize, cosine sim
  2. Hyp probe             — linear → exp_map0, hyperbolic InfoNCE
  3. Hyp + MERU λ=0.1      — + entailment cone loss
  4. Hyp + MERU λ=0.5      — + entailment cone loss

Run:
    conda activate pannot-infer
    cd /home/yining_yang/NLP/hyperbolic/experiments/hyp_ssf_probe
    python run_experiment_go.py
    python run_experiment_go.py --epochs 100 --proj-dim 128
"""

from __future__ import annotations

import argparse
import ast
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

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent
CACHE_DIR   = REPO_ROOT / "cache"
RESULTS_DIR = REPO_ROOT / "results"
CKPT_DIR    = REPO_ROOT / "checkpoints"
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

PUBMEDBERT_HF = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
ESM2_HF       = "facebook/esm2_t33_650M_UR50D"
HF_DATASET    = "mila-intel/ProtST-GeneOntology-MF"

# ESM standard vocab for detokenization
ESM_VOCAB = {
    4:'A', 5:'R', 6:'N', 7:'D', 8:'C', 9:'Q', 10:'E', 11:'G',
    12:'H', 13:'I', 14:'L', 15:'K', 16:'M', 17:'F', 18:'P',
    19:'S', 20:'T', 21:'W', 22:'Y', 23:'V'
}

GO_TERMS    = 489
ESM_DIM     = 1280
BERT_DIM    = 768
PROJ_DIM    = 256
BATCH_SIZE  = 128
LR          = 3e-4
EPOCHS      = 60
TEMPERATURE = 0.07
CURV_INIT   = 1.0
EPS         = 1e-8


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA  (download, decode, cache)
# ══════════════════════════════════════════════════════════════════════════════

def _download_csv_split(split_name: str) -> tuple[list[str], torch.Tensor]:
    """Download one CSV split from HuggingFace, parse prot_seq + targets."""
    import csv, io, requests

    fname = {"train": "gene_ontology_mf_train.csv",
             "validation": "gene_ontology_mf_valid.csv",
             "test": "gene_ontology_mf_test.csv"}[split_name]
    url = (f"https://huggingface.co/datasets/mila-intel/ProtST-GeneOntology-MF"
           f"/resolve/main/{fname}")

    local = CACHE_DIR / fname
    if not local.exists():
        print(f"    Downloading {fname} ...")
        r = requests.get(url, timeout=120, stream=True)
        r.raise_for_status()
        with open(local, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        print(f"    Saved {local.stat().st_size/1e6:.1f} MB")

    seqs, targets = [], []
    with open(local, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)          # '', prot_seq, targets, pdb_files
        col_seq = header.index("prot_seq")
        col_tgt = header.index("targets")
        for row in reader:
            tok = ast.literal_eval(row[col_seq])
            seq_str = "".join(ESM_VOCAB.get(t, "X") for t in tok)
            seqs.append(seq_str)
            tgt = ast.literal_eval(row[col_tgt])
            targets.append(torch.tensor(tgt, dtype=torch.float32))

    return seqs, torch.stack(targets)


def download_and_decode():
    """Download ProtST-GO-MF CSVs, decode sequences + targets, cache result.
    Returns dict 'train'/'validation'/'test' → {'seqs': list[str], 'targets': Tensor[N, 489]}.
    """
    cache_path = CACHE_DIR / "protst_go_mf_decoded.pt"
    if cache_path.exists():
        print(f"  [data] Using cached decoded data: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    splits = {}
    for split in ("train", "validation", "test"):
        seqs, targets_t = _download_csv_split(split)
        splits[split] = {"seqs": seqs, "targets": targets_t}
        n_pos = (targets_t > 0.5).sum(1)
        print(f"    {split}: {len(seqs)} proteins, "
              f"avg {n_pos.float().mean():.1f} GO terms/protein "
              f"(min {n_pos.min()}, max {n_pos.max()})")

    torch.save(splits, cache_path)
    print(f"  [data] Saved to {cache_path}")
    return splits


# ══════════════════════════════════════════════════════════════════════════════
# 2. ESM-2 FEATURES  (compute + cache per split)
# ══════════════════════════════════════════════════════════════════════════════

def compute_esm2_features(splits: dict, device: torch.device) -> dict:
    """Returns dict split → Tensor[N, ESM_DIM]."""
    _model_tag  = ESM2_HF.split("/")[-1]   # e.g. esm2_t33_650M_UR50D
    cache_path  = CACHE_DIR / f"esm2_{_model_tag}_go_feats.pt"
    if cache_path.exists():
        print(f"  [esm2] Using cache: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    print(f"  [esm2] Computing {ESM2_HF} features (mean-pool last hidden state) ...")
    from transformers import EsmTokenizer, EsmModel
    tok   = EsmTokenizer.from_pretrained(ESM2_HF)
    model = EsmModel.from_pretrained(ESM2_HF).to(device).eval()

    feats = {}
    for split, data in splits.items():
        seqs = data["seqs"]
        all_feats = []
        with torch.no_grad():
            for i in range(0, len(seqs), 32):
                batch_seqs = seqs[i:i+32]
                enc = tok(batch_seqs, return_tensors="pt", truncation=True,
                          max_length=512, padding=True).to(device)
                out = model(**enc).last_hidden_state  # [B, L, 320]
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out * mask).sum(1) / mask.sum(1)  # mean over residues
                all_feats.append(pooled.cpu())
                if i % 320 == 0:
                    print(f"    {split}: {i}/{len(seqs)}")
        feats[split] = torch.cat(all_feats)
        print(f"    {split}: {feats[split].shape}")

    torch.save(feats, cache_path)
    del model; torch.cuda.empty_cache()
    print(f"  [esm2] Saved to {cache_path}")
    return feats


# ══════════════════════════════════════════════════════════════════════════════
# 3. GO TERM EMBEDDINGS  (PubMedBERT, 489 terms, cached)
# ══════════════════════════════════════════════════════════════════════════════

def compute_go_embeddings(device: torch.device) -> torch.Tensor:
    """Returns Tensor[489, 768] — one CLS embedding per GO term."""
    cache_path = CACHE_DIR / "go_term_embs.pt"
    if cache_path.exists():
        print(f"  [go-emb] Using cache: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    vocab = json.load(open(CACHE_DIR / "go_mf_vocab.json"))
    go_texts = [f"FUNCTION: {name}." for name in vocab["go_names"]]

    print(f"  [go-emb] Encoding {len(go_texts)} GO terms with PubMedBERT ...")
    from transformers import AutoTokenizer, AutoModel
    tok   = AutoTokenizer.from_pretrained(PUBMEDBERT_HF)
    model = AutoModel.from_pretrained(PUBMEDBERT_HF).to(device).eval()

    out = []
    with torch.no_grad():
        for i in range(0, len(go_texts), 64):
            enc = tok(go_texts[i:i+64], return_tensors="pt", truncation=True,
                      max_length=128, padding=True).to(device)
            out.append(model(**enc).last_hidden_state[:, 0].cpu())

    result = torch.cat(out)  # [489, 768]
    torch.save(result, cache_path)
    del model; torch.cuda.empty_cache()
    print(f"  [go-emb] Saved: {result.shape} → {cache_path}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 4. GO DAG EDGES  (is_a edges among the 489 vocab terms)
# ══════════════════════════════════════════════════════════════════════════════

def build_go_dag_edges() -> list[tuple[int, int]]:
    """Parse GO OBO, return (parent_idx, child_idx) pairs within the 489-term vocab.

    is_a means: child IS A subtype of parent → parent entails child in hyperbolic space.
    We keep only edges where BOTH parent and child are in our 489-term vocabulary
    (induced subgraph), so the DAG loss trains on terms that already have embeddings.
    """
    import requests

    vocab    = json.load(open(CACHE_DIR / "go_mf_vocab.json"))
    id_to_idx = vocab["id_to_idx"]

    obo_path = CACHE_DIR / "go-basic.obo"
    if not obo_path.exists():
        print("  [dag] Downloading go-basic.obo ...")
        r = requests.get("http://purl.obolibrary.org/obo/go/go-basic.obo", timeout=120)
        r.raise_for_status()
        obo_path.write_text(r.text)
        print(f"  [dag] Saved {obo_path.stat().st_size/1e6:.1f} MB")

    # Parse OBO: collect (parent_go_id, child_go_id) from is_a lines
    raw_edges = []
    current_id, in_term = None, False
    with open(obo_path) as fh:
        for line in fh:
            line = line.strip()
            if line == "[Term]":
                in_term, current_id = True, None
            elif line.startswith("id:") and in_term:
                current_id = line.split("id:", 1)[1].strip().split()[0]
            elif line.startswith("is_a:") and in_term and current_id:
                parent_id = line.split("is_a:", 1)[1].strip().split()[0]
                raw_edges.append((parent_id, current_id))   # parent → child
            elif line == "" and in_term:
                in_term = False

    # Filter to induced subgraph over our 489 terms
    dag_edges = [
        (id_to_idx[p], id_to_idx[c])
        for p, c in raw_edges
        if p in id_to_idx and c in id_to_idx
    ]
    print(f"  [dag] {len(raw_edges)} total MF is_a edges → "
          f"{len(dag_edges)} within vocab (489 terms)")
    return dag_edges


# ══════════════════════════════════════════════════════════════════════════════
# 5. LORENTZ MATH
# ══════════════════════════════════════════════════════════════════════════════

def _time(x: torch.Tensor, curv: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(1.0 / curv + (x * x).sum(-1))


def exp_map0(x: torch.Tensor, curv: torch.Tensor) -> torch.Tensor:
    sqrt_c = curv.sqrt()
    xn = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    return torch.sinh((sqrt_c * xn).clamp(max=math.asinh(2.0 ** 15))) * x / xn


def pairwise_dist(x: torch.Tensor, y: torch.Tensor, curv: torch.Tensor) -> torch.Tensor:
    xt = _time(x, curv).unsqueeze(1)
    yt = _time(y, curv).unsqueeze(0)
    inner = x @ y.T - xt * yt
    return torch.acosh((-curv * inner).clamp(min=1.0 + EPS)) / curv.sqrt()


def poincare_radius(x: torch.Tensor, curv: torch.Tensor) -> torch.Tensor:
    t = _time(x, curv)
    return x.norm(dim=-1) / (t + 1.0 / curv.sqrt())


def half_aperture(x: torch.Tensor, curv: torch.Tensor, min_r: float = 0.1) -> torch.Tensor:
    arg = (2.0 * min_r / (curv.sqrt() * x.norm(dim=-1) + EPS)).clamp(max=1.0 - EPS)
    return torch.asin(arg)


def oxy_angle(x: torch.Tensor, y: torch.Tensor, curv: torch.Tensor) -> torch.Tensor:
    xt  = _time(x, curv)
    yt  = _time(y, curv)
    c   = curv * ((x * y).sum(-1) - xt * yt)
    num = yt + c * xt
    den = x.norm(dim=-1) * torch.sqrt((c ** 2 - 1).clamp(min=EPS))
    return torch.acos((num / (den + EPS)).clamp(min=-1.0 + EPS, max=1.0 - EPS))


def meru_entailment_loss(text_r: torch.Tensor, seq_r: torch.Tensor,
                          curv: torch.Tensor) -> torch.Tensor:
    """text (GO term = general) should entail seq (specific protein)."""
    angle    = oxy_angle(text_r, seq_r, curv)
    aperture = half_aperture(text_r, curv)
    return torch.clamp(angle - aperture, min=0.0).mean()


# ══════════════════════════════════════════════════════════════════════════════
# 5. MODEL
# ══════════════════════════════════════════════════════════════════════════════

class LorentzHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=in_dim ** -0.5)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(1.0 / math.sqrt(out_dim))))

    def forward(self, z: torch.Tensor, curv: torch.Tensor) -> torch.Tensor:
        return exp_map0(self.proj(z) * self.log_alpha.exp(), curv)

    def clamp(self):
        self.log_alpha.data.clamp_(max=0.0)


class EuclideanHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=in_dim ** -0.5)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(z), dim=-1)


class GOProbe(nn.Module):
    """Sequence encoder + GO-term text encoder in shared embedding space.

    Both heads share a single learnable curvature κ.
    The GO text embedding table (489 × BERT_DIM) is an external cache;
    we pass a batch of GO embeddings as the text side each step.
    """
    def __init__(self, seq_dim: int, text_dim: int, proj_dim: int,
                 geometry: str = "euclidean", learn_curv: bool = True):
        super().__init__()
        assert geometry in ("euclidean", "lorentz")
        self.geometry = geometry

        if geometry == "lorentz":
            if learn_curv:
                self.log_curv = nn.Parameter(torch.tensor(math.log(CURV_INIT)))
            else:
                self.register_buffer("log_curv", torch.tensor(math.log(CURV_INIT)))
            self.seq_head  = LorentzHead(seq_dim,  proj_dim)
            self.text_head = LorentzHead(text_dim, proj_dim)
        else:
            self.seq_head  = EuclideanHead(seq_dim,  proj_dim)
            self.text_head = EuclideanHead(text_dim, proj_dim)

        self.log_temp = nn.Parameter(torch.tensor(math.log(TEMPERATURE)))

    @property
    def curv(self) -> torch.Tensor:
        return self.log_curv.exp()

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(min=0.01, max=1.0)

    def encode_seq(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq_head(x, self.curv) if self.geometry == "lorentz" else self.seq_head(x)

    def encode_text(self, x: torch.Tensor) -> torch.Tensor:
        return self.text_head(x, self.curv) if self.geometry == "lorentz" else self.text_head(x)

    def sim_matrix(self, sr: torch.Tensor, tr: torch.Tensor) -> torch.Tensor:
        if self.geometry == "lorentz":
            return -pairwise_dist(sr, tr, self.curv)
        return sr @ tr.T

    def contrastive_loss(self, sr: torch.Tensor, tr: torch.Tensor) -> torch.Tensor:
        logits = self.sim_matrix(sr, tr) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    def clamp_params(self):
        if self.geometry == "lorentz":
            self.seq_head.clamp()
            self.text_head.clamp()
            # Keep κ in [0.1, 10] to prevent numerical overflow
            self.log_curv.data.clamp_(min=math.log(0.1), max=math.log(10.0))


# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAINING  (1 positive GO term sampled per protein per batch)
# ══════════════════════════════════════════════════════════════════════════════

def train_probe(probe: GOProbe,
                seq_feats: torch.Tensor,          # [N_train, 320]
                targets: torch.Tensor,            # [N_train, 489]  multi-hot
                go_embs: torch.Tensor,            # [489, 768]  pre-computed
                val_seq: torch.Tensor,
                val_tgt: torch.Tensor,
                device: torch.device,
                label: str,
                entail_weight: float = 0.0,       # λ for cross-modal MERU loss
                dag_weight: float = 0.0,          # λ for GO DAG entailment loss
                dag_edges: list | None = None,    # (parent_idx, child_idx) pairs
                n_epochs: int = EPOCHS,
                batch_size: int = BATCH_SIZE) -> GOProbe:

    seq_feats = seq_feats.to(device)
    go_embs   = go_embs.to(device)
    val_seq   = val_seq.to(device)
    val_tgt   = val_tgt  # kept on CPU for Fmax

    N = len(seq_feats)
    probe = probe.to(device)
    opt   = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    # Pre-compute pos_indices list (list of 1D tensors, one per protein)
    pos_indices = [targets[i].nonzero(as_tuple=True)[0] for i in range(N)]

    best_val   = float("inf")
    best_state = None
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        probe.train()
        tc, te, nb = 0.0, 0.0, 0

        # Shuffle
        perm = torch.randperm(N)
        for start in range(0, N - batch_size + 1, batch_size):
            idx = perm[start:start + batch_size]

            seq_b = seq_feats[idx]  # [B, 320]

            # Sample 1 positive GO term per protein
            go_idx = torch.stack([
                pos_indices[i.item()][torch.randint(len(pos_indices[i.item()]), (1,))]
                for i in idx
            ]).squeeze(1)  # [B]

            text_b = go_embs[go_idx]  # [B, 768]

            opt.zero_grad()
            sr = probe.encode_seq(seq_b)
            tr = probe.encode_text(text_b)

            Lc = probe.contrastive_loss(sr, tr)

            # Cross-modal MERU: GO text entails protein sequence
            Le = (meru_entailment_loss(tr, sr, probe.curv)
                  if entail_weight > 0 and probe.geometry == "lorentz"
                  else torch.tensor(0.0, device=device))

            # GO DAG entailment: parent GO term entails child GO term
            if dag_weight > 0 and probe.geometry == "lorentz" and dag_edges:
                dag_sample = torch.randint(len(dag_edges), (batch_size,))
                p_idx = torch.tensor([dag_edges[i][0] for i in dag_sample])
                c_idx = torch.tensor([dag_edges[i][1] for i in dag_sample])
                p_r = probe.encode_text(go_embs[p_idx])
                c_r = probe.encode_text(go_embs[c_idx])
                Ld = meru_entailment_loss(p_r, c_r, probe.curv)
            else:
                Ld = torch.tensor(0.0, device=device)

            (Lc + entail_weight * Le + dag_weight * Ld).backward()
            nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            opt.step()
            probe.clamp_params()

            tc += Lc.item(); te += Le.item(); nb += 1

        sched.step()

        # Validation: sample 1 GO term per val protein, compute InfoNCE
        probe.eval()
        with torch.no_grad():
            val_pos = [targets[i].nonzero(as_tuple=True)[0] for i in range(len(val_tgt))]
            val_go_idx = torch.stack([
                vp[0]  # always pick the first pos for a stable val signal
                for vp in val_pos
            ])
            val_go_embs = go_embs[val_go_idx.to(device)]
            vsr = probe.encode_seq(val_seq)
            vtr = probe.encode_text(val_go_embs)
            val_loss = probe.contrastive_loss(vsr, vtr).item()

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            extra = ""
            if probe.geometry == "lorentz":
                extra = f"  κ={probe.curv.item():.4f}"
                if entail_weight > 0:
                    extra += f"  L_ent={te/nb:.4f}"
                if dag_weight > 0:
                    extra += f"  L_dag={Ld.item():.4f}"
            print(f"  [{label}] ep {epoch:3}/{n_epochs}  "
                  f"contrast={tc/nb:.4f}  val={val_loss:.4f}{extra}")

    probe.load_state_dict(best_state)
    print(f"  [{label}] done in {time.time()-t0:.1f}s  best_val={best_val:.4f}")
    return probe


# ══════════════════════════════════════════════════════════════════════════════
# 7. EVALUATION  (Fmax — protein-centric, standard GO metric)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_sim_matrix(probe: GOProbe,
                       seq_feats: torch.Tensor,   # [N_test, 320]
                       go_embs: torch.Tensor,      # [489, 768]
                       device: torch.device) -> torch.Tensor:
    """Returns [N_test, 489] similarity matrix."""
    probe.eval()
    go_r = probe.encode_text(go_embs.to(device)).cpu()  # [489, PROJ_DIM]

    sim_rows = []
    for i in range(0, len(seq_feats), 256):
        sr = probe.encode_seq(seq_feats[i:i+256].to(device)).cpu()
        if probe.geometry == "lorentz":
            curv = probe.curv.detach().cpu()
            sim_rows.append(-pairwise_dist(sr, go_r, curv))
        else:
            sim_rows.append(sr @ go_r.T)
    return torch.cat(sim_rows)  # [N_test, 489]


def fmax_and_aupr(sim: torch.Tensor, targets: torch.Tensor) -> dict:
    """Compute Fmax, best threshold, and AUPR.

    Thresholds are quantile-derived from the sim matrix so the sweep adapts
    to the actual score range (works for Euclidean cosine and hyperbolic negdist).

    Fmax: protein-centric max F1 over all thresholds (CAFA standard).
    AUPR: area under protein-centric precision-recall curve.
    """
    sim_np  = sim.numpy().astype(np.float32)
    tgt_np  = targets.numpy().astype(np.float32)

    qs = np.linspace(1, 99, 199)   # 199 thresholds, 1st–99th percentile
    thresholds = np.percentile(sim_np, qs)

    best_f, best_t = 0.0, float(thresholds[0])
    precs, recs = [], []

    for t in thresholds:
        preds = (sim_np > t).astype(np.float32)
        tp    = (preds * tgt_np).sum(1)
        prec  = float((tp / np.maximum(preds.sum(1), 1)).mean())
        rec   = float((tp / np.maximum(tgt_np.sum(1), 1)).mean())
        f1    = 2.0 * prec * rec / (prec + rec + 1e-8)
        precs.append(prec)
        recs.append(rec)
        if f1 > best_f:
            best_f, best_t = f1, float(t)

    # AUPR: integrate P-R curve (sort by recall ascending)
    order  = np.argsort(recs)
    s_rec  = np.array(recs)[order]
    s_prec = np.array(precs)[order]
    aupr   = float(np.trapz(s_prec, s_rec))

    return {"fmax": best_f, "best_threshold": best_t, "aupr": aupr}


def poincare_analysis(probe: GOProbe,
                      seq_feats: torch.Tensor,
                      go_embs: torch.Tensor,
                      targets: torch.Tensor,
                      device: torch.device) -> dict:
    """Poincaré radius statistics for sequences and GO terms."""
    probe.eval()
    with torch.no_grad():
        curv = probe.curv.detach().cpu()
        sr   = probe.encode_seq(seq_feats[:1000].to(device)).cpu()
        go_r = probe.encode_text(go_embs.to(device)).cpu()

    seq_r = poincare_radius(sr, curv)
    go_r_ = poincare_radius(go_r, curv)
    n_pos = (targets[:1000] > 0.5).sum(1).float()

    # Spearman correlation: #GO annotations vs Poincaré radius
    # (proteins with more GO terms should sit further from origin if they are
    #  more functionally specific in the multi-label sense)
    from scipy.stats import spearmanr
    rho, pval = spearmanr(n_pos.numpy(), seq_r.numpy())

    return {
        "mean_seq_radius":  seq_r.mean().item(),
        "mean_go_radius":   go_r_.mean().item(),
        "spearman_rho":     float(rho),
        "spearman_p":       float(pval),
        "go_radius_mean":   go_r_.mean().item(),
        "go_radius_std":    go_r_.std().item(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int,   default=60)
    parser.add_argument("--proj-dim", type=int,   default=256)
    parser.add_argument("--batch",    type=int,   default=128)
    args = parser.parse_args()

    n_epochs   = args.epochs
    proj_dim   = args.proj_dim
    batch_size = args.batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  epochs={n_epochs}  proj_dim={proj_dim}  batch={batch_size}\n")

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("[1/4] Loading / downloading dataset ...")
    splits = download_and_decode()

    # ── 2. ESM-2 features ────────────────────────────────────────────────────
    print("\n[2/4] Computing ESM-2 features ...")
    esm_feats = compute_esm2_features(splits, device)

    # Filter proteins with no positive GO terms (can't form training pairs)
    def _filter(feats, tgts):
        has_pos = (tgts > 0.5).any(1)
        return feats[has_pos], tgts[has_pos]

    seq_train, tgt_train = _filter(esm_feats["train"],      splits["train"]["targets"])
    seq_val,   tgt_val   = _filter(esm_feats["validation"], splits["validation"]["targets"])
    seq_test  = esm_feats["test"]
    tgt_test  = splits["test"]["targets"]
    print(f"  After filtering: train={len(seq_train)}, val={len(seq_val)}, test={len(seq_test)}")

    # ── 3. GO term embeddings + DAG edges ────────────────────────────────────
    print("\n[3/4] Computing / loading GO term embeddings and DAG edges ...")
    go_embs   = compute_go_embeddings(device)   # [489, 768]
    dag_edges = build_go_dag_edges()            # list of (parent_idx, child_idx)

    # ── 4. Train + evaluate each condition ────────────────────────────────────
    print("\n[4/4] Training probes ...\n")
    # (label, geometry, learn_curv, cross-modal λ, dag λ)
    conditions = [
        ("Euclidean",            "euclidean", True,  0.0, 0.0),
        ("Hyp",                  "lorentz",   True,  0.0, 0.0),
        ("Hyp+MERU λ=0.5",       "lorentz",   True,  0.5, 0.0),
        ("Hyp+MERU+DAG λ=0.5",   "lorentz",   True,  0.5, 0.5),
    ]

    results = {}
    slug = lambda s: s.replace("+", "p").replace(" ", "_").replace("=", "").replace(".", "").replace("λ", "lam")
    for label, geometry, learn_curv, lam, lam_dag in conditions:
        print(f"── {label} ────────────────────────────────")
        probe = GOProbe(ESM_DIM, BERT_DIM, proj_dim, geometry, learn_curv)
        probe = train_probe(
            probe,
            seq_train, tgt_train, go_embs,
            seq_val,   tgt_val,
            device, label,
            entail_weight=lam,
            dag_weight=lam_dag,
            dag_edges=dag_edges if lam_dag > 0 else None,
            n_epochs=n_epochs,
            batch_size=batch_size,
        )

        # Save checkpoint
        ckpt_path = CKPT_DIR / f"probe_{slug(label)}.pt"
        torch.save({
            "label":      label,
            "geometry":   geometry,
            "state_dict": probe.state_dict(),
            "proj_dim":   proj_dim,
            "n_epochs":   n_epochs,
            "lam":        lam,
        }, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

        # Evaluate on test set
        sim = compute_sim_matrix(probe, seq_test, go_embs, device)
        metrics = fmax_and_aupr(sim, tgt_test)

        row: dict = {k: round(v, 4) for k, v in metrics.items()}

        if geometry == "lorentz":
            row["kappa"] = round(probe.curv.item(), 4)
            anl = poincare_analysis(probe, seq_test, go_embs, tgt_test, device)
            row.update({k: round(v, 4) if isinstance(v, float) else v
                        for k, v in anl.items()})

        results[label] = row
        print(f"  Fmax={metrics['fmax']:.4f}  AUPR={metrics['aupr']:.4f}  "
              f"(threshold={metrics['best_threshold']:.3f})")
        if geometry == "lorentz":
            print(f"  κ={row['kappa']:.4f}  "
                  f"seq_r={anl['mean_seq_radius']:.3f}  "
                  f"go_r={anl['mean_go_radius']:.3f}  "
                  f"spearman_ρ={anl['spearman_rho']:.3f}")
        print()

    # ── Print table ───────────────────────────────────────────────────────────
    w = 22
    print("=" * 72)
    print(f"{'Model':<{w}}  {'Fmax':>8}  {'AUPR':>8}  {'Threshold':>10}")
    print("-" * 72)
    for label, row in results.items():
        print(f"  {label:<{w-2}}  {row['fmax']:>8.4f}  {row['aupr']:>8.4f}  "
              f"{row['best_threshold']:>10.3f}")
    print("=" * 72)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "results_go.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
