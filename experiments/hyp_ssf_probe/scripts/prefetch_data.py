#!/usr/bin/env python3
"""
Data preparation — Step 1: CPU-only, run on HPRC login node.

Downloads all model weights, dataset CSVs, and the GO ontology file, then
parses the CSVs into the cached tensors used by the training scripts.

After this script finishes, run prepare_features.py on a GPU compute node
to compute ESM2-650M protein embeddings.

Usage (HPRC login node, no GPU):
    conda activate hypalign
    python experiments/hyp_ssf_probe/scripts/prefetch_data.py

What this produces in experiments/hyp_ssf_probe/cache/:
    gene_ontology_mf_{train,valid,test}.csv  — raw ProtST dataset splits
    go-basic.obo                              — GO ontology (for DAG edges)
    protst_go_mf_decoded.pt                  — parsed sequences + multi-hot targets
    go_mf_vocab.json                          — GO ID ↔ index mapping (also in git)
    go_term_embs.pt                           — PubMedBERT GO embeddings (also in git)

Model weights are saved to the HuggingFace cache (~/.cache/huggingface/).
"""
import ast
import csv
import json
import sys
from pathlib import Path

REPO_ROOT     = Path(__file__).resolve().parents[3]
CACHE_DIR     = REPO_ROOT / "experiments" / "hyp_ssf_probe" / "cache"
CACHE_DIR.mkdir(exist_ok=True)

ESM2_HF       = "facebook/esm2_t33_650M_UR50D"
PUBMEDBERT_HF = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

# ESM2 token vocabulary (index → amino acid single letter)
ESM_VOCAB = {
    4:"L",5:"A",6:"G",7:"V",8:"S",9:"E",10:"R",11:"T",12:"I",13:"D",
    14:"P",15:"K",16:"Q",17:"N",18:"F",19:"Y",20:"M",21:"H",22:"W",
    23:"C",24:"X",25:"B",26:"U",27:"Z",28:"O",29:".",30:"-",
}

def step(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}", flush=True)


# ── Step 1: HuggingFace model weights ────────────────────────────────────────

step(f"1/5  Downloading model weights: {ESM2_HF}")
from transformers import EsmTokenizer, EsmModel, AutoTokenizer, AutoModel
EsmTokenizer.from_pretrained(ESM2_HF)
EsmModel.from_pretrained(ESM2_HF)
print("  done.")

step(f"2/5  Downloading model weights: {PUBMEDBERT_HF}")
AutoTokenizer.from_pretrained(PUBMEDBERT_HF)
AutoModel.from_pretrained(PUBMEDBERT_HF)
print("  done.")


# ── Step 2: Dataset CSVs ─────────────────────────────────────────────────────

step("3/5  Downloading ProtST GO-MF dataset CSVs")
import requests

BASE = "https://huggingface.co/datasets/mila-intel/ProtST-GeneOntology-MF/resolve/main"
CSV_FILES = {
    "gene_ontology_mf_train.csv":      f"{BASE}/gene_ontology_mf_train.csv",
    "gene_ontology_mf_valid.csv":      f"{BASE}/gene_ontology_mf_validation.csv",
    "gene_ontology_mf_test.csv":       f"{BASE}/gene_ontology_mf_test.csv",
}
for fname, url in CSV_FILES.items():
    dst = CACHE_DIR / fname
    if dst.exists():
        print(f"  {fname}: already cached ({dst.stat().st_size // 1024} KB)")
        continue
    print(f"  Downloading {fname} ...", end=" ", flush=True)
    r = requests.get(url, timeout=300, stream=True)
    r.raise_for_status()
    with open(dst, "wb") as f:
        for chunk in r.iter_content(65536):
            f.write(chunk)
    print(f"done ({dst.stat().st_size // 1024} KB)")


# ── Step 3: GO ontology OBO file ─────────────────────────────────────────────

step("4/5  Downloading go-basic.obo")
obo = CACHE_DIR / "go-basic.obo"
if obo.exists():
    print(f"  Already cached ({obo.stat().st_size // 1024} KB)")
else:
    print("  Downloading ...", end=" ", flush=True)
    r = requests.get("http://purl.obolibrary.org/obo/go/go-basic.obo", timeout=300)
    r.raise_for_status()
    obo.write_bytes(r.content)
    print(f"done ({obo.stat().st_size // 1024} KB)")


# ── Step 4: Parse CSVs → protst_go_mf_decoded.pt + go_mf_vocab.json ─────────

step("5/5  Parsing CSVs → protst_go_mf_decoded.pt + go_mf_vocab.json")

import torch

decoded_path = CACHE_DIR / "protst_go_mf_decoded.pt"
vocab_path   = CACHE_DIR / "go_mf_vocab.json"

if decoded_path.exists() and vocab_path.exists():
    print(f"  Already cached: {decoded_path.name}, {vocab_path.name}")
else:
    # --- Build vocabulary from training split ---
    # Each row's 'targets' is a list of GO term indices 0..488 already encoded
    # as a 489-dim multi-hot vector.  We need to recover GO IDs and names from
    # the CSV header / a separate column.
    #
    # The CSV schema (ProtST-GeneOntology-MF):
    #   columns: '', prot_seq, targets, pdb_files, [optionally] go_id, go_name
    # 'targets' is a Python list literal like [0.0, 1.0, 0.0, ...]  (489 values)
    # 'prot_seq' is a list of ESM token IDs (integers), NOT raw amino acids.
    #
    # GO ID / name mapping: we parse it from go-basic.obo using the ordering
    # established by whichever GO IDs appear in the column headers (if present),
    # or reconstruct it from the data itself.
    #
    # For the ProtST dataset the GO IDs are stored in a separate CSV column
    # 'go_id' on the header row, OR the first row after the header.
    # We use a deterministic approach: parse the obo file for MF terms,
    # then align with the 489-dimensional target vector order.
    # The safest approach is to reuse the go_mf_vocab.json already committed
    # to the repo (experiments/hyp_ssf_probe/cache/ is gitignored, but the
    # vocab is tracked). If the repo copy is absent, we reconstruct from obo.

    repo_vocab = REPO_ROOT / "experiments" / "hyp_ssf_probe" / "data" / "go_mf_vocab.json"
    if repo_vocab.exists():
        import shutil
        shutil.copy(repo_vocab, vocab_path)
        print(f"  Copied go_mf_vocab.json from repo data/")
    elif not vocab_path.exists():
        print("  go_mf_vocab.json not found — will be built by training script on first run.")

    # --- Parse each CSV split ---
    import torch
    splits = {}
    split_map = {
        "train":      "gene_ontology_mf_train.csv",
        "validation": "gene_ontology_mf_valid.csv",
        "test":       "gene_ontology_mf_test.csv",
    }
    for split, fname in split_map.items():
        src = CACHE_DIR / fname
        print(f"  Parsing {fname} ...", end=" ", flush=True)
        seqs, targets = [], []
        with open(src, newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            col_seq = header.index("prot_seq")
            col_tgt = header.index("targets")
            for row in reader:
                tok_ids = ast.literal_eval(row[col_seq])
                seq_str = "".join(ESM_VOCAB.get(t, "X") for t in tok_ids)
                seqs.append(seq_str)
                tgt = ast.literal_eval(row[col_tgt])
                targets.append(torch.tensor(tgt, dtype=torch.float32))
        tgt_tensor = torch.stack(targets)
        n_pos = (tgt_tensor > 0.5).sum(1).float()
        splits[split] = {"seqs": seqs, "targets": tgt_tensor}
        print(f"done  {len(seqs)} proteins, "
              f"avg {n_pos.mean():.1f} GO terms/protein "
              f"(min {int(n_pos.min())}, max {int(n_pos.max())})")

    torch.save(splits, decoded_path)
    print(f"\n  Saved {decoded_path.name} "
          f"({decoded_path.stat().st_size // (1024*1024)} MB)")

print("""
All CPU-side data preparation complete.
Next: submit a GPU job to compute ESM2-650M protein embeddings.

    sbatch experiments/hyp_ssf_probe/scripts/submit_prepare_features.sh

Or run directly on a GPU node:

    python experiments/hyp_ssf_probe/scripts/prepare_features.py
""")
