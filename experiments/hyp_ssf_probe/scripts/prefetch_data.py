#!/usr/bin/env python3
"""
Run this on the HPRC login node (no GPU needed) before submitting a job.
Downloads all model weights and dataset CSVs so compute nodes can run offline.

Usage:
    conda activate hypalign
    python experiments/hyp_ssf_probe/scripts/prefetch_data.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "experiments" / "hyp_ssf_probe" / "cache"
CACHE_DIR.mkdir(exist_ok=True)

ESM2_HF      = "facebook/esm2_t33_650M_UR50D"
PUBMEDBERT_HF = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
HF_DATASET   = "mila-intel/ProtST-GeneOntology-MF"

def step(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")

# 1. HuggingFace model weights
step(f"Downloading {ESM2_HF}")
from transformers import EsmTokenizer, EsmModel
EsmTokenizer.from_pretrained(ESM2_HF)
EsmModel.from_pretrained(ESM2_HF)
print("  done.")

step(f"Downloading {PUBMEDBERT_HF}")
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained(PUBMEDBERT_HF)
AutoModel.from_pretrained(PUBMEDBERT_HF)
print("  done.")

# 2. Dataset CSVs from HuggingFace Hub
step("Downloading ProtST GO-MF dataset CSVs")
import requests
base = "https://huggingface.co/datasets/mila-intel/ProtST-GeneOntology-MF/resolve/main"
files = {
    "gene_ontology_mf_train.csv": f"{base}/gene_ontology_mf_train.csv",
    "gene_ontology_mf_valid.csv": f"{base}/gene_ontology_mf_validation.csv",
    "gene_ontology_mf_test.csv":  f"{base}/gene_ontology_mf_test.csv",
}
for fname, url in files.items():
    dst = CACHE_DIR / fname
    if dst.exists():
        print(f"  {fname} already cached, skipping.")
        continue
    print(f"  Downloading {fname} ...", end=" ", flush=True)
    r = requests.get(url, timeout=300, stream=True)
    r.raise_for_status()
    with open(dst, "wb") as f:
        for chunk in r.iter_content(65536):
            f.write(chunk)
    print(f"done ({dst.stat().st_size // 1024} KB)")

# 3. GO ontology OBO file
step("Downloading go-basic.obo")
obo = CACHE_DIR / "go-basic.obo"
if obo.exists():
    print("  Already cached, skipping.")
else:
    print("  Downloading ...", end=" ", flush=True)
    r = requests.get("http://purl.obolibrary.org/obo/go/go-basic.obo", timeout=300)
    r.raise_for_status()
    obo.write_bytes(r.content)
    print(f"done ({obo.stat().st_size // 1024} KB)")

print("\nAll downloads complete. Submit the SLURM job now.")
print(f"Cache location: {CACHE_DIR}")
