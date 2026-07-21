#!/usr/bin/env python3
"""
Prepare HypAlign data and model caches on a Grace login node.

This script avoids GPU work. It downloads the ProtST GO-MF CSV splits, decodes
the ESM-tokenized protein sequences into amino-acid strings, writes
protst_go_mf_decoded.pt, downloads go-basic.obo, and optionally warms the
Hugging Face cache for the ESM-2/PubMedBERT models used by the experiments.

Usage:
    conda activate hypalign
    python experiments/hyp_ssf_probe/scripts/prefetch_data.py
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import zipfile
from pathlib import Path

import requests
import torch
from huggingface_hub import HfApi, snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "experiments" / "hyp_ssf_probe" / "cache"
SHARED_ROOT = Path(os.environ.get("HYPALIGN_SHARED_ROOT", REPO_ROOT))
CACHE_ROOT = Path(os.environ.get("HYPALIGN_CACHE_ROOT", SHARED_ROOT / ".cache"))
HF_CACHE_DIR = Path(os.environ.get("HF_HOME", CACHE_ROOT / "huggingface"))
CACHE_DIR.mkdir(exist_ok=True)
HF_CACHE_DIR.mkdir(exist_ok=True)

DEFAULT_ESM2_HF = "facebook/esm2_t33_650M_UR50D"
DEFAULT_PUBMEDBERT_HF = "NeuML/pubmedbert-base-embeddings"
HF_DATASET = "mila-intel/ProtST-GeneOntology-MF"
HF_BASE = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main"
GENEONTOLOGY_URL = "https://zenodo.org/records/6622158/files/GeneOntology.zip"
GENEONTOLOGY_ZIP = CACHE_ROOT / "hypalign_data" / "GeneOntology.zip"

SPLIT_FILES = {
    "train": "gene_ontology_mf_train.csv",
    "validation": "gene_ontology_mf_valid.csv",
    "test": "gene_ontology_mf_test.csv",
}

ESM_VOCAB = {
    4: "A", 5: "R", 6: "N", 7: "D", 8: "C", 9: "Q", 10: "E", 11: "G",
    12: "H", 13: "I", 14: "L", 15: "K", 16: "M", 17: "F", 18: "P",
    19: "S", 20: "T", 21: "W", 22: "Y", 23: "V",
}


def step(msg: str) -> None:
    print(f"\n{'=' * 60}\n{msg}\n{'=' * 60}", flush=True)


def download_file(url: str, dst: Path, timeout: int = 300) -> None:
    if dst.exists():
        print(f"  {dst.name} already cached, skipping.")
        return
    print(f"  Downloading {dst.name} ...", end=" ", flush=True)
    r = requests.get(url, timeout=timeout, stream=True)
    r.raise_for_status()
    with open(dst, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            if chunk:
                f.write(chunk)
    print(f"done ({dst.stat().st_size / 1e6:.1f} MB)")


def download_csv_splits() -> None:
    step("Downloading ProtST GO-MF CSV splits")
    for fname in SPLIT_FILES.values():
        download_file(f"{HF_BASE}/{fname}", CACHE_DIR / fname)


def decode_split(split_name: str) -> dict:
    path = CACHE_DIR / SPLIT_FILES[split_name]
    seqs, targets = [], []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col_seq = header.index("prot_seq")
        col_tgt = header.index("targets")
        for row in reader:
            tok = ast.literal_eval(row[col_seq])
            seqs.append("".join(ESM_VOCAB.get(t, "X") for t in tok))
            targets.append(torch.tensor(ast.literal_eval(row[col_tgt]),
                                        dtype=torch.float32))
    targets_t = torch.stack(targets)
    n_pos = (targets_t > 0.5).sum(1)
    print(f"  {split_name:<10} {len(seqs):>6} proteins  "
          f"avg_pos={n_pos.float().mean():.2f}  "
          f"min={int(n_pos.min())}  max={int(n_pos.max())}")
    return {"seqs": seqs, "targets": targets_t}


def build_decoded_cache(force: bool = False) -> None:
    step("Building decoded torch cache")
    out = CACHE_DIR / "protst_go_mf_decoded.pt"
    if out.exists() and not force:
        print(f"  {out.name} already exists, skipping.")
        return
    splits = {split: decode_split(split) for split in SPLIT_FILES}
    torch.save(splits, out)
    print(f"  Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


def download_go_obo() -> None:
    step("Downloading go-basic.obo")
    download_file(
        "http://purl.obolibrary.org/obo/go/go-basic.obo",
        CACHE_DIR / "go-basic.obo",
    )


def parse_go_names_from_obo() -> dict[str, str]:
    obo_path = CACHE_DIR / "go-basic.obo"
    if not obo_path.exists():
        download_go_obo()
    names = {}
    current_id, current_name, in_term = None, None, False
    with open(obo_path) as fh:
        for line in fh:
            line = line.strip()
            if line == "[Term]":
                current_id, current_name, in_term = None, None, True
            elif in_term and line.startswith("id:"):
                current_id = line.split("id:", 1)[1].strip()
            elif in_term and line.startswith("name:"):
                current_name = line.split("name:", 1)[1].strip()
            elif line == "" and in_term:
                if current_id and current_name:
                    names[current_id] = current_name
                current_id, current_name, in_term = None, None, False
    if current_id and current_name:
        names[current_id] = current_name
    return names


def prepare_go_mf_vocab(zip_path: Path | None = None) -> None:
    step("Preparing GO-MF vocabulary from TorchDrug GeneOntology metadata")
    zip_path = zip_path or GENEONTOLOGY_ZIP
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        download_file(GENEONTOLOGY_URL, zip_path, timeout=1800)

    with zipfile.ZipFile(zip_path) as zf:
        annot_name = None
        for name in zf.namelist():
            if name.endswith("GeneOntology/nrPDB-GO_annot.tsv"):
                annot_name = name
                break
        if annot_name is None:
            raise FileNotFoundError("nrPDB-GO_annot.tsv not found in GeneOntology.zip")
        rows = []
        with zf.open(annot_name) as fh:
            for raw in fh:
                rows.append(raw.decode("utf-8").rstrip("\n").split("\t"))
                if len(rows) >= 12:
                    break

    # The ProtST target vectors follow the ordered MF labels in this archive.
    go_ids = [go_id.strip() for go_id in rows[1]]
    if len(go_ids) != 489:
        raise ValueError(f"Expected 489 MF GO terms, found {len(go_ids)}")
    archive_names = [name.strip() for name in rows[3]]
    if len(archive_names) != len(go_ids):
        id_to_name = parse_go_names_from_obo()
        go_names = [id_to_name.get(go_id, go_id) for go_id in go_ids]
    else:
        go_names = archive_names
    vocab = {
        "go_ids": go_ids,
        "go_names": go_names,
        "id_to_idx": {go_id: i for i, go_id in enumerate(go_ids)},
    }
    out = CACHE_DIR / "go_mf_vocab.json"
    out.write_text(json_dumps(vocab))
    print(f"  Saved {out}")


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, indent=2) + "\n"


def weight_patterns_for_repo(model_id: str) -> list[str]:
    files = set(HfApi().list_repo_files(model_id))
    if "model.safetensors" in files:
        return ["model.safetensors"]
    if any(f.startswith("model-") and f.endswith(".safetensors") for f in files):
        return ["model.safetensors.index.json", "model-*.safetensors"]
    if "pytorch_model.bin" in files:
        return ["pytorch_model.bin"]
    if any(f.startswith("pytorch_model-") and f.endswith(".bin") for f in files):
        return ["pytorch_model.bin.index.json", "pytorch_model-*.bin"]
    return ["*.safetensors", "*.bin"]


def warm_hf_model_cache(model_id: str, cache_dir: Path) -> None:
    step(f"Warming Hugging Face cache: {model_id}")
    weight_patterns = weight_patterns_for_repo(model_id)
    snapshot_download(
        repo_id=model_id,
        allow_patterns=[
            "*.json", "*.txt", "*.model",
            "vocab.*", "tokenizer.*", "special_tokens_map.json",
            *weight_patterns,
        ],
        cache_dir=cache_dir,
        max_workers=1,
    )
    print(f"  weights: {', '.join(weight_patterns)}")
    print("  done.")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare HypAlign data caches.")
    parser.add_argument("--esm-model", default=DEFAULT_ESM2_HF)
    parser.add_argument("--pubmedbert-model", default=DEFAULT_PUBMEDBERT_HF)
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-pubmedbert", action="store_true")
    parser.add_argument("--hf-cache-dir", type=Path, default=HF_CACHE_DIR)
    parser.add_argument("--prepare-go-vocab", action="store_true")
    parser.add_argument("--geneontology-zip", type=Path, default=None)
    parser.add_argument("--force-decode", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_csv_splits()
    build_decoded_cache(force=args.force_decode)
    download_go_obo()
    if args.prepare_go_vocab:
        prepare_go_mf_vocab(args.geneontology_zip)
    if not args.skip_models:
        args.hf_cache_dir.mkdir(parents=True, exist_ok=True)
        warm_hf_model_cache(args.esm_model, args.hf_cache_dir)
        if not args.skip_pubmedbert:
            warm_hf_model_cache(args.pubmedbert_model, args.hf_cache_dir)
    print(f"\nAll requested caches are ready under: {CACHE_DIR}")
    if not args.skip_models:
        print(f"Hugging Face cache: {args.hf_cache_dir}")


if __name__ == "__main__":
    main()
