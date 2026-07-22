#!/usr/bin/env python3
"""Build ProtST-style GO namespace caches from the TorchDrug GeneOntology archive.

The existing HypAlign cache is the 489-term molecular-function subset. This
script rebuilds the same split structure for MF, BP, or CC directly from
GeneOntology.zip:

    {
      "train": {"ids": [...], "seqs": [...], "targets": FloatTensor[N, C]},
      "validation": ...,
      "test": ...
    }

The target order follows nrPDB-GO_annot.tsv for the requested namespace.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "experiments" / "hyp_ssf_probe" / "cache"
SHARED_ROOT = Path(os.environ.get("HYPALIGN_SHARED_ROOT", REPO_ROOT))
CACHE_ROOT = Path(os.environ.get("HYPALIGN_CACHE_ROOT", SHARED_ROOT / ".cache"))
GENEONTOLOGY_ZIP = CACHE_ROOT / "hypalign_data" / "GeneOntology.zip"

NAMESPACE_META = {
    "mf": {
        "long": "molecular_function",
        "column": 1,
        "terms_row": 1,
        "names_row": 3,
    },
    "bp": {
        "long": "biological_process",
        "column": 2,
        "terms_row": 5,
        "names_row": 7,
    },
    "cc": {
        "long": "cellular_component",
        "column": 3,
        "terms_row": 9,
        "names_row": 11,
    },
}


def read_lines(zf: zipfile.ZipFile, name: str) -> list[str]:
    return zf.read(name).decode("utf-8").splitlines()


def parse_fasta(lines: list[str]) -> dict[str, str]:
    seqs: dict[str, list[str]] = {}
    current_id: str | None = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            current_id = line[1:].split()[0]
            seqs[current_id] = []
        elif current_id is not None:
            seqs[current_id].append(line)
    return {key: "".join(parts) for key, parts in seqs.items()}


def parse_annotation_table(
    zf: zipfile.ZipFile,
    namespace: str,
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    meta = NAMESPACE_META[namespace]
    with zf.open("GeneOntology/nrPDB-GO_annot.tsv") as fh:
        rows = [fh.readline().decode("utf-8").rstrip("\n").split("\t")
                for _ in range(13)]
        go_ids = [x.strip() for x in rows[meta["terms_row"]] if x.strip()]
        go_names = [x.strip() for x in rows[meta["names_row"]] if x.strip()]
        if len(go_ids) != len(go_names):
            raise ValueError(
                f"{namespace}: {len(go_ids)} GO IDs but {len(go_names)} names"
            )

        labels: dict[str, list[str]] = {}
        col = meta["column"]
        for raw in fh:
            parts = raw.decode("utf-8").rstrip("\n").split("\t")
            if len(parts) <= col:
                continue
            chain_id = parts[0].strip()
            labels[chain_id] = [
                x.strip() for x in parts[col].strip().split(",") if x.strip()
            ]
    return go_ids, go_names, labels


def build_split(
    split_name: str,
    split_ids: list[str],
    seqs_by_id: dict[str, str],
    labels_by_id: dict[str, list[str]],
    id_to_idx: dict[str, int],
) -> dict:
    ids, seqs, targets = [], [], []
    for chain_id in split_ids:
        if chain_id not in seqs_by_id:
            continue
        target = torch.zeros(len(id_to_idx), dtype=torch.float32)
        for go_id in labels_by_id.get(chain_id, []):
            idx = id_to_idx.get(go_id)
            if idx is not None:
                target[idx] = 1.0
        ids.append(chain_id)
        seqs.append(seqs_by_id[chain_id])
        targets.append(target)
    if not targets:
        raise ValueError(f"{split_name}: no examples were built")
    return {"ids": ids, "seqs": seqs, "targets": torch.stack(targets)}


def summarize_split(name: str, data: dict) -> dict:
    targets = data["targets"]
    n_pos = targets.sum(1)
    nonempty = int((n_pos > 0).sum().item())
    return {
        "proteins": len(data["seqs"]),
        "nonempty": nonempty,
        "avg_pos": round(float(n_pos.float().mean().item()), 4),
        "min_pos": int(n_pos.min().item()),
        "max_pos": int(n_pos.max().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", choices=sorted(NAMESPACE_META), required=True)
    parser.add_argument("--geneontology-zip", type=Path, default=GENEONTOLOGY_ZIP)
    parser.add_argument("--out-prefix", default=None,
                        help="Default: protst_go_<namespace>")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_prefix = args.out_prefix or f"protst_go_{args.namespace}"

    if not args.geneontology_zip.exists():
        raise FileNotFoundError(args.geneontology_zip)

    with zipfile.ZipFile(args.geneontology_zip) as zf:
        go_ids, go_names, labels_by_id = parse_annotation_table(zf, args.namespace)
        seqs_by_id = parse_fasta(read_lines(zf, "GeneOntology/nrPDB-GO_sequences.fasta"))
        test_seqs_by_id = parse_fasta(
            read_lines(zf, "GeneOntology/nrPDB-GO_test_sequences.fasta")
        )
        seqs_by_id.update(test_seqs_by_id)

        split_ids = {
            "train": read_lines(zf, "GeneOntology/nrPDB-GO_train.txt"),
            "validation": read_lines(zf, "GeneOntology/nrPDB-GO_valid.txt"),
            "test": read_lines(zf, "GeneOntology/nrPDB-GO_test.txt"),
        }

    id_to_idx = {go_id: i for i, go_id in enumerate(go_ids)}
    splits = {
        split: build_split(split, ids, seqs_by_id, labels_by_id, id_to_idx)
        for split, ids in split_ids.items()
    }

    vocab = {
        "namespace": args.namespace,
        "namespace_name": NAMESPACE_META[args.namespace]["long"],
        "go_ids": go_ids,
        "go_names": go_names,
        "id_to_idx": id_to_idx,
    }
    stats = {
        "namespace": args.namespace,
        "namespace_name": NAMESPACE_META[args.namespace]["long"],
        "num_terms": len(go_ids),
        "splits": {name: summarize_split(name, data)
                   for name, data in splits.items()},
    }

    splits_path = CACHE_DIR / f"{out_prefix}_decoded.pt"
    vocab_path = CACHE_DIR / f"go_{args.namespace}_vocab.json"
    stats_path = CACHE_DIR / f"{out_prefix}_stats.json"
    torch.save(splits, splits_path)
    vocab_path.write_text(json.dumps(vocab, indent=2) + "\n")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")

    print(f"Saved {splits_path}")
    print(f"Saved {vocab_path}")
    print(f"Saved {stats_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
