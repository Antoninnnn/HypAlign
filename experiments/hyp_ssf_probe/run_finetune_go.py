#!/usr/bin/env python3
"""
Strong Euclidean baseline: fine-tuned ESM-2 for GO-MF function prediction.

Key improvements over the frozen probe (v1):
  1. ESM-2 150M fine-tuned end-to-end (vs ESM-2 8M frozen)
  2. Direct multi-label BCE loss (vs InfoNCE)
     → no false-negative problem from multi-label sampling
     → gradients flow to ESM-2 directly from GO supervision
  3. Asymmetric loss (ASL) for long-tailed GO class distribution
  4. Positive class reweighting for severe imbalance
  5. Separate LR: backbone 5e-5, head 3e-4  with warmup + cosine schedule

Architecture:
  ESM-2 150M → mean-pool → Linear(640, 489) → sigmoid → BCE

Evaluation: same Fmax + AUPR as v1 (protein-centric, quantile thresholds).

Run:
    conda activate pannot-infer
    cd /home/yining_yang/NLP/hyperbolic/experiments/hyp_ssf_probe
    python run_finetune_go.py
"""

from __future__ import annotations
import argparse, json, math, os, sys, time
import atexit
from pathlib import Path

# Avoid CUDA memory allocator assertion failures on some driver versions
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import EsmTokenizer, EsmModel
from transformers import get_cosine_schedule_with_warmup

REPO_ROOT  = Path(__file__).resolve().parent
CACHE_DIR  = REPO_ROOT / "cache"
CKPT_DIR   = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"
CKPT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ── Distributed helpers ───────────────────────────────────────────────────────

def setup_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    atexit.register(cleanup_distributed)
    return True, rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def log(msg: str = "") -> None:
    if is_rank0():
        print(msg, flush=True)


# ── Config ────────────────────────────────────────────────────────────────────
ESM_MODEL   = "facebook/esm2_t30_150M_UR50D"   # 150M params, 640-dim
N_GO        = 489
PROJ_DIM    = 640        # same as ESM-2 150M hidden dim (no bottleneck)
N_EPOCHS    = 50
BATCH_SIZE   = 8           # micro-batch; effective batch = 8 * GRAD_ACCUM
GRAD_ACCUM   = 4           # effective batch = 32
BACKBONE_LR  = 5e-5
HEAD_LR      = 3e-4
WEIGHT_DECAY = 1e-4
MAX_LEN      = 512
WARMUP_FRAC = 0.05       # 5% of total steps
GRAD_CLIP   = 1.0
POS_WEIGHT_CAP = 10.0    # cap inverse-frequency weights
LOSS_NAME   = "asl"      # choices: asl, bce, bce_unweighted
USE_AMP     = True
NUM_WORKERS = 2

# Asymmetric Loss hyperparameters (ASL)
ASL_GAMMA_POS = 0        # focusing on positives (0 = no focusing)
ASL_GAMMA_NEG = 4        # hard down-weighting of easy negatives
ASL_CLIP      = 0.05     # probability margin shift for negatives


# ── Asymmetric Loss ───────────────────────────────────────────────────────────

def asymmetric_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """ASL for long-tailed multi-label classification.

    Decouples the focusing exponents for positive (γ+) and negative (γ-)
    samples, and applies a probability shift (clip) on negatives to avoid
    the easy-negative domination problem in severe class imbalance.

    Zamir et al., "Asymmetric Loss For Multi-Label Classification" (ICCV 2021).
    """
    prob = torch.sigmoid(logits)

    # Positive branch: standard focal weighting
    loss_pos = (1 - prob).pow(ASL_GAMMA_POS) * targets \
               * torch.log(prob.clamp(min=1e-7))

    # Negative branch: shifted probability + asymmetric focus
    prob_neg = (prob - ASL_CLIP).clamp(min=0)   # shift negatives down
    loss_neg = prob_neg.pow(ASL_GAMMA_NEG) * (1 - targets) \
               * torch.log((1 - prob_neg).clamp(min=1e-7))

    return -(loss_pos + loss_neg).mean()


def supervised_loss(logits: torch.Tensor, targets: torch.Tensor,
                    pos_weight: torch.Tensor | None) -> torch.Tensor:
    if LOSS_NAME == "asl":
        return asymmetric_loss(logits, targets)
    if LOSS_NAME == "bce":
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight
        )
    if LOSS_NAME == "bce_unweighted":
        return F.binary_cross_entropy_with_logits(logits, targets)
    raise ValueError(f"Unknown loss: {LOSS_NAME}")


# ── Dataset ───────────────────────────────────────────────────────────────────

class GODataset(Dataset):
    def __init__(self, seqs: list[str], targets: torch.Tensor,
                 filter_empty: bool = True):
        if filter_empty:
            keep = (targets.sum(1) > 0).nonzero(as_tuple=True)[0]
            seqs    = [seqs[i] for i in keep.tolist()]
            targets = targets[keep]
        self.seqs    = seqs
        self.targets = targets.float()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], self.targets[i]


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool,
                pin_memory: bool = False, sampler=None) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle if sampler is None else False,
                      sampler=sampler,
                      num_workers=NUM_WORKERS, pin_memory=pin_memory,
                      collate_fn=lambda b: (
                          [x[0] for x in b],
                          torch.stack([x[1] for x in b])
                      ))


# ── Model ─────────────────────────────────────────────────────────────────────

class GOClassifier(nn.Module):
    """ESM-2 backbone + linear classification head for 489 GO-MF terms."""

    def __init__(self, backbone: EsmModel, esm_dim: int = 640):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(esm_dim, N_GO)
        nn.init.normal_(self.head.weight, std=esm_dim**-0.5)
        nn.init.zeros_(self.head.bias)

    def encode(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids,
                            attention_mask=attention_mask)
        mask = attention_mask.float().unsqueeze(-1)   # [B, L, 1]
        emb  = (out.last_hidden_state * mask).sum(1) \
               / mask.sum(1).clamp(min=1)              # [B, esm_dim]
        return emb

    def forward(self, input_ids, attention_mask):
        return self.head(self.encode(input_ids, attention_mask))  # [B, 489]


def freeze_unused_esm_parameters(backbone: EsmModel) -> int:
    """Disable gradients for ESM heads/embeddings that this classifier never uses."""
    frozen = 0
    unused_prefixes = ("contact_head.", "pooler.")
    unused_exact = {"embeddings.position_embeddings.weight"}
    for name, param in backbone.named_parameters():
        if name.startswith(unused_prefixes) or name in unused_exact:
            param.requires_grad_(False)
            frozen += param.numel()
    return frozen


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model: GOClassifier, loader: DataLoader,
            tokenizer: EsmTokenizer, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_targets = [], []
    use_amp = USE_AMP and device.type == "cuda"
    for seqs, targets in loader:
        enc = tokenizer(seqs, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LEN)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(
                input_ids      = enc["input_ids"].to(device),
                attention_mask = enc["attention_mask"].to(device),
            )
        all_logits.append(logits.cpu())
        all_targets.append(targets)
    return (torch.cat(all_logits).numpy(),
            torch.cat(all_targets).numpy())


def fmax_aupr(logits_np: np.ndarray, tgt_np: np.ndarray):
    """Protein-centric Fmax and AUPR over 199 quantile thresholds."""
    qs = np.linspace(1, 99, 199)
    thresholds = np.percentile(logits_np, qs)
    best_f = 0.0
    precs, recs = [], []
    for t in thresholds:
        preds = (logits_np > t).astype(np.float32)
        tp    = (preds * tgt_np).sum(1)
        prec  = float((tp / np.maximum(preds.sum(1), 1)).mean())
        rec   = float((tp / np.maximum(tgt_np.sum(1), 1)).mean())
        precs.append(prec)
        recs.append(rec)
        best_f = max(best_f, 2 * prec * rec / (prec + rec + 1e-8))
    order = np.argsort(recs)
    aupr  = float(np.trapz(np.array(precs)[order], np.array(recs)[order]))
    return best_f, aupr


def average_precision_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision for one GO term, matching term-centric AUPR usage."""
    y_true = y_true.astype(np.float32)
    n_pos = float(y_true.sum())
    if n_pos <= 0:
        return float("nan")
    order = np.argsort(-scores)
    hits = y_true[order]
    hit_rank = np.flatnonzero(hits > 0.5)
    if len(hit_rank) == 0:
        return 0.0
    cum_hits = np.cumsum(hits)[hit_rank]
    precision_at_hits = cum_hits / (hit_rank + 1)
    return float(precision_at_hits.sum() / n_pos)


def macro_aupr(logits_np: np.ndarray, tgt_np: np.ndarray) -> float:
    """Term-centric macro-AUPR: AP per GO term, averaged over non-empty terms."""
    aps = [
        average_precision_binary(tgt_np[:, j], logits_np[:, j])
        for j in range(tgt_np.shape[1])
        if tgt_np[:, j].sum() > 0
    ]
    return float(np.mean(aps)) if aps else 0.0


def retrieval_metrics(logits_np, tgt_np, ks=(1, 5, 10)):
    """Protein→GO recall@k and precision@k."""
    order = np.argsort(-logits_np, axis=1)
    out = {}
    for k in ks:
        topk = order[:, :k]
        tp_k = np.array([tgt_np[i, topk[i]].sum() for i in range(len(tgt_np))])
        n_pos = tgt_np.sum(1)
        out[f"R@{k}"] = float((tp_k / np.maximum(n_pos, 1)).mean())
        out[f"P@{k}"] = float((tp_k / k).mean())
    return out


# ── Training ──────────────────────────────────────────────────────────────────

def train(model: GOClassifier, tokenizer: EsmTokenizer,
          train_loader: DataLoader, val_loader: DataLoader,
          pos_weight: torch.Tensor, device, label: str, args: argparse.Namespace,
          train_sampler: DistributedSampler | None = None,
          distributed: bool = False):

    # Separate learning rates for backbone and classification head
    base_model = unwrap_model(model)
    optimizer = torch.optim.AdamW([
        {"params": base_model.backbone.parameters(), "lr": BACKBONE_LR},
        {"params": base_model.head.parameters(),     "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    # n_steps counts optimizer steps (after gradient accumulation)
    n_opt_steps = math.ceil(len(train_loader) / GRAD_ACCUM) * N_EPOCHS
    n_warmup    = int(WARMUP_FRAC * n_opt_steps)
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=n_warmup, num_training_steps=n_opt_steps
    )

    use_amp = USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    pos_weight = pos_weight.to(device) if LOSS_NAME == "bce" else None

    best_val_fmax, best_state = 0.0, None
    t0 = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss  = 0.0
        optimizer.zero_grad()

        for step, (seqs, targets) in enumerate(train_loader):
            targets = targets.to(device)

            enc = tokenizer(seqs, return_tensors="pt", padding=True,
                            truncation=True, max_length=MAX_LEN)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(
                    input_ids      = enc["input_ids"].to(device),
                    attention_mask = enc["attention_mask"].to(device),
                )                               # [B, 489]

                loss = supervised_loss(logits, targets, pos_weight) / GRAD_ACCUM
            scaler.scale(loss).backward()
            epoch_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        # Validation Fmax every 5 epochs
        if epoch % 5 == 0 or epoch == N_EPOCHS:
            if distributed:
                dist.barrier()
            if is_rank0():
                val_logits, val_tgt = predict(unwrap_model(model), val_loader,
                                              tokenizer, device)
                val_fmax, val_aupr  = fmax_aupr(val_logits, val_tgt)
                val_macro_aupr = macro_aupr(val_logits, val_tgt)

                if val_fmax > best_val_fmax:
                    best_val_fmax = val_fmax
                    best_state = cpu_state_dict(model)
                    best_path = save_training_checkpoint(
                        model, label, args, epoch, val_fmax, val_aupr,
                        val_macro_aupr, best_val_fmax, "best"
                    )
                    log(f"    saved best checkpoint: {best_path}")

                if args.save_every > 0 and (
                    epoch % args.save_every == 0 or epoch == N_EPOCHS
                ):
                    latest_path = save_training_checkpoint(
                        model, label, args, epoch, val_fmax, val_aupr,
                        val_macro_aupr, best_val_fmax, "latest"
                    )
                    log(f"    saved latest checkpoint: {latest_path}")

                elapsed = time.time() - t0
                log(f"  [{label}] ep {epoch:3d}/{N_EPOCHS}  "
                    f"loss={epoch_loss/len(train_loader):.4f}  "
                    f"val_Fmax={val_fmax:.4f}  val_AUPR={val_aupr:.4f}  "
                    f"val_macro_AUPR={val_macro_aupr:.4f}  "
                    f"({elapsed:.0f}s)")
            if distributed:
                dist.barrier()

    if is_rank0() and best_state is None:
        best_state = cpu_state_dict(model)
    return best_state, best_val_fmax


# ── CLI ───────────────────────────────────────────────────────────────────────

def default_label(model_name: str, loss_name: str) -> str:
    tag = model_name.split("/")[-1]
    tag = tag.replace("esm2_", "ESM2-").replace("_UR50D", "").replace("_", "-")
    return f"{tag}-FT-{loss_name}"


def slugify(label: str) -> str:
    keep = []
    for ch in label:
        if ch.isalnum():
            keep.append(ch)
        elif ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, (nn.DataParallel, DDP)):
        return model.module
    return model


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    model = unwrap_model(model)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def atomic_torch_save(payload: dict, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def save_training_checkpoint(model: nn.Module, label: str, args: argparse.Namespace,
                             epoch: int, val_fmax: float, val_aupr: float,
                             val_macro_aupr: float,
                             best_val_fmax: float, kind: str) -> Path:
    path = CKPT_DIR / f"{kind}_{slugify(label)}.pt"
    atomic_torch_save({
        "state_dict": cpu_state_dict(model),
        "esm_model": ESM_MODEL,
        "label": label,
        "loss": LOSS_NAME,
        "epoch": epoch,
        "val_fmax": float(val_fmax),
        "val_aupr": float(val_aupr),
        "val_macro_aupr": float(val_macro_aupr),
        "best_val_fmax": float(best_val_fmax),
        "args": vars(args),
    }, path)
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune ESM-2 as a supervised GO-MF baseline."
    )
    parser.add_argument("--esm-model", default=ESM_MODEL)
    parser.add_argument("--label", default=None)
    parser.add_argument("--loss", choices=["asl", "bce", "bce_unweighted"],
                        default=LOSS_NAME)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--backbone-lr", type=float, default=BACKBONE_LR)
    parser.add_argument("--head-lr", type=float, default=HEAD_LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--warmup-frac", type=float, default=WARMUP_FRAC)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--pos-weight-cap", type=float, default=POS_WEIGHT_CAP)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction,
                        default=USE_AMP)
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save latest checkpoint every N epochs; 0 disables periodic saves.")
    parser.add_argument("--data-parallel", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use all visible CUDA devices with torch.nn.DataParallel.")
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    distributed, rank, local_rank, world_size = setup_distributed()
    args = parse_args()
    global ESM_MODEL, N_EPOCHS, BATCH_SIZE, GRAD_ACCUM, BACKBONE_LR, HEAD_LR
    global WEIGHT_DECAY, MAX_LEN, WARMUP_FRAC, GRAD_CLIP, POS_WEIGHT_CAP
    global LOSS_NAME, USE_AMP, NUM_WORKERS

    ESM_MODEL = args.esm_model
    N_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    GRAD_ACCUM = args.grad_accum
    BACKBONE_LR = args.backbone_lr
    HEAD_LR = args.head_lr
    WEIGHT_DECAY = args.weight_decay
    MAX_LEN = args.max_len
    WARMUP_FRAC = args.warmup_frac
    GRAD_CLIP = args.grad_clip
    POS_WEIGHT_CAP = args.pos_weight_cap
    LOSS_NAME = args.loss
    USE_AMP = args.amp
    NUM_WORKERS = args.num_workers

    label = args.label or default_label(ESM_MODEL, LOSS_NAME)
    device = torch.device(f"cuda:{local_rank}" if distributed else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    log(f"Device: {device}  |  ESM-2: {ESM_MODEL}")
    if distributed:
        log(f"DDP: world_size={world_size}")
    if device.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(local_rank if distributed else 0)}  "
            f"VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    log()

    # ── Load data ─────────────────────────────────────────────────────────────
    log("Loading data ...")
    splits = torch.load(CACHE_DIR / "protst_go_mf_decoded.pt",
                        map_location="cpu", weights_only=False)

    pin_memory = device.type == "cuda"
    train_ds = GODataset(splits["train"]["seqs"], splits["train"]["targets"],
                         filter_empty=True)
    val_ds = GODataset(splits["validation"]["seqs"],
                       splits["validation"]["targets"], filter_empty=False)
    test_ds = GODataset(splits["test"]["seqs"], splits["test"]["targets"],
                        filter_empty=False)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    train_loader = make_loader(train_ds, BATCH_SIZE, shuffle=True,
                               pin_memory=pin_memory, sampler=train_sampler)
    val_loader = make_loader(val_ds, BATCH_SIZE, shuffle=False,
                             pin_memory=pin_memory)
    test_loader = make_loader(test_ds, BATCH_SIZE, shuffle=False,
                              pin_memory=pin_memory)

    # Positive class weights from training labels (for reference / fallback)
    tgt_train = splits["train"]["targets"].float()
    pos_freq  = (tgt_train > 0.5).float().mean(0)              # [489]
    pos_weight = ((1 - pos_freq) / pos_freq.clamp(min=1e-6)).clamp(max=POS_WEIGHT_CAP)

    n_train = sum(1 for s in splits["train"]["seqs"]
                  if splits["train"]["targets"][
                      splits["train"]["seqs"].index(s)].sum() > 0) \
              if False else len(train_loader.dataset)
    log(f"  train={len(train_loader.dataset)}  "
        f"val={len(val_loader.dataset)}  "
        f"test={len(test_loader.dataset)}")
    log(f"  GO terms: {N_GO}  "
        f"avg pos/protein (train): "
        f"{(tgt_train > 0.5).float().sum(1).mean():.1f}\n")

    # ── Load ESM-2 ─────────────────────────────────────────────────────────────
    log(f"Loading {ESM_MODEL} ...")
    tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL)
    backbone  = EsmModel.from_pretrained(ESM_MODEL)
    try:
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        backbone.gradient_checkpointing_enable()
    frozen_unused = freeze_unused_esm_parameters(backbone)
    esm_dim   = backbone.config.hidden_size    # 640 for 150M
    log(f"  ESM-2 hidden dim: {esm_dim}  "
        f"params: {sum(p.numel() for p in backbone.parameters())/1e6:.0f}M")
    log("  Gradient checkpointing: enabled")
    log(f"  Frozen unused ESM params: {frozen_unused/1e6:.2f}M")
    log(f"  Effective batch size: {BATCH_SIZE} × {GRAD_ACCUM} × {world_size} = "
        f"{BATCH_SIZE * GRAD_ACCUM * world_size}\n")

    # ── Single condition: strong Euclidean baseline ────────────────────────────
    log(f"── {label} ──────────────────────────────")
    if LOSS_NAME == "asl":
        log(f"  Loss: Asymmetric (γ+={ASL_GAMMA_POS}, γ-={ASL_GAMMA_NEG}, "
            f"clip={ASL_CLIP})")
    elif LOSS_NAME == "bce":
        log(f"  Loss: BCEWithLogits + capped pos_weight (cap={POS_WEIGHT_CAP})")
    else:
        log("  Loss: BCEWithLogits")
    log(f"  LR: backbone={BACKBONE_LR}  head={HEAD_LR}  "
        f"epochs={N_EPOCHS}  batch={BATCH_SIZE}  grad_accum={GRAD_ACCUM}  "
        f"amp={USE_AMP}\n")

    model = GOClassifier(backbone, esm_dim).to(device)
    if distributed:
        log("  DDP: enabled")
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)
    elif args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        n_gpu = torch.cuda.device_count()
        log(f"  DataParallel: enabled across {n_gpu} GPUs")
        model = nn.DataParallel(model)
    else:
        log("  DataParallel: disabled")

    best_state, best_val_fmax = train(
        model, tokenizer, train_loader, val_loader, pos_weight, device, label,
        args, train_sampler=train_sampler, distributed=distributed
    )

    if distributed and not is_rank0():
        return

    # ── Test evaluation ───────────────────────────────────────────────────────
    log(f"\n  Best val Fmax = {best_val_fmax:.4f}")
    log("  Running test evaluation ...")

    unwrap_model(model).load_state_dict(best_state)
    model.to(device)

    test_logits, test_tgt = predict(unwrap_model(model), test_loader, tokenizer, device)
    test_fmax, test_aupr  = fmax_aupr(test_logits, test_tgt)
    test_macro_aupr = macro_aupr(test_logits, test_tgt)
    ret = retrieval_metrics(test_logits, test_tgt)

    # IC for wFmax
    tgt_np   = test_tgt
    ic_np    = -np.log((pos_freq.numpy() + 1e-6))
    qs       = np.linspace(1, 99, 199)
    tholds   = np.percentile(test_logits, qs)
    best_wf  = 0.0
    for t in tholds:
        preds  = (test_logits > t).astype(np.float32)
        tp_w   = (preds * tgt_np * ic_np).sum(1)
        pred_w = (preds * ic_np).sum(1)
        true_w = (tgt_np * ic_np).sum(1)
        wp = float((tp_w / np.maximum(pred_w, 1e-8)).mean())
        wr = float((tp_w / np.maximum(true_w, 1e-8)).mean())
        best_wf = max(best_wf, 2 * wp * wr / (wp + wr + 1e-8))

    ndcg_scores, ap_scores = [], []
    full_order = np.argsort(-test_logits, axis=1)
    for i in range(len(test_logits)):
        true_idx = set(np.where(tgt_np[i] > 0.5)[0])
        if not true_idx: continue
        o = full_order[i]
        dcg  = sum(1/math.log2(r+2) for r, j in enumerate(o[:10]) if j in true_idx)
        idcg = sum(1/math.log2(r+2) for r in range(min(len(true_idx), 10)))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
        hits, ap = 0, 0.0
        for r, j in enumerate(o):
            if j in true_idx:
                hits += 1; ap += hits / (r + 1)
        ap_scores.append(ap / len(true_idx))

    results = {
        "fmax":   round(test_fmax, 4),
        "aupr":   round(test_aupr, 4),
        "macro_aupr": round(test_macro_aupr, 4),
        "wfmax":  round(best_wf,   4),
        "ndcg10": round(float(np.mean(ndcg_scores)), 4),
        "map":    round(float(np.mean(ap_scores)),   4),
        **{k: round(v, 4) for k, v in ret.items()},
    }

    # ── Print table ──────────────────────────────────────────────────────────
    sep = "=" * 78
    print(f"\n{sep}")
    print(f"  {'Model':<22}  Fmax    AUPR  mAUPR   wFmax  nDCG@10    MAP")
    print("-" * 78)
    print(f"  {'ESM2-8M frozen (v1)':<22}  0.0748  0.0318    n/a  0.0614   0.0881  0.0877"
          "  ← previous")
    print(f"  {label:<22}  "
          f"{results['fmax']:.4f}  {results['aupr']:.4f}  "
          f"{results['macro_aupr']:.4f}  "
          f"{results['wfmax']:.4f}   {results['ndcg10']:.4f}  "
          f"{results['map']:.4f}  ← fine-tuned")
    print(sep)
    print(f"\n  Protein→GO retrieval:")
    for k in [1, 5, 10]:
        print(f"    R@{k}={results[f'R@{k}']:.4f}  P@{k}={results[f'P@{k}']:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    ckpt_path = CKPT_DIR / f"probe_{slugify(label)}.pt"
    torch.save({
        "state_dict": best_state,
        "esm_model":  ESM_MODEL,
        "label":      label,
        "loss":       LOSS_NAME,
        "args":       vars(args),
        "results":    results,
    }, ckpt_path)
    print(f"\n  Checkpoint saved: {ckpt_path}")

    out_path = RESULTS_DIR / f"results_{slugify(label)}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved:    {out_path}")


if __name__ == "__main__":
    main()
