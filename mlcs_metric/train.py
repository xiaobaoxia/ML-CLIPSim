#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train ML-CLIPSim on COCO 2000 pairwise soft labels."""

import os

import json
import random
import math
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .metric import CLIPMultiLayerMetric
    from .pair_dataset import PairMetricDataset
except ImportError:  # Direct script execution.
    from metric import CLIPMultiLayerMetric
    from pair_dataset import PairMetricDataset


# Utils: cached split
def split_labels_cached(
    label_json,
    out_dir,
    train_ratio=0.8,
    seed=42,
    force=False
):
    os.makedirs(out_dir, exist_ok=True)

    train_json = os.path.join(out_dir, "labels_train.json")
    val_json   = os.path.join(out_dir, "labels_val.json")
    meta_json  = os.path.join(out_dir, "split_meta.json")

    if (
        not force
        and os.path.isfile(train_json)
        and os.path.isfile(val_json)
        and os.path.isfile(meta_json)
    ):
        print(f"[Split] Reusing cached split from {out_dir}")
        return train_json, val_json

    with open(label_json, "r") as f:
        labels = json.load(f)

    random.seed(seed)
    random.shuffle(labels)

    n_train = int(len(labels) * train_ratio)
    train_labels = labels[:n_train]
    val_labels   = labels[n_train:]

    with open(train_json, "w") as f:
        json.dump(train_labels, f, indent=2)
    with open(val_json, "w") as f:
        json.dump(val_labels, f, indent=2)

    meta = {
        "source": os.path.abspath(label_json),
        "seed": seed,
        "train_ratio": train_ratio,
        "num_total": len(labels),
        "num_train": len(train_labels),
        "num_val": len(val_labels),
    }
    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Split] Created | train={len(train_labels)} | val={len(val_labels)}")
    return train_json, val_json


# Ranking loss (probability space)
def rank_loss_prob(s0, s1, target, temperature=20.0):
    """
    s0, s1: similarity scores (higher is better)
    target: soft label in [0,1]
    """
    logit = (s0 - s1) * temperature
    return F.binary_cross_entropy_with_logits(logit, target)


# Cosine decay + Warmup (step-level)
def get_cosine_warmup_scheduler(
    optimizer,
    warmup_steps,
    total_steps,
    min_lr_ratio=0.01,
    last_epoch=-1
):
    warmup_steps = int(warmup_steps)
    total_steps = int(total_steps)
    assert total_steps > 0
    assert total_steps > warmup_steps, f"total_steps({total_steps}) must be > warmup_steps({warmup_steps})"
    assert 0.0 <= min_lr_ratio <= 1.0

    def lr_lambda(step: int):
        # step is 0-based
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))

        # cosine progress after warmup in [0,1]
        denom = max(1, total_steps - warmup_steps)
        progress = float(step - warmup_steps) / float(denom)
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=last_epoch
    )


# Save / Load checkpoint
def save_checkpoint(path, model, optimizer, epoch, best_acc, metrics, args, scheduler=None):
    save_obj = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_acc": best_acc,
        "metrics": metrics,
        "args": vars(args),
    }
    if scheduler is not None:
        save_obj["scheduler"] = scheduler.state_dict()

    torch.save(save_obj, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    start_epoch = ckpt.get("epoch", -1) + 1
    best_acc = ckpt.get("best_acc", -1e9)

    print(f"[Resume] Loaded checkpoint from: {path}")
    print(f"[Resume] start_epoch={start_epoch}, best_acc={best_acc:.6f}")

    return start_epoch, best_acc, ckpt


# Evaluation
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    hard_correct = 0
    hard_total = 0
    soft_l1 = 0.0
    margin_abs = 0.0

    for ref, p0, p1, hard, soft in loader:
        ref = ref.to(device, non_blocking=True)
        p0 = p0.to(device, non_blocking=True)
        p1 = p1.to(device, non_blocking=True)
        soft = soft.to(device, non_blocking=True)

        s0 = model(ref, p0)
        s1 = model(ref, p1)

        diff = s0 - s1
        prob = torch.sigmoid(diff * 5.0)

        if hard is not None:
            hard = hard.to(device, non_blocking=True)
            pred = (diff > 0).float()
            hard_correct += (pred == hard).sum().item()
            hard_total += hard.numel()

        soft_l1 += (prob - soft).abs().mean().item()
        margin_abs += diff.abs().mean().item()

    return {
        "hard_acc": hard_correct / hard_total if hard_total > 0 else 0.0,
        "soft_L1": soft_l1 / len(loader),
        "mean_margin": margin_abs / len(loader),
    }


# Train
def train():
    parser = argparse.ArgumentParser()

    # data / split
    parser.add_argument("--label_json", type=str, default="dataset/pcmp/labels.json")
    parser.add_argument("--split_out_dir", type=str, default="dataset/pcmp/split")
    parser.add_argument("--dist_root", type=str, default="dataset/pcmp/images")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--force_split", action="store_true")

    # dataloader
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--neutral_ratio", type=float, default=0.0)

    # train
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clip-path", "--clip_path", dest="clip_path",
                        help="local OpenAI CLIP ViT-B/16 checkpoint for offline training")

    # scheduler
    parser.add_argument("--use_scheduler", action="store_true",
                        help="Enable cosine decay + warmup scheduler.")
    parser.add_argument("--warmup_ratio", type=float, default=0.01,
                        help="Warmup ratio of total steps. e.g. 0.01 means 1%%.")
    parser.add_argument("--min_lr_ratio", type=float, default=0.05,
                        help="Final lr = base_lr * min_lr_ratio.")

    # loss
    parser.add_argument("--temperature", type=float, default=20.0)

    # save
    parser.add_argument("--ckpt_dir", type=str, default="runs/metric")
    parser.add_argument("--ckpt_name", type=str, default="best.pth")
    parser.add_argument("--save_every_epoch", action="store_true",default=True,
                        help="Save one checkpoint for every epoch.")
    parser.add_argument("--save_last", action="store_true",default=True,
                        help="Always overwrite ckpt_dir/last.pth each epoch.")

    # resume
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint for resuming training.")

    args = parser.parse_args()

    # ---------- basic ----------
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"[Info] device={device}")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    last_ckpt_path = os.path.join(args.ckpt_dir, "last.pth")

    # ---------- split ----------
    train_json, val_json = split_labels_cached(
        label_json=args.label_json,
        out_dir=args.split_out_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        force=args.force_split
    )

    # ---------- datasets ----------
    train_set = PairMetricDataset(
        label_json=train_json,
        dist_root=args.dist_root,
        neutral_ratio=args.neutral_ratio,
        seed=args.seed
    )
    val_set = PairMetricDataset(
        label_json=val_json,
        dist_root=args.dist_root,
        neutral_ratio=args.neutral_ratio,
        seed=args.seed
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"[Data] train={len(train_set)} | val={len(val_set)}")
    print(f"[Data] train_loader steps/epoch={len(train_loader)}")

    # ---------- model ----------
    model = CLIPMultiLayerMetric(device=device, use_attn_weight=False,
                                clip_checkpoint=args.clip_path).to(device)

    # only optimize params with requires_grad=True
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    # ---------- scheduler ----------
    scheduler = None
    total_steps = args.epochs * len(train_loader)

    if args.use_scheduler:
        warmup_steps = int(args.warmup_ratio * total_steps)
        warmup_steps = min(warmup_steps, max(0, total_steps - 1))

        scheduler = get_cosine_warmup_scheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        print(
            f"[Sched] Cosine+Warmup enabled | "
            f"total_steps={total_steps} | "
            f"warmup_steps={warmup_steps} | "
            f"min_lr_ratio={args.min_lr_ratio}"
        )
    else:
        print("[Sched] disabled (constant LR)")

    # ---------- resume ----------
    start_epoch = 0
    best_acc = -1e9

    if args.resume:
        start_epoch, best_acc, _ = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )

    # ---------- training ----------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        running_loss = 0.0

        for ref, p0, p1, hard, soft in pbar:
            ref = ref.to(device, non_blocking=True)
            p0 = p0.to(device, non_blocking=True)
            p1 = p1.to(device, non_blocking=True)
            soft = soft.to(device, non_blocking=True)

            if hard is not None:
                hard = hard.to(device, non_blocking=True)

            s0 = model(ref, p0)
            s1 = model(ref, p1)

            loss = rank_loss_prob(s0, s1, soft, temperature=args.temperature)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item()
            avg_loss = running_loss / (pbar.n + 1)
            lr_now = optimizer.param_groups[0]["lr"]

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{avg_loss:.4f}",
                lr=f"{lr_now:.2e}"
            )

        # ---------- eval ----------
        metrics = evaluate(model, val_loader, device)

        print(
            f"[Epoch {epoch}] "
            f"HardAcc={metrics['hard_acc']:.4f} | "
            f"SoftL1={metrics['soft_L1']:.4f} | "
            f"MeanMargin={metrics['mean_margin']:.4f}"
        )

        # ---------- update best ----------
        mean_acc = metrics["hard_acc"]
        is_best = mean_acc > best_acc
        if is_best:
            best_acc = mean_acc

        # ---------- save every epoch ----------
        if args.save_every_epoch:
            epoch_ckpt_path = os.path.join(args.ckpt_dir, f"epoch_{epoch:03d}.pth")
            save_checkpoint(
                path=epoch_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_acc,
                metrics=metrics,
                args=args,
                scheduler=scheduler
            )
            print(f"[Save] Epoch checkpoint saved -> {epoch_ckpt_path}")

        # ---------- save last ----------
        if args.save_last:
            save_checkpoint(
                path=last_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_acc,
                metrics=metrics,
                args=args,
                scheduler=scheduler
            )
            print(f"[Save] Last checkpoint updated -> {last_ckpt_path}")

        # ---------- save best ----------
        if is_best:
            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_acc,
                metrics=metrics,
                args=args,
                scheduler=scheduler
            )
            print(f"[Save] New best model at epoch {epoch}, best_acc={best_acc:.4f} -> {best_ckpt_path}")


if __name__ == "__main__":
    train()
