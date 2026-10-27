"""Train MLIC++ with MSE or the frozen ML-CLIPSim metric.

Run ``python scripts/train_mlic.py --help`` for the supported, portable training CLI.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from MLIC_pp.config.config import model_config
from MLIC_pp.models import MLICPlusPlus
from codec.checkpoint_utils import _load_checkpoint_file


class ImageDataset(Dataset):
    """Read RGB images recursively, including class-subdirectory ImageNet layouts."""

    def __init__(self, root, patch_size=256, training=True):
        self.files = sorted(
            p for p in Path(root).rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        )
        if not self.files:
            raise ValueError(f"no images found under {root}")
        self.patch_size = patch_size
        self.transform = transforms.Compose([
            transforms.RandomCrop(patch_size, pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]) if training else transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            return self.transform(image.convert("RGB"))


def pad_image(images):
    """Replicate-pad a BCHW image batch to multiples of 64."""
    h, w = images.shape[-2:]
    return F.pad(images, (0, (-w) % 64, 0, (-h) % 64), mode="replicate")


def rate_distortion(output, target, objective, lmbda, metric=None, bpp_weight=1.0):
    """Return the loss, estimated bits/pixel, and unweighted distortion."""
    num_pixels = target.shape[0] * target.shape[-2] * target.shape[-1]
    bpp = sum(-values.clamp_min(1e-9).log2().sum() / num_pixels
              for values in output["likelihoods"].values())
    reconstruction = output["x_hat"][..., :target.shape[-2], :target.shape[-1]]
    if objective == "mse":
        distortion = 255 ** 2 * F.mse_loss(reconstruction, target)
    elif objective == "mlclipsim":
        if metric is None:
            raise ValueError("ML-CLIPSim objective requires a metric")
        distortion = 1 - metric(target, reconstruction).mean()
    else:
        raise ValueError(f"unknown objective: {objective}")
    return bpp_weight * bpp + lmbda * distortion, bpp, distortion


def configure_optimizers(model, learning_rate, aux_learning_rate):
    main, auxiliary = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (auxiliary if name.endswith(".quantiles") else main).append(parameter)
    return (torch.optim.Adam(main, lr=learning_rate),
            torch.optim.Adam(auxiliary, lr=aux_learning_rate))


def train_step(model, images, optimizer, aux_optimizer, *, objective, lmbda,
               metric=None, bpp_weight=1.0, clip_max_norm=1.0):
    """One complete main/entropy-auxiliary update, with input gradients through MLCS."""
    optimizer.zero_grad(set_to_none=True)
    aux_optimizer.zero_grad(set_to_none=True)
    output = model(pad_image(images))
    loss, bpp, distortion = rate_distortion(
        output, images, objective, lmbda, metric, bpp_weight)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite rate-distortion loss")
    loss.backward()
    if clip_max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
    optimizer.step()
    # The entropy auxiliary loss learns quantiles independently of the RD update.
    aux_optimizer.zero_grad(set_to_none=True)
    aux_loss = model.aux_loss()
    if not torch.isfinite(aux_loss):
        raise FloatingPointError("non-finite entropy auxiliary loss")
    aux_loss.backward()
    aux_optimizer.step()
    return {"loss": float(loss.detach()), "bpp": float(bpp.detach()),
            "distortion": float(distortion.detach()), "aux_loss": float(aux_loss.detach())}


@torch.no_grad()
def evaluate(model, loader, device, *, objective, lmbda, metric=None, bpp_weight=1.0,
             max_steps=None):
    model.eval()
    totals = {"loss": 0.0, "bpp": 0.0, "distortion": 0.0}
    count = 0
    for step, images in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        images = images.to(device)
        values = rate_distortion(model(pad_image(images)), images, objective,
                                 lmbda, metric, bpp_weight)
        for key, value in zip(totals, values):
            totals[key] += float(value) * images.shape[0]
        count += images.shape[0]
    if not count:
        raise ValueError("validation loader produced no images")
    return {key: value / count for key, value in totals.items()}


def save_checkpoint(path, *, model, optimizer, aux_optimizer, scheduler,
                    epoch, best_loss, args):
    """Atomically save the complete training state (epoch = completed epochs)."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"lic_state": model.state_dict(), "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch,
                "best_loss": best_loss, "cfg": vars(args),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "python_rng_state": random.getstate()}, temporary)
    temporary.replace(path)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument("--val-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--objective", choices=("mse", "mlclipsim"), default="mlclipsim")
    parser.add_argument("--lambda", dest="lmbda", type=float, default=None,
                        help="RD distortion weight; defaults to 2 for MLCS, .01 for MSE")
    parser.add_argument("--bpp-weight", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=Path, help="initial codec weights: lic_state/state_dict/raw")
    parser.add_argument("--resume", type=Path, help="resume complete state from train_mlic.py checkpoint")
    parser.add_argument("--metric-checkpoint", type=Path, help="optional MLCS aggregation checkpoint")
    parser.add_argument("--clip-path", type=Path, help="local OpenAI CLIP backbone for offline use")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--aux-lr", type=float, default=1e-3)
    parser.add_argument("--clip-max-norm", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, help="limit training batches per epoch for smoke tests")
    parser.add_argument("--max-val-steps", type=int, help="limit validation batches for smoke tests")
    parser.add_argument("--threads", type=int, help="optional CPU intra-op thread count")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.checkpoint and args.resume:
        parser.error("use --checkpoint for initialization or --resume for continuation")
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs/batch-size must be positive and workers nonnegative")
    if args.patch_size < 64 or args.patch_size % 64:
        parser.error("patch-size must be a multiple of 64, at least 64")
    if any(value is not None and value < 1 for value in (args.max_steps, args.max_val_steps)):
        parser.error("step limits must be positive")
    if args.objective == "mlclipsim" and not (args.checkpoint or args.resume):
        parser.error("MLCS fine-tuning requires --checkpoint with an MSE-trained codec (or --resume)")
    if args.lmbda is None:
        args.lmbda = 0.01 if args.objective == "mse" else 2.0
    if args.lmbda < 0 or args.bpp_weight < 0:
        parser.error("loss weights must be nonnegative")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    train_loader = DataLoader(ImageDataset(args.train_dir, args.patch_size),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(ImageDataset(args.val_dir, training=False), batch_size=1,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    model = MLICPlusPlus(model_config()).to(device)
    payload = None
    if args.checkpoint or args.resume:
        payload = _load_checkpoint_file(args.checkpoint or args.resume)
        state = payload.get("lic_state", payload.get("state_dict", payload))
        try:
            model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        except RuntimeError as exc:
            raise RuntimeError(
                "Initialization checkpoint uses an incompatible MLIC++ synthesis/LRP variant. "
                "Use matching MSE weights or the released ML-CLIPSim codecs. "
                f"Original error: {exc}"
            ) from exc
    optimizer, aux_optimizer = configure_optimizers(model, args.lr, args.aux_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * min(len(train_loader), args.max_steps or len(train_loader)))
    start, best = 0, math.inf
    if args.resume:
        saved_config = payload.get("cfg", {})
        for key in ("objective", "lmbda", "bpp_weight"):
            if saved_config.get(key) != getattr(args, key):
                parser.error(f"resume requires the saved {key}={saved_config.get(key)!r}")
        optimizer.load_state_dict(payload["optimizer"])
        aux_optimizer.load_state_dict(payload["aux_optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        if args.epochs != saved_config.get("epochs"):
            # If explicitly extending a run, continue along the new cosine horizon.
            scheduler.T_max = args.epochs * min(len(train_loader), args.max_steps or len(train_loader))
            for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                group["lr"] = scheduler.eta_min + (base_lr - scheduler.eta_min) * (
                    1 + math.cos(math.pi * scheduler.last_epoch / scheduler.T_max)) / 2
        start, best = payload["epoch"], payload["best_loss"]
    metric = None
    if args.objective == "mlclipsim":
        from mlclipsim import load_metric
        options = {"device": device, "freeze": True}
        if args.clip_path is not None:
            options["clip_checkpoint"] = args.clip_path
        metric = load_metric(args.metric_checkpoint, **options)
    if args.resume:
        # Restore after constructing the metric: module initialization consumes RNG state.
        torch.set_rng_state(payload["torch_rng_state"])
        random.setstate(payload["python_rng_state"])
        if payload.get("cuda_rng_states") and device.type == "cuda":
            torch.cuda.set_rng_state_all(payload["cuda_rng_states"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Store readable run settings alongside resumable checkpoints.
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    # Path objects are converted before torch.save so weights_only=True can reload safely.
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, str(value))
    output_dir = Path(args.output_dir)
    for epoch in range(start, args.epochs):
        model.train()
        for step, images in enumerate(train_loader):
            if args.max_steps is not None and step >= args.max_steps:
                break
            stats = train_step(model, images.to(device), optimizer, aux_optimizer,
                               objective=args.objective, lmbda=args.lmbda, metric=metric,
                               bpp_weight=args.bpp_weight, clip_max_norm=args.clip_max_norm)
            scheduler.step()
            print(json.dumps({"epoch": epoch + 1, "step": step + 1, "train": stats}), flush=True)
        val = evaluate(model, val_loader, device, objective=args.objective, lmbda=args.lmbda,
                       metric=metric, bpp_weight=args.bpp_weight, max_steps=args.max_val_steps)
        improved = val["loss"] < best
        best = min(best, val["loss"])
        common = dict(model=model, optimizer=optimizer, aux_optimizer=aux_optimizer,
                      scheduler=scheduler, epoch=epoch + 1, best_loss=best, args=args)
        save_checkpoint(output_dir / "last.pt", **common)
        if improved:
            save_checkpoint(output_dir / "best.pt", **common)
        record = {"epoch": epoch + 1, "validation": val, "best_loss": best}
        with (output_dir / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
