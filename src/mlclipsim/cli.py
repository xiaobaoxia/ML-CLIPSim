"""Command-line image scoring."""

from __future__ import annotations

import argparse

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from .checkpoint import load_metric


def _load_rgb(path: str) -> torch.Tensor:
    with Image.open(path) as image:
        return to_tensor(image.convert("RGB")).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score an image with ML-CLIPSim")
    parser.add_argument("--ref", required=True, help="reference image")
    parser.add_argument("--dist", required=True, help="distorted or reconstructed image")
    parser.add_argument("--checkpoint", default=None, help="optional local metric checkpoint")
    parser.add_argument("--device", default=None, help="cpu, cuda, or automatic when omitted")
    parser.add_argument("--clip-checkpoint", help="local OpenAI CLIP ViT-B/16 backbone for offline use")
    parser.add_argument("--download-root", help="CLIP backbone cache directory")
    args = parser.parse_args()

    metric = load_metric(
        args.checkpoint, device=args.device, freeze=True,
        clip_checkpoint=args.clip_checkpoint, download_root=args.download_root,
    )
    device = next(metric.parameters()).device
    reference = _load_rgb(args.ref).to(device)
    candidate = _load_rgb(args.dist).to(device)
    score = metric.score(reference, candidate)
    print(f"ML-CLIPSim: {score.item():.6f}")


if __name__ == "__main__":
    main()
