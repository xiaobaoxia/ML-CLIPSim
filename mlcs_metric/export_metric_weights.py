#!/usr/bin/env python3
"""Export lightweight ML-CLIPSim weights from a full training checkpoint."""

import argparse
import json
from pathlib import Path

try:
    from .checkpoint_utils import export_learnable_weights, LEARNABLE_KEYS
except ImportError:  # Preserve direct script execution.
    from checkpoint_utils import export_learnable_weights, LEARNABLE_KEYS


def main():
    ap = argparse.ArgumentParser(
        description="Export weights/ML-CLIPSim_ViTB16.pt (~2 KB) from a full .pth checkpoint"
    )
    ap.add_argument(
        "--src",
        default="runs/metric/best.pth",
        help="Full training checkpoint",
    )
    ap.add_argument(
        "--dst",
        default="weights/ML-CLIPSim_ViTB16.pt",
        help="Output path for the lightweight aggregation checkpoint",
    )
    ap.add_argument("--write_meta_json", action="store_true", default=True)
    args = ap.parse_args()

    dst = export_learnable_weights(args.src, args.dst)
    print(f"[OK] Wrote {dst} ({dst.stat().st_size} bytes)")

    if args.write_meta_json:
        import torch

        ckpt = torch.load(dst, map_location="cpu", weights_only=False)
        meta_path = dst.with_suffix(".meta.json")
        meta = {
            "model": "ML-CLIPSim",
            "checkpoint_file": dst.name,
            "source_checkpoint": str(Path(args.src).resolve()),
            "learnable_keys": list(LEARNABLE_KEYS),
            "validation_hard_acc": ckpt.get("best_acc"),
            "validation_metrics": ckpt.get("validation"),
            "learnable_parameters": {
                k: v.flatten().tolist() for k, v in ckpt["state_dict"].items()
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[OK] Wrote {meta_path}")


if __name__ == "__main__":
    main()
