"""Verify released codec weights without downloading them.

Example: python -m codec.validate --weights-root weights/codec --report codec-report.json
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from .checkpoint_utils import RATE_POINTS, load_mlicpp_mlclipsim


def verify_codec(quality, *, weights_root, device="cpu", size=256):
    start = time.monotonic()
    model = load_mlicpp_mlclipsim(quality, device=device, weights_root=weights_root)
    torch.manual_seed(42)
    image = torch.rand(1, 3, size, size, device=device)
    with torch.no_grad():
        expected = model(image)["x_hat"]
        model.update(force=True)
        encoded = model.compress(image)
        actual = model.decompress(encoded["strings"], encoded["shape"])["x_hat"]
    error = float((expected - actual).abs().max())
    if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
        raise AssertionError(f"q{quality}: reconstruction contains non-finite values")
    if not torch.allclose(expected, actual, atol=1e-5, rtol=1e-5):
        raise AssertionError(f"q{quality}: bitstream reconstruction error {error}")
    report = {"rate_point": quality, "parameters": sum(p.numel() for p in model.parameters()),
              "input_shape": list(image.shape), "finite": True,
              "roundtrip_max_abs_error": error,
              "encoded_bytes": sum(len(s) for group in encoded["strings"] for s in group),
              "seconds": time.monotonic() - start}
    del model
    gc.collect()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", type=Path, default=Path("weights/codec"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.size < 64 or args.size % 64:
        parser.error("--size must be a multiple of 64, at least 64")
    torch.set_num_threads(args.threads)
    results = []
    for quality in RATE_POINTS:
        result = verify_codec(quality, weights_root=args.weights_root,
                              device=args.device, size=args.size)
        results.append(result)
        print(json.dumps(result), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
