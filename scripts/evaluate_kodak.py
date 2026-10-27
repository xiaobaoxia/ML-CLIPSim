#!/usr/bin/env python3
"""Evaluate released weights on original Kodak images with real entropy coding.

Run from an installed source checkout. No model or dataset downloads occur.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time

import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision.transforms.functional import to_tensor

from codec import load_mlicpp_mlclipsim


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--weights-root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--qualities', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--cdf-policy', choices=['update', 'saved'], default='update',
                        help='update entropy tables, or preserve tables stored in the checkpoint')
    parser.add_argument('--device', default='cpu', help='cpu or an explicit CUDA device, e.g. cuda:0')
    parser.add_argument('--disable-cudnn', action='store_true',
                        help='match the original Kodak evaluation backend setting')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--limit', type=int, help='optional image limit for a timing check')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    images = sorted(p for p in args.data.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    if args.limit:
        images = images[:args.limit]
    if not images:
        parser.error('no images found')
    report = {'device': str(device), 'torch': torch.__version__, 'threads': args.threads,
              'cdf_policy': args.cdf_policy,
              'cudnn_enabled': torch.backends.cudnn.enabled,
              'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
              'protocol': 'RGB float32; symmetric zero padding to 128; real compress/decompress; crop+clamp [0,1]; arithmetic per-image means',
              'images': [{'filename': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in images], 'results': []}
    for q in args.qualities:
        model = load_mlicpp_mlclipsim(q, device=device, weights_root=args.weights_root)
        if args.cdf_policy == 'update':
            model.update(force=True)
        records = []
        for path in images:
            start = time.monotonic()
            with Image.open(path) as im:
                x = to_tensor(im.convert('RGB')).unsqueeze(0).to(device)
            h, w = x.shape[-2:]
            dh, dw = (-h) % 128, (-w) % 128
            left, top = dw // 2, dh // 2
            padded = F.pad(x, (left, dw-left, top, dh-top))
            with torch.inference_mode():
                encoded = model.compress(padded)
                decoded = model.decompress(encoded['strings'], encoded['shape'])['x_hat']
                decoded = decoded[..., top:top+h, left:left+w].clamp(0, 1)
                nbytes = sum(len(s) for group in encoded['strings'] for s in group)
                mse = float((decoded-x).square().mean())
                record = {'filename': path.name, 'height': h, 'width': w,
                          'bytes': nbytes, 'bpp': nbytes*8/(h*w),
                          'psnr': -10*math.log10(mse) if mse else 99.99,
                          'ms_ssim': float(ms_ssim(x, decoded, data_range=1.0)),
                          'seconds': time.monotonic()-start}
            records.append(record)
            print(json.dumps({'quality': q, **record}), flush=True)
        summary = {k: sum(r[k] for r in records)/len(records)
                   for k in ['bpp', 'psnr', 'ms_ssim', 'seconds']}
        report['results'].append({'quality': q, 'checkpoint_id':getattr(model,'paper_checkpoint_id',None), 'count': len(records),
                                  'total_bytes':sum(record['bytes'] for record in records),
                                  'mean': summary, 'per_image': records})
        write_report(args.report, report)
        print(json.dumps({'quality': q, 'mean': summary}), flush=True)
        del model


if __name__ == '__main__':
    main()
