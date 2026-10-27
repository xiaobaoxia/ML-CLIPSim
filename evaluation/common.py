"""Shared checkpoint loading, real compression and result files."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--quality', type=int, default=1)
    parser.add_argument('--weights-root', type=Path, default=Path('weights/codec'))
    parser.add_argument('--checkpoint', type=Path, help='optional compatible MLIC++ checkpoint')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cdf', choices=['update', 'saved'], default='update')
    parser.add_argument('--limit', type=int, help='evaluate only the first N samples')
    parser.add_argument('--output', type=Path, required=True)
    return parser


def load_codec(args):
    from codec import load_mlicpp_mlclipsim
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    model = load_mlicpp_mlclipsim(1 if args.checkpoint else args.quality, device=args.device,
                                 weights_root=args.weights_root, checkpoint=args.checkpoint)
    if args.cdf == 'update':
        model.update(force=True)
    return model


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


@torch.inference_mode()
def reconstruct(model, image, device, multiple=128, mode='constant', numpy_division=False):
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert('RGB')).copy()
        if numpy_division:
            tensor = torch.from_numpy(array.astype(np.float32) / 255).permute(2, 0, 1)
        else:
            tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255)
    else:
        tensor = image
    x = tensor.unsqueeze(0).to(device)
    height, width = x.shape[-2:]
    dh, dw = (-height) % multiple, (-width) % multiple
    top, left = dh // 2, dw // 2
    padded = F.pad(x, (left, dw - left, top, dh - top), mode=mode)
    encoded = model.compress(padded)
    decoded = model.decompress(encoded['strings'], encoded['shape'])['x_hat']
    decoded = decoded[..., top:top + height, left:left + width]
    decoded = torch.nan_to_num(decoded, nan=0, posinf=1, neginf=0).clamp(0, 1)
    nbytes = sum(len(value) for group in encoded['strings'] for value in group)
    return decoded[0], dict(bytes=nbytes, bpp=8 * nbytes / (height * width),
                           width=width, height=height)


def as_pil(tensor):
    array = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


class Results:
    """Append per-sample results and resume only an identical run configuration."""
    def __init__(self, args):
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=True)
        checkpoint = args.checkpoint or args.weights_root / f'mlicpp_ml_clipsim_q{args.quality}.pt'
        config = json.loads(json.dumps(vars(args), default=str))
        config.update(checkpoint_sha256=file_hash(checkpoint), torch=torch.__version__)
        path = self.output / 'config.json'
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError('Output contains a different run; choose a new --output directory')
        path.write_text(json.dumps(config, indent=2) + '\n')
        self.path = self.output / 'records.jsonl'
        self.rows = [json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []
        self.done = {str(row['id']) for row in self.rows}
        if len(self.done) != len(self.rows):
            raise ValueError('Duplicate result IDs')

    def add(self, row):
        if str(row['id']) in self.done:
            raise ValueError('Duplicate result ID')
        with self.path.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        self.rows.append(row)
        self.done.add(str(row['id']))

    def save(self, **metrics):
        if not self.rows:
            raise ValueError('No samples were evaluated')
        weight = sum(row.get('weight', 1) for row in self.rows)
        summary = dict(count=len(self.rows), mean_bpp=sum(row['bpp'] * row.get('weight', 1) for row in self.rows) / weight, **metrics)
        (self.output / 'metrics.json').write_text(json.dumps(summary, indent=2, default=str) + '\n')
        print(json.dumps(summary, default=str))
        return summary
