#!/usr/bin/env python3
"""Verify the published PCMP annotations and optionally every saved image."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root, row):
    path = root / row['path']
    if not path.is_file() or path.stat().st_size != row['size_bytes']:
        raise ValueError(f'missing file or wrong size: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    if digest.hexdigest() != row['sha256']:
        raise ValueError(f'wrong SHA256: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('dataset/pcmp'))
    parser.add_argument('--full-images', action='store_true')
    args = parser.parse_args()
    rows = json.loads((args.dataset / 'annotations_manifest.json').read_text())['files']
    for row in rows:
        verify(args.dataset, row)
    count = 0
    if args.full_images:
        with (args.dataset / 'images_manifest.jsonl').open() as stream:
            for line in stream:
                verify(args.dataset, json.loads(line))
                count += 1
    print(json.dumps(dict(annotation_files=len(rows), images=count, sha256_verified=True)))


if __name__ == '__main__':
    main()
