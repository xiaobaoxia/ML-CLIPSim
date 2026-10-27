#!/usr/bin/env python3
"""Resolve the exact ordered 2,000 COCO references on another machine."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coco-root', type=Path, required=True,
                        help='COCO train2017 directory containing JPEG files')
    parser.add_argument('--manifest', type=Path,
                        default=Path('dataset/pcmp/reference_images.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())['images']
    paths = []
    for row in rows:
        path = args.coco_root / row['filename']
        if not path.is_file():
            raise FileNotFoundError(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['source_sha256']:
            raise ValueError(f'COCO source bytes differ: {path}')
        paths.append(str(path.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(paths) + '\n')
    print(f'Verified {len(paths)} references in the saved generation order.')


if __name__ == '__main__':
    main()
