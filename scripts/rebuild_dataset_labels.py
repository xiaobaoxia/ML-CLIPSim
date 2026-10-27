#!/usr/bin/env python3
"""Rebuild final labels and fixed splits from the released PCMP evidence."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlcs_metric.merge_cls_det_seg_and_make_labels import make_labels_from_loss


def read(path):
    return json.loads(path.read_text())


def key(row):
    return tuple(row[name] for name in ('image', 'p0', 'p1'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('dataset/pcmp'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.dataset.resolve():
        parser.error('use a separate output directory to preserve the release snapshot')
    root = args.dataset
    labels = make_labels_from_loss(read(root / 'pairs.json'), read(root / 'scores.json'),
                                   read(root / 'psnr.json'), force_odd=True)
    expected = read(root / 'labels.json')
    if labels != expected:
        raise ValueError('rebuilt labels differ from the published final label snapshot')
    indexed = {key(row): row for row in labels}
    initial = {key(row): row for row in read(root / 'records' / 'labels_initial.json')}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'labels.json').open('w') as stream:
        json.dump(labels, stream, indent=2)
    for split in ('train', 'val'):
        rows = []
        for identity in read(root / f'{split}_pairs.json'):
            identity = key(identity)
            # Preserve diagnostic fields; update the three supervised targets.
            row = initial[identity].copy()
            row.update({field: indexed[identity][field]
                        for field in ('winner', 'mean_vote', 'num_votes')})
            rows.append(row)
        if rows != read(root / 'split' / f'labels_{split}.json'):
            raise ValueError(f'{split} differs from the saved training split')
        (args.output / 'split').mkdir(exist_ok=True)
        with (args.output / 'split' / f'labels_{split}.json').open('w') as stream:
            json.dump(rows, stream, indent=2)
    print(json.dumps(dict(labels=len(labels), identical_full_records=True)))


if __name__ == '__main__':
    main()
