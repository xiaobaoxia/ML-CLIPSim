#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample a reproducible list of COCO train2017 images.

Usage:
  python gen_coco_list.py --coco_root /path/to/coco2017/images --n 2000 --seed 123 \
      --out lists/coco_train2017_2000.txt
"""

import os
import glob
import random
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", default="/path/to/coco2017/images", help="COCO root, containing train2017/")
    ap.add_argument("--split", default="train2017", help="train2017 or val2017")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default="lists/coco_train2017_2000.txt", help="output txt, one absolute path per line")
    args = ap.parse_args()

    img_dir = os.path.join(args.coco_root, args.split)
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"split dir not found: {img_dir}")

    imgs = []
    imgs += glob.glob(os.path.join(img_dir, "*.jpg"))
    imgs += glob.glob(os.path.join(img_dir, "*.jpeg"))
    imgs += glob.glob(os.path.join(img_dir, "*.png"))

    imgs = sorted(imgs)
    if len(imgs) == 0:
        raise RuntimeError(f"no images found under: {img_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(imgs)

    n = min(args.n, len(imgs))
    imgs = imgs[:n]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for p in imgs:
            f.write(os.path.abspath(p) + "\n")

    print(f"[OK] wrote {len(imgs)} paths to {args.out}")
    print(f"[OK] first 3:\n" + "\n".join(imgs[:3]))


if __name__ == "__main__":
    main()