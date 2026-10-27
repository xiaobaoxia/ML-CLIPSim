#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Merge classification, detection and segmentation consistency scores.

Use a fixed candidate-pair list to compute per-model votes, soft preferences
and hard winners."""

import os
import json
import glob
import math

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor


# Default paths and scoring parameters; command-line paths override these defaults.

DIST_ROOT = "distorted_samples/coco2000"
OUT_DIR = "mv_loss_out2000"
DEVICE = "cuda"
FORCE_ODD = True

OLD_CLS_SCORES_PATH = os.path.join(OUT_DIR, "scores.json")
OLD_PAIRS_PATH = os.path.join(OUT_DIR, "pairs.json")

DET_SCORES_PATH = os.path.join(OUT_DIR, "scores_det_only.json")
SEG_SCORES_PATH = os.path.join(OUT_DIR, "scores_seg_only.json")

# outputs
MERGED_SCORES_PATH = os.path.join(OUT_DIR, "scores_cls_det_seg_merged.json")
LABELS_PATH = os.path.join(OUT_DIR, "labels_cls_det_seg_merged.json")


# utils

def read_json(p):
    with open(p, "r") as f:
        return json.load(f)


def write_json(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)


def pil_rgb(p):
    return Image.open(p).convert("RGB")


def list_png_keys(d):
    return sorted([
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(d, "*.png"))
    ])


@torch.no_grad()
def compute_psnr_table(dist_dir, point_ids, keys, device, out_json):
    def psnr(a, b):
        mse = torch.mean((a - b) ** 2).item()
        return 99.0 if mse < 1e-12 else 10 * math.log10(1.0 / mse)

    psnr_dict = {}
    if os.path.isfile(out_json):
        psnr_dict = read_json(out_json)

    for pid in point_ids:
        psnr_dict.setdefault(pid, {})

    ref_dir = os.path.join(dist_dir, "ref")

    for k in tqdm(keys, desc="[PSNR]"):
        rp = os.path.join(ref_dir, f"{k}.png")
        if not os.path.isfile(rp):
            continue
        ref = to_tensor(pil_rgb(rp)).to(device)

        for pid in point_ids:
            if k in psnr_dict[pid]:
                continue
            dp = os.path.join(dist_dir, pid, f"{k}.png")
            if not os.path.isfile(dp):
                continue
            dist = to_tensor(pil_rgb(dp)).to(device)
            if dist.shape != ref.shape:
                dist = F.interpolate(
                    dist.unsqueeze(0),
                    ref.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )[0]
            psnr_dict[pid][k] = float(psnr(ref, dist))

    write_json(out_json, psnr_dict)
    return psnr_dict


def make_labels_from_loss(pairs, scores_by_point, psnr_dict, force_odd=True):
    labels = []

    for it in pairs:
        k, p0, p1 = it["image"], it["p0"], it["p1"]

        # skip broken entries safely
        if p0 not in scores_by_point or p1 not in scores_by_point:
            continue
        if k not in scores_by_point[p0] or k not in scores_by_point[p1]:
            continue

        a = scores_by_point[p0][k]
        b = scores_by_point[p1][k]

        task_dump = {}
        vote_dump = {}
        votes = []
        diffs = []

        for task in a:
            if task not in b:
                continue
            for m in a[task]:
                if m not in b[task]:
                    continue

                v0 = a[task][m]
                v1 = b[task][m]

                task_dump.setdefault(task, {})[m] = {
                    "p0": float(v0),
                    "p1": float(v1)
                }

                # equal -> skip this metric
                if v0 == v1:
                    continue

                # lower loss = better
                vote = 1 if v0 < v1 else 0
                vote_dump.setdefault(task, {})[m] = vote
                votes.append(vote)
                diffs.append((abs(float(v0) - float(v1)), task, m))

        if len(votes) == 0:
            continue

        if force_odd and (len(votes) % 2 == 0):
            min_idx = int(np.argmin([d[0] for d in diffs]))
            drop_task, drop_metric = diffs[min_idx][1], diffs[min_idx][2]

            votes2 = []
            vote_dump2 = {}

            for task in task_dump:
                for m in task_dump[task]:
                    if task == drop_task and m == drop_metric:
                        continue

                    v0 = task_dump[task][m]["p0"]
                    v1 = task_dump[task][m]["p1"]

                    if v0 == v1:
                        continue

                    vote = 1 if v0 < v1 else 0
                    vote_dump2.setdefault(task, {})[m] = vote
                    votes2.append(vote)

            votes = votes2
            vote_dump = vote_dump2

            if len(votes) == 0:
                continue

        mean_vote = float(np.mean(votes))
        winner = int(mean_vote > 0.5)

        labels.append({
            "image": k,
            "p0": p0,
            "p1": p1,
            "psnr": {
                "p0": float(psnr_dict[p0][k]),
                "p1": float(psnr_dict[p1][k])
            },
            "tasks": task_dump,
            "votes": vote_dump,
            "mean_vote": mean_vote,
            "num_votes": len(votes),
            "winner": winner,
        })

    return labels


# main

def main():
    import argparse
    global DIST_ROOT, OUT_DIR, DEVICE, OLD_CLS_SCORES_PATH, OLD_PAIRS_PATH
    global DET_SCORES_PATH, SEG_SCORES_PATH, MERGED_SCORES_PATH, LABELS_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dist-root', default='dataset/pcmp/images')
    parser.add_argument('--out-dir', default='outputs/pcmp')
    parser.add_argument('--classification-scores', default='dataset/pcmp/records/scores_initial.json')
    parser.add_argument('--detection-scores', default='dataset/pcmp/records/scores_detection.json')
    parser.add_argument('--segmentation-scores', default='dataset/pcmp/records/scores_segmentation.json')
    parser.add_argument('--pairs', default='dataset/pcmp/pairs.json')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    DIST_ROOT, OUT_DIR, DEVICE = args.dist_root, args.out_dir, args.device
    OLD_CLS_SCORES_PATH, OLD_PAIRS_PATH = args.classification_scores, args.pairs
    DET_SCORES_PATH, SEG_SCORES_PATH = args.detection_scores, args.segmentation_scores
    MERGED_SCORES_PATH = os.path.join(OUT_DIR, 'scores.json')
    LABELS_PATH = os.path.join(OUT_DIR, 'labels.json')
    old_scores = read_json(OLD_CLS_SCORES_PATH)
    det_scores = read_json(DET_SCORES_PATH)
    seg_scores = read_json(SEG_SCORES_PATH)
    old_pairs = read_json(OLD_PAIRS_PATH)

    dist_dir = DIST_ROOT
    ref_dir = os.path.join(dist_dir, "ref")
    keys = list_png_keys(ref_dir)
    point_ids = sorted([
        d for d in os.listdir(dist_dir)
        if os.path.isdir(os.path.join(dist_dir, d)) and d != "ref"
    ])

    merged = {
        pid: {k: {"cls": {}, "det": {}, "seg": {}} for k in keys}
        for pid in point_ids
    }

    for pid in point_ids:
        for k in keys:
            if pid in old_scores and k in old_scores[pid] and "cls" in old_scores[pid][k]:
                merged[pid][k]["cls"] = old_scores[pid][k]["cls"]

            if pid in det_scores and k in det_scores[pid] and "det" in det_scores[pid][k]:
                merged[pid][k]["det"] = det_scores[pid][k]["det"]

            if pid in seg_scores and k in seg_scores[pid] and "seg" in seg_scores[pid][k]:
                merged[pid][k]["seg"] = seg_scores[pid][k]["seg"]

    psnr = compute_psnr_table(
        dist_dir,
        point_ids,
        keys,
        DEVICE,
        os.path.join(dist_dir, "psnr.json")
    )

    labels = make_labels_from_loss(
        old_pairs,
        merged,
        psnr,
        force_odd=FORCE_ODD
    )

    write_json(MERGED_SCORES_PATH, merged)
    write_json(LABELS_PATH, labels)

    print(f"[DONE] merged scores: {MERGED_SCORES_PATH}")
    print(f"[DONE] reused old pairs: {OLD_PAIRS_PATH}")
    print(f"[DONE] labels: {LABELS_PATH}")

    if labels:
        mv = np.mean([x["mean_vote"] for x in labels])
        print(f"[STATS] labels={len(labels)} mean(mean_vote)={mv:.6f}")
    else:
        print("[STATS] labels=0")


if __name__ == "__main__":
    main()