#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recompute segmentation losses only.

Outputs:
  {OUT_DIR}/scores_seg_only.json

Structure:
  scores_seg[pid][key]["seg"] = {...}
"""

import os
import json
import glob

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision


# Default paths and scoring parameters; command-line paths override these defaults.

DIST_ROOT = "distorted_samples/coco2000"
OUT_DIR = "mv_loss_out2000"
DEVICE = "cuda"

SEG_SCORES_PATH = os.path.join(OUT_DIR, "scores_seg_only.json")

DISABLE_DEEPLAB = False
DISABLE_MASKRCNN = False

DEEPLAB_MODE = "hard_ce"   # "hard_ce" or "soft_mse"

MASKRCNN_USE_SCORE_THRESH = False
MASKRCNN_REF_SCORE_THRESH = 0.5
MASKRCNN_DIST_SCORE_THRESH = 0.2
MASKRCNN_BOX_LOSS_WEIGHT = 1.0
MASKRCNN_MASK_LOSS_WEIGHT = 1.0


# utils

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


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-12)


def weighted_mean(losses, weights, eps=1e-12):
    if len(losses) == 0:
        return 0.0
    losses = np.array(losses, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    return float((losses * weights).sum() / (weights.sum() + eps))


def dice_loss_from_probs(prob_pred, prob_tgt, eps=1e-6):
    inter = (prob_pred * prob_tgt).sum()
    union = prob_pred.sum() + prob_tgt.sum()
    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice


def greedy_match_label_iou(ref_boxes, ref_labels, dist_boxes, dist_labels):
    used_dist = set()
    matches = []
    unmatched_ref = []

    for i in range(len(ref_boxes)):
        best_j = -1
        best_iou = -1.0
        for j in range(len(dist_boxes)):
            if j in used_dist:
                continue
            if dist_labels[j] != ref_labels[i]:
                continue
            iou = iou_xyxy(ref_boxes[i], dist_boxes[j])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j < 0:
            unmatched_ref.append(i)
        else:
            used_dist.add(best_j)
            matches.append((i, best_j, best_iou))
    return matches, unmatched_ref


# segmentation

@torch.no_grad()
def eval_deeplab_loss(dist_dir, pid, keys, device):
    out = {k: {"seg": {}} for k in keys}
    ref_dir = os.path.join(dist_dir, "ref")
    img_dir = os.path.join(dist_dir, pid)

    weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights).to(device).eval()
    preprocess = weights.transforms()

    for k in tqdm(keys, desc=f"[SEG] {pid} DeepLabV3"):
        rp = os.path.join(ref_dir, f"{k}.png")
        dp = os.path.join(img_dir, f"{k}.png")
        if not os.path.isfile(rp) or not os.path.isfile(dp):
            continue

        xr = preprocess(pil_rgb(rp)).unsqueeze(0).to(device)
        xd = preprocess(pil_rgb(dp)).unsqueeze(0).to(device)

        lr = model(xr)["out"]
        ld = model(xd)["out"]

        if DEEPLAB_MODE == "hard_ce":
            tgt = torch.argmax(lr, dim=1)
            loss = F.cross_entropy(ld, tgt).item()
        elif DEEPLAB_MODE == "soft_mse":
            pr = F.softmax(lr, dim=1)
            pd = F.softmax(ld, dim=1)
            loss = F.mse_loss(pd, pr).item()
        else:
            raise ValueError(f"Unknown DEEPLAB_MODE: {DEEPLAB_MODE}")

        out[k]["seg"]["deeplab_loss"] = float(loss)

    return out


@torch.no_grad()
def eval_maskrcnn_loss(dist_dir, pid, keys, device):
    out = {k: {"seg": {}} for k in keys}
    ref_dir = os.path.join(dist_dir, "ref")
    img_dir = os.path.join(dist_dir, pid)

    weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()
    preprocess = weights.transforms()

    for k in tqdm(keys, desc=f"[SEG] {pid} MaskRCNN"):
        rp = os.path.join(ref_dir, f"{k}.png")
        dp = os.path.join(img_dir, f"{k}.png")
        if not os.path.isfile(rp) or not os.path.isfile(dp):
            continue

        xr = preprocess(pil_rgb(rp)).unsqueeze(0).to(device)
        xd = preprocess(pil_rgb(dp)).unsqueeze(0).to(device)

        pr = model(xr)[0]
        pd = model(xd)[0]

        ref_boxes = pr["boxes"].detach().cpu().numpy()
        ref_labels = pr["labels"].detach().cpu().numpy().astype(np.int64)
        ref_scores = pr["scores"].detach().cpu().numpy()
        ref_masks = pr["masks"]

        if MASKRCNN_USE_SCORE_THRESH:
            keep_ref = ref_scores >= MASKRCNN_REF_SCORE_THRESH
            ref_boxes = ref_boxes[keep_ref]
            ref_labels = ref_labels[keep_ref]
            ref_scores = ref_scores[keep_ref]
            ref_masks = ref_masks[keep_ref]

        if len(ref_boxes) == 0:
            out[k]["seg"]["maskrcnn_loss"] = 0.0
            continue

        dist_boxes = pd["boxes"].detach().cpu().numpy()
        dist_labels = pd["labels"].detach().cpu().numpy().astype(np.int64)
        dist_scores = pd["scores"].detach().cpu().numpy()
        dist_masks = pd["masks"]

        if MASKRCNN_USE_SCORE_THRESH:
            keep_dist = dist_scores >= MASKRCNN_DIST_SCORE_THRESH
            dist_boxes = dist_boxes[keep_dist]
            dist_labels = dist_labels[keep_dist]
            dist_scores = dist_scores[keep_dist]
            dist_masks = dist_masks[keep_dist]

        if len(dist_boxes) == 0:
            out[k]["seg"]["maskrcnn_loss"] = 1.0
            continue

        matches, unmatched_ref = greedy_match_label_iou(ref_boxes, ref_labels, dist_boxes, dist_labels)

        losses, weights_arr = [], []
        for i, j, best_iou in matches:
            m_ref = ref_masks[i, 0].float()
            m_dist = dist_masks[j, 0].float()

            tgt_mask = m_ref.clamp(1e-6, 1.0 - 1e-6)
            pred_mask = m_dist.clamp(1e-6, 1.0 - 1e-6)

            bce = F.binary_cross_entropy(pred_mask, tgt_mask)
            dice = dice_loss_from_probs(pred_mask, tgt_mask)
            mask_loss = 0.5 * bce + 0.5 * dice

            box_loss = 1.0 - best_iou
            total_loss = MASKRCNN_BOX_LOSS_WEIGHT * box_loss + MASKRCNN_MASK_LOSS_WEIGHT * float(mask_loss.item())
            losses.append(total_loss)
            weights_arr.append(float(ref_scores[i]))

        for i in unmatched_ref:
            losses.append(1.0)
            weights_arr.append(float(ref_scores[i]))

        out[k]["seg"]["maskrcnn_loss"] = weighted_mean(losses, weights_arr)

    return out


# main

def main():
    import argparse
    global DIST_ROOT, OUT_DIR, DEVICE, SEG_SCORES_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dist-root', default='dataset/pcmp/images')
    parser.add_argument('--out-dir', default='outputs/pcmp')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    DIST_ROOT, OUT_DIR, DEVICE = args.dist_root, args.out_dir, args.device
    SEG_SCORES_PATH = os.path.join(OUT_DIR, 'scores_segmentation.json')
    dist_dir = DIST_ROOT
    ref_dir = os.path.join(dist_dir, "ref")
    keys = list_png_keys(ref_dir)
    point_ids = sorted([
        d for d in os.listdir(dist_dir)
        if os.path.isdir(os.path.join(dist_dir, d)) and d != "ref"
    ])

    scores = {pid: {k: {"seg": {}} for k in keys} for pid in point_ids}

    for pid in point_ids:
        if not DISABLE_DEEPLAB:
            r = eval_deeplab_loss(dist_dir, pid, keys, DEVICE)
            for k in keys:
                scores[pid][k]["seg"].update(r[k]["seg"])

        if not DISABLE_MASKRCNN:
            r = eval_maskrcnn_loss(dist_dir, pid, keys, DEVICE)
            for k in keys:
                scores[pid][k]["seg"].update(r[k]["seg"])

    write_json(SEG_SCORES_PATH, scores)
    print(f"[DONE] Segmentation scores saved to: {SEG_SCORES_PATH}")


if __name__ == "__main__":
    main()
