#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recompute detection losses only.

Outputs:
  {OUT_DIR}/scores_det_only.json

Structure:
  scores_det[pid][key]["det"] = {...}
"""

import os
import json
import glob
from typing import Dict

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision


# Default paths and scoring parameters; command-line paths override these defaults.

DIST_ROOT = "distorted_samples/coco2000"
OUT_DIR = "mv_loss_out2000"
DEVICE = "cuda"

DET_SCORES_PATH = os.path.join(OUT_DIR, "scores_det_only.json")
YOLO_WEIGHTS = "yolov8n.pt"

DISABLE_YOLO = False
DISABLE_FRCNN = False

YOLO_USE_SCORE_THRESH = False
YOLO_REF_SCORE_THRESH = 0.25
YOLO_DIST_SCORE_THRESH = 0.001
YOLO_SCORE_LOSS_WEIGHT = 0.0

FRCNN_USE_SCORE_THRESH = False
FRCNN_REF_SCORE_THRESH = 0.5
FRCNN_DIST_SCORE_THRESH = 0.2
FRCNN_SCORE_LOSS_WEIGHT = 0.0


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


# detection

@torch.no_grad()
def eval_yolo_loss(dist_dir, pid, keys, device):
    from ultralytics import YOLO

    out = {k: {"det": {}} for k in keys}
    ref_dir = os.path.join(dist_dir, "ref")
    img_dir = os.path.join(dist_dir, pid)

    model = YOLO(YOLO_WEIGHTS)

    for k in tqdm(keys, desc=f"[DET] {pid} YOLOv8n"):
        rp = os.path.join(ref_dir, f"{k}.png")
        dp = os.path.join(img_dir, f"{k}.png")
        if not os.path.isfile(rp) or not os.path.isfile(dp):
            continue

        tr = model.predict(rp, imgsz=640, conf=0.001, iou=0.7, max_det=500,
                           device=torch.device(device), half=False, augment=False,
                           agnostic_nms=False, verbose=False)[0]
        ts = model.predict(dp, imgsz=640, conf=0.001, iou=0.7, max_det=500,
                           device=torch.device(device), half=False, augment=False,
                           agnostic_nms=False, verbose=False)[0]

        if tr.boxes is None or len(tr.boxes) == 0:
            out[k]["det"]["yolo_loss"] = 0.0
            continue

        ref_boxes = tr.boxes.xyxy.detach().cpu().numpy()
        ref_labels = tr.boxes.cls.detach().cpu().numpy().astype(np.int64)
        ref_scores = tr.boxes.conf.detach().cpu().numpy()

        if YOLO_USE_SCORE_THRESH:
            keep_ref = ref_scores >= YOLO_REF_SCORE_THRESH
            ref_boxes = ref_boxes[keep_ref]
            ref_labels = ref_labels[keep_ref]
            ref_scores = ref_scores[keep_ref]

        if len(ref_boxes) == 0:
            out[k]["det"]["yolo_loss"] = 0.0
            continue

        if ts.boxes is None or len(ts.boxes) == 0:
            out[k]["det"]["yolo_loss"] = 1.0
            continue

        dist_boxes = ts.boxes.xyxy.detach().cpu().numpy()
        dist_labels = ts.boxes.cls.detach().cpu().numpy().astype(np.int64)
        dist_scores = ts.boxes.conf.detach().cpu().numpy()

        if YOLO_USE_SCORE_THRESH:
            keep_dist = dist_scores >= YOLO_DIST_SCORE_THRESH
            dist_boxes = dist_boxes[keep_dist]
            dist_labels = dist_labels[keep_dist]
            dist_scores = dist_scores[keep_dist]

        if len(dist_boxes) == 0:
            out[k]["det"]["yolo_loss"] = 1.0
            continue

        matches, unmatched_ref = greedy_match_label_iou(ref_boxes, ref_labels, dist_boxes, dist_labels)

        losses, weights = [], []
        for i, j, best_iou in matches:
            box_loss = 1.0 - best_iou
            score_loss = abs(float(ref_scores[i]) - float(dist_scores[j]))
            total_loss = box_loss + YOLO_SCORE_LOSS_WEIGHT * score_loss
            losses.append(total_loss)
            weights.append(float(ref_scores[i]))
        for i in unmatched_ref:
            losses.append(1.0)
            weights.append(float(ref_scores[i]))

        out[k]["det"]["yolo_loss"] = weighted_mean(losses, weights)

    return out


@torch.no_grad()
def eval_frcnn_loss(dist_dir, pid, keys, device):
    out = {k: {"det": {}} for k in keys}
    ref_dir = os.path.join(dist_dir, "ref")
    img_dir = os.path.join(dist_dir, pid)

    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()
    preprocess = weights.transforms()

    for k in tqdm(keys, desc=f"[DET] {pid} FasterRCNN"):
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

        if FRCNN_USE_SCORE_THRESH:
            keep_ref = ref_scores >= FRCNN_REF_SCORE_THRESH
            ref_boxes = ref_boxes[keep_ref]
            ref_labels = ref_labels[keep_ref]
            ref_scores = ref_scores[keep_ref]

        if len(ref_boxes) == 0:
            out[k]["det"]["frcnn_loss"] = 0.0
            continue

        dist_boxes = pd["boxes"].detach().cpu().numpy()
        dist_labels = pd["labels"].detach().cpu().numpy().astype(np.int64)
        dist_scores = pd["scores"].detach().cpu().numpy()

        if FRCNN_USE_SCORE_THRESH:
            keep_dist = dist_scores >= FRCNN_DIST_SCORE_THRESH
            dist_boxes = dist_boxes[keep_dist]
            dist_labels = dist_labels[keep_dist]
            dist_scores = dist_scores[keep_dist]

        if len(dist_boxes) == 0:
            out[k]["det"]["frcnn_loss"] = 1.0
            continue

        matches, unmatched_ref = greedy_match_label_iou(ref_boxes, ref_labels, dist_boxes, dist_labels)

        losses, weights_arr = [], []
        for i, j, best_iou in matches:
            box_loss = 1.0 - best_iou
            score_loss = abs(float(ref_scores[i]) - float(dist_scores[j]))
            total_loss = box_loss + FRCNN_SCORE_LOSS_WEIGHT * score_loss
            losses.append(total_loss)
            weights_arr.append(float(ref_scores[i]))
        for i in unmatched_ref:
            losses.append(1.0)
            weights_arr.append(float(ref_scores[i]))

        out[k]["det"]["frcnn_loss"] = weighted_mean(losses, weights_arr)

    return out


# main

def main():
    import argparse
    global DIST_ROOT, OUT_DIR, DEVICE, DET_SCORES_PATH, YOLO_WEIGHTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dist-root', default='dataset/pcmp/images')
    parser.add_argument('--out-dir', default='outputs/pcmp')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--yolo-weights', default='yolov8n.pt')
    args = parser.parse_args()
    YOLO_WEIGHTS = args.yolo_weights
    DIST_ROOT, OUT_DIR, DEVICE = args.dist_root, args.out_dir, args.device
    DET_SCORES_PATH = os.path.join(OUT_DIR, 'scores_detection.json')
    dist_dir = DIST_ROOT
    ref_dir = os.path.join(dist_dir, "ref")
    keys = list_png_keys(ref_dir)
    point_ids = sorted([
        d for d in os.listdir(dist_dir)
        if os.path.isdir(os.path.join(dist_dir, d)) and d != "ref"
    ])

    scores = {pid: {k: {"det": {}} for k in keys} for pid in point_ids}

    for pid in point_ids:
        if not DISABLE_YOLO:
            r = eval_yolo_loss(dist_dir, pid, keys, DEVICE)
            for k in keys:
                scores[pid][k]["det"].update(r[k]["det"])

        if not DISABLE_FRCNN:
            r = eval_frcnn_loss(dist_dir, pid, keys, DEVICE)
            for k in keys:
                scores[pid][k]["det"].update(r[k]["det"])

    write_json(DET_SCORES_PATH, scores)
    print(f"[DONE] Detection scores saved to: {DET_SCORES_PATH}")


if __name__ == "__main__":
    main()
