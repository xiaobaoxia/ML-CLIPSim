#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate multi-model preference labels from COCO reference and distorted images.

Teacher = model prediction on ref image
Student = model prediction on distorted image
Score   = model-specific consistency loss (lower is better)

Models (default voters = 7, odd to avoid ties):
- Classification (3): ResNet50 / ViT-B/16 / ConvNeXt-Tiny  (KL loss)
- Detection     (2): YOLOv8n + Faster R-CNN (pseudo-GT matching loss)
- Segmentation  (2): DeepLabV3 (hard CE or soft MSE) + Mask R-CNN (box + soft mask loss)
"""

import os, json, glob, argparse, math
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision

YOLO_WEIGHTS = "yolov8n.pt"
from torchvision.transforms.functional import to_tensor


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
                dist = F.interpolate(dist.unsqueeze(0), ref.shape[-2:], mode="bilinear", align_corners=False)[0]
            psnr_dict[pid][k] = float(psnr(ref, dist))

    write_json(out_json, psnr_dict)
    return psnr_dict


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    areaA = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    areaB = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (areaA + areaB - inter + 1e-12)

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
        rb = ref_boxes[i]
        rl = ref_labels[i]
        best_j = -1
        best_iou = -1.0
        for j in range(len(dist_boxes)):
            if j in used_dist:
                continue
            if dist_labels[j] != rl:
                continue
            iou = iou_xyxy(rb, dist_boxes[j])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j < 0:
            unmatched_ref.append(i)
        else:
            used_dist.add(best_j)
            matches.append((i, best_j, best_iou))
    return matches, unmatched_ref


@torch.no_grad()
def eval_cls_loss(dist_dir, pid, keys, device, cls_models: List[str]):
    out = {k: {"cls": {}} for k in keys}
    ref_dir = os.path.join(dist_dir, "ref")
    img_dir = os.path.join(dist_dir, pid)
    registry = {
        "resnet50": (torchvision.models.resnet50, torchvision.models.ResNet50_Weights.DEFAULT),
        "vit_b_16": (torchvision.models.vit_b_16, torchvision.models.ViT_B_16_Weights.DEFAULT),
        "convnext_tiny": (torchvision.models.convnext_tiny, torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT),
    }
    for name in cls_models:
        ctor, weights = registry[name]
        model = ctor(weights=weights).to(device).eval()
        preprocess = weights.transforms()
        for k in tqdm(keys, desc=f"[CLS] {pid} {name}"):
            rp = os.path.join(ref_dir, f"{k}.png")
            dp = os.path.join(img_dir, f"{k}.png")
            if not os.path.isfile(rp) or not os.path.isfile(dp):
                continue
            xr = preprocess(pil_rgb(rp)).unsqueeze(0).to(device)
            xd = preprocess(pil_rgb(dp)).unsqueeze(0).to(device)
            lr = model(xr)
            ld = model(xd)
            loss = F.kl_div(F.log_softmax(ld, dim=1), F.softmax(lr, dim=1), reduction="batchmean").item()
            out[k]["cls"][f"{name}_loss"] = float(loss)
    return out


@torch.no_grad()
def eval_yolo_loss(dist_dir, pid, keys, device,
                   use_score_thresh=False,
                   ref_score_thresh=0.25,
                   dist_score_thresh=0.001,
                   score_loss_weight=0.0):
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
        tr = model.predict(rp, imgsz=640, conf=0.001, iou=0.7, max_det=500, device=torch.device(device),
                           half=False, augment=False, agnostic_nms=False, verbose=False)[0]
        ts = model.predict(dp, imgsz=640, conf=0.001, iou=0.7, max_det=500, device=torch.device(device),
                           half=False, augment=False, agnostic_nms=False, verbose=False)[0]
        if tr.boxes is None or len(tr.boxes) == 0:
            out[k]["det"]["yolo_loss"] = 0.0
            continue
        ref_boxes = tr.boxes.xyxy.detach().cpu().numpy()
        ref_labels = tr.boxes.cls.detach().cpu().numpy().astype(np.int64)
        ref_scores = tr.boxes.conf.detach().cpu().numpy()
        if use_score_thresh:
            keep_ref = ref_scores >= ref_score_thresh
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
        if use_score_thresh:
            keep_dist = dist_scores >= dist_score_thresh
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
            losses.append(box_loss + score_loss_weight * score_loss)
            weights.append(float(ref_scores[i]))
        for i in unmatched_ref:
            losses.append(1.0)
            weights.append(float(ref_scores[i]))
        out[k]["det"]["yolo_loss"] = weighted_mean(losses, weights)
    return out


@torch.no_grad()
def eval_frcnn_loss(dist_dir, pid, keys, device,
                    use_score_thresh=True,
                    ref_score_thresh=0.5,
                    dist_score_thresh=0.2,
                    score_loss_weight=0.0):
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
        ref_keep = torch.ones_like(pr["scores"], dtype=torch.bool)
        dist_keep = torch.ones_like(pd["scores"], dtype=torch.bool)
        if use_score_thresh:
            ref_keep = pr["scores"] >= ref_score_thresh
            dist_keep = pd["scores"] >= dist_score_thresh
        ref_boxes = pr["boxes"][ref_keep].detach().cpu().numpy()
        ref_labels = pr["labels"][ref_keep].detach().cpu().numpy().astype(np.int64)
        ref_scores = pr["scores"][ref_keep].detach().cpu().numpy()
        if len(ref_boxes) == 0:
            out[k]["det"]["frcnn_loss"] = 0.0
            continue
        dist_boxes = pd["boxes"][dist_keep].detach().cpu().numpy()
        dist_labels = pd["labels"][dist_keep].detach().cpu().numpy().astype(np.int64)
        dist_scores = pd["scores"][dist_keep].detach().cpu().numpy()
        if len(dist_boxes) == 0:
            out[k]["det"]["frcnn_loss"] = 1.0
            continue
        matches, unmatched_ref = greedy_match_label_iou(ref_boxes, ref_labels, dist_boxes, dist_labels)
        losses, weights = [], []
        for i, j, best_iou in matches:
            box_loss = 1.0 - best_iou
            score_loss = abs(float(ref_scores[i]) - float(dist_scores[j]))
            losses.append(box_loss + score_loss_weight * score_loss)
            weights.append(float(ref_scores[i]))
        for i in unmatched_ref:
            losses.append(1.0)
            weights.append(float(ref_scores[i]))
        out[k]["det"]["frcnn_loss"] = weighted_mean(losses, weights)
    return out


@torch.no_grad()
def eval_deeplab_loss(dist_dir, pid, keys, device, mode="hard_ce"):
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
        if mode == "hard_ce":
            tgt = torch.argmax(lr, dim=1)
            loss = F.cross_entropy(ld, tgt).item()
        else:
            pr = F.softmax(lr, dim=1)
            pd = F.softmax(ld, dim=1)
            loss = F.mse_loss(pd, pr).item()
        out[k]["seg"]["deeplab_loss"] = float(loss)
    return out


@torch.no_grad()
def eval_maskrcnn_loss(dist_dir, pid, keys, device,
                       use_score_thresh=False,
                       ref_score_thresh=0.5,
                       dist_score_thresh=0.2,
                       box_loss_weight=1.0,
                       mask_loss_weight=1.0):
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
        ref_keep = torch.ones_like(pr["scores"], dtype=torch.bool)
        dist_keep = torch.ones_like(pd["scores"], dtype=torch.bool)
        if use_score_thresh:
            ref_keep = pr["scores"] >= ref_score_thresh
            dist_keep = pd["scores"] >= dist_score_thresh
        ref_boxes = pr["boxes"][ref_keep].detach().cpu().numpy()
        ref_labels = pr["labels"][ref_keep].detach().cpu().numpy().astype(np.int64)
        ref_scores = pr["scores"][ref_keep].detach().cpu().numpy()
        ref_masks = pr["masks"][ref_keep]
        if len(ref_boxes) == 0:
            out[k]["seg"]["maskrcnn_loss"] = 0.0
            continue
        dist_boxes = pd["boxes"][dist_keep].detach().cpu().numpy()
        dist_labels = pd["labels"][dist_keep].detach().cpu().numpy().astype(np.int64)
        dist_masks = pd["masks"][dist_keep]
        if len(dist_boxes) == 0:
            out[k]["seg"]["maskrcnn_loss"] = 1.0
            continue
        matches, unmatched_ref = greedy_match_label_iou(ref_boxes, ref_labels, dist_boxes, dist_labels)
        losses, weights = [], []
        for i, j, best_iou in matches:
            tgt_mask = ref_masks[i, 0].float().clamp(1e-6, 1.0 - 1e-6)
            pred_mask = dist_masks[j, 0].float().clamp(1e-6, 1.0 - 1e-6)
            bce = F.binary_cross_entropy(pred_mask, tgt_mask)
            dice = dice_loss_from_probs(pred_mask, tgt_mask)
            box_loss = 1.0 - best_iou
            total_loss = box_loss_weight * box_loss + mask_loss_weight * float((0.5 * bce + 0.5 * dice).item())
            losses.append(total_loss)
            weights.append(float(ref_scores[i]))
        for i in unmatched_ref:
            losses.append(1.0)
            weights.append(float(ref_scores[i]))
        out[k]["seg"]["maskrcnn_loss"] = weighted_mean(losses, weights)
    return out


def sample_pairs(keys, psnr, tol):
    pairs = []
    pids = list(psnr.keys())
    for k in keys:
        vals = [(pid, psnr[pid].get(k)) for pid in pids if k in psnr[pid]]
        vals = [(pid, v) for pid, v in vals if v is not None]
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if abs(vals[i][1] - vals[j][1]) <= tol:
                    pairs.append({"image": k, "p0": vals[i][0], "p1": vals[j][0]})
    return pairs

def make_labels_from_loss(pairs, scores_by_point, psnr_dict, force_odd=True):
    labels = []
    for it in pairs:
        k, p0, p1 = it["image"], it["p0"], it["p1"]
        a = scores_by_point[p0][k]
        b = scores_by_point[p1][k]
        task_dump = {}
        vote_dump = {}
        votes = []
        diffs = []
        for task in a:
            for m in a[task]:
                if m not in b[task]:
                    continue
                v0 = a[task][m]
                v1 = b[task][m]
                task_dump.setdefault(task, {})[m] = {"p0": float(v0), "p1": float(v1)}
                if v0 == v1:
                    continue
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
            "psnr": {"p0": float(psnr_dict[p0][k]), "p1": float(psnr_dict[p1][k])},
            "tasks": task_dump,
            "votes": vote_dump,
            "mean_vote": mean_vote,
            "num_votes": len(votes),
            "winner": winner
        })
    return labels


def main():
    global YOLO_WEIGHTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist_root", default="dataset/pcmp/images")
    ap.add_argument("--out_dir", default="mv_loss_out2000")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--psnr_tol", type=float, default=0.5)
    ap.add_argument("--disable_yolo", action="store_true")
    ap.add_argument("--disable_frcnn", action="store_true")
    ap.add_argument("--disable_deeplab", action="store_true")
    ap.add_argument("--disable_maskrcnn", action="store_true")
    ap.add_argument("--cls_models", default="resnet50,vit_b_16,convnext_tiny")
    ap.add_argument("--yolo_use_score_thresh", action="store_true")
    ap.add_argument("--yolo_ref_score_thresh", type=float, default=0.25)
    ap.add_argument("--yolo_dist_score_thresh", type=float, default=0.001)
    ap.add_argument("--yolo_score_loss_weight", type=float, default=0.0)
    ap.add_argument("--frcnn_use_score_thresh", action="store_true")
    ap.add_argument("--frcnn_ref_score_thresh", type=float, default=0.5)
    ap.add_argument("--frcnn_dist_score_thresh", type=float, default=0.2)
    ap.add_argument("--frcnn_score_loss_weight", type=float, default=0.0)
    ap.add_argument("--deeplab_mode", type=str, default="hard_ce", choices=["hard_ce", "soft_mse"])
    ap.add_argument("--maskrcnn_use_score_thresh", action="store_true")
    ap.add_argument("--maskrcnn_ref_score_thresh", type=float, default=0.5)
    ap.add_argument("--maskrcnn_dist_score_thresh", type=float, default=0.2)
    ap.add_argument("--maskrcnn_box_loss_weight", type=float, default=1.0)
    ap.add_argument("--maskrcnn_mask_loss_weight", type=float, default=1.0)
    ap.add_argument("--force_odd", action="store_true")
    ap.add_argument("--yolo-weights", default="yolov8n.pt")
    args = ap.parse_args()
    YOLO_WEIGHTS = args.yolo_weights

    device = args.device
    dist_dir = os.path.join(args.dist_root)
    ref_dir = os.path.join(dist_dir, "ref")
    out_dir = os.path.join(args.out_dir)
    keys = list_png_keys(ref_dir)
    point_ids = [d for d in os.listdir(dist_dir) if os.path.isdir(os.path.join(dist_dir, d)) and d != "ref"]
    cls_models = [s.strip() for s in args.cls_models.split(",") if s.strip()]
    scores = {pid: {k: {"cls": {}, "det": {}, "seg": {}} for k in keys} for pid in point_ids}
    psnr = compute_psnr_table(dist_dir, point_ids, keys, device, os.path.join(dist_dir, "psnr.json"))

    for pid in point_ids:
        r = eval_cls_loss(dist_dir, pid, keys, device, cls_models)
        for k in keys:
            scores[pid][k]["cls"].update(r[k]["cls"])
        if not args.disable_yolo:
            r = eval_yolo_loss(dist_dir, pid, keys, device, args.yolo_use_score_thresh,
                               args.yolo_ref_score_thresh, args.yolo_dist_score_thresh, args.yolo_score_loss_weight)
            for k in keys:
                scores[pid][k]["det"].update(r[k]["det"])
        if not args.disable_frcnn:
            r = eval_frcnn_loss(dist_dir, pid, keys, device, args.frcnn_use_score_thresh,
                                args.frcnn_ref_score_thresh, args.frcnn_dist_score_thresh, args.frcnn_score_loss_weight)
            for k in keys:
                scores[pid][k]["det"].update(r[k]["det"])
        if not args.disable_deeplab:
            r = eval_deeplab_loss(dist_dir, pid, keys, device, args.deeplab_mode)
            for k in keys:
                scores[pid][k]["seg"].update(r[k]["seg"])
        if not args.disable_maskrcnn:
            r = eval_maskrcnn_loss(dist_dir, pid, keys, device, args.maskrcnn_use_score_thresh,
                                   args.maskrcnn_ref_score_thresh, args.maskrcnn_dist_score_thresh,
                                   args.maskrcnn_box_loss_weight, args.maskrcnn_mask_loss_weight)
            for k in keys:
                scores[pid][k]["seg"].update(r[k]["seg"])

    pairs = sample_pairs(keys, psnr, args.psnr_tol)
    labels = make_labels_from_loss(pairs, scores, psnr, force_odd=args.force_odd)
    write_json(os.path.join(out_dir, "scores.json"), scores)
    write_json(os.path.join(out_dir, "pairs.json"), pairs)
    write_json(os.path.join(out_dir, "labels.json"), labels)
    if labels:
        mv = np.mean([x["mean_vote"] for x in labels])
        print(f"DONE. labels={len(labels)}  mean(mean_vote)={mv:.4f}")
    else:
        print("DONE. labels=0 (no valid votes/pairs?)")

if __name__ == "__main__":
    main()
