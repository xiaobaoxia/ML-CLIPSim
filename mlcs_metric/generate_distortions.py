#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate reference PNGs and 66 image distortion points per input image.

Write per-image PSNR records and per-point summary statistics. Learned-codec
points use forward reconstruction; this script does not measure bitrates."""

import os, json, argparse, math, io, random
from typing import Dict, List, Tuple
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

# Learned-codec dependencies are required when --enable_codecs is enabled.
try:
    from compressai.zoo import bmshj2018_hyperprior, cheng2020_attn
    HAS_COMPRESSAI = True
except Exception:
    HAS_COMPRESSAI = False


# Utils
def pil_to_tensor(im: Image.Image) -> torch.Tensor:
    arr = np.asarray(im).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    return torch.from_numpy(arr).permute(2, 0, 1)

def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.clamp(0, 1).cpu().numpy()
    return Image.fromarray((x.transpose(1, 2, 0) * 255.0).astype(np.uint8))

def compute_psnr(x, y, eps=1e-12):
    # x,y: torch [1,3,H,W] in [0,1]
    mse = F.mse_loss(x, y).item()
    if mse <= eps:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)

def pad_to_multiple(x, m=64):
    _, _, h, w = x.shape
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    pad = (ph // 2, ph - ph // 2, pw // 2, pw - pw // 2)  # (top,bottom,left,right)
    x_pad = F.pad(x, (pad[2], pad[3], pad[0], pad[1]), mode="replicate")
    return x_pad, pad

def unpad(x, pad):
    pt, pb, pl, pr = pad
    return x[:, :, pt:x.shape[2] - pb, pl:x.shape[3] - pr]

def percentile(v: List[float], p: float) -> float:
    if len(v) == 0:
        return float("nan")
    return float(np.percentile(np.asarray(v, dtype=np.float32), p))


# Image-space CODECS (points)
def jpeg_point(q, subsampling="4:2:0"):
    def fn(im: Image.Image):
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=int(q), subsampling=subsampling)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    return fn

def webp_point(q):
    def fn(im: Image.Image):
        buf = io.BytesIO()
        im.save(buf, format="WEBP", quality=int(q))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    return fn


# Learned CODECS (points) - optional
def bmshj_point(model, device):
    def fn(im: Image.Image):
        x = pil_to_tensor(im).unsqueeze(0).to(device)
        with torch.no_grad():
            x_pad, pad = pad_to_multiple(x, 64)
            out = model(x_pad)
            x_hat = unpad(out["x_hat"], pad)
        return tensor_to_pil(x_hat[0])
    return fn

def cheng_point(model, device):
    def fn(im: Image.Image):
        x = pil_to_tensor(im).unsqueeze(0).to(device)
        with torch.no_grad():
            x_pad, pad = pad_to_multiple(x, 64)
            out = model(x_pad)
            x_hat = unpad(out["x_hat"], pad)
        return tensor_to_pil(x_hat[0])
    return fn


def gaussian_noise_point(std01):
    # std01 in [0,1] space; 0.01 ~= 2.55/255
    std01 = float(std01)
    def fn(im: Image.Image):
        arr = np.asarray(im).astype(np.float32) / 255.0
        arr = np.clip(arr + np.random.randn(*arr.shape).astype(np.float32) * std01, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))
    return fn

def poisson_noise_point(scale=1.0):
    scale = float(scale)
    def fn(im: Image.Image):
        arr = np.asarray(im).astype(np.float32) / 255.0
        lam = np.clip(arr * 255.0 * scale, 1e-3, None)
        noisy = np.random.poisson(lam).astype(np.float32) / (255.0 * scale)
        return Image.fromarray(np.clip(noisy * 255, 0, 255).astype(np.uint8))
    return fn

def gaussian_blur_point(sigma):
    sigma = float(sigma)
    def fn(im: Image.Image):
        return im.filter(ImageFilter.GaussianBlur(radius=sigma))
    return fn

def motion_blur_point(k: int):
    """
    Horizontal motion blur via 1D box filter with EDGE padding.
    - No black borders
    - Output size exactly matches input size
    """
    k = int(k)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    r = k // 2

    def fn(im: Image.Image):
        arr = np.asarray(im).astype(np.float32)  # [H,W,3]
        pad = np.pad(arr, ((0, 0), (r, r), (0, 0)), mode="edge")  # [H,W+2r,3]
        c = np.cumsum(pad, axis=1, dtype=np.float32)              # [H,W+2r,3]
        z = np.zeros((c.shape[0], 1, c.shape[2]), dtype=np.float32)
        c = np.concatenate([z, c], axis=1)                        # [H,W+2r+1,3]
        win = c[:, k:, :] - c[:, :-k, :]                          # [H,W,3]
        out = win / float(k)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    return fn

def chroma_jitter_point(delta=6.0, seed=None):
    """
      Convert to YCbCr, add small bias to Cb/Cr, convert back.
    delta in [0..255] range (e.g., 4~12)
    """
    delta = float(delta)
    rng = np.random.RandomState(seed) if seed is not None else np.random

    def fn(im: Image.Image):
        ycbcr = im.convert("YCbCr")
        arr = np.asarray(ycbcr).astype(np.float32)
        dc = rng.uniform(-delta, delta)
        dr = rng.uniform(-delta, delta)
        arr[..., 1] = np.clip(arr[..., 1] + dc, 0, 255)
        arr[..., 2] = np.clip(arr[..., 2] + dr, 0, 255)
        return Image.fromarray(arr.astype(np.uint8), mode="YCbCr").convert("RGB")
    return fn

def gamma_point(gamma=1.05):
    gamma = float(gamma)
    def fn(im: Image.Image):
        arr = np.asarray(im).astype(np.float32) / 255.0
        arr = np.clip(arr, 1e-6, 1.0)
        out = np.power(arr, gamma)
        return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8))
    return fn

def contrast_point(factor=0.97):
    factor = float(factor)
    def fn(im: Image.Image):
        return ImageEnhance.Contrast(im).enhance(factor)
    return fn

def brightness_point(factor=0.97):
    factor = float(factor)
    def fn(im: Image.Image):
        return ImageEnhance.Brightness(im).enhance(factor)
    return fn

def saturation_point(factor=0.97):
    factor = float(factor)
    def fn(im: Image.Image):
        return ImageEnhance.Color(im).enhance(factor)
    return fn

def resize_down_up_point(scale=0.75, interp="bicubic"):
    """
    Downsample then upsample back to original size.
    """
    scale = float(scale)
    interp = str(interp).lower()
    if interp == "bilinear":
        res = Image.Resampling.BILINEAR
    elif interp == "nearest":
        res = Image.Resampling.NEAREST
    else:
        res = Image.Resampling.BICUBIC

    def fn(im: Image.Image):
        w, h = im.size
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        small = im.resize((nw, nh), resample=res)
        back = small.resize((w, h), resample=res)
        return back
    return fn


# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_list", default="lists/coco_train2017_2000.txt")
    ap.add_argument("--out_root", default="dataset/pcmp/images")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--enable_codecs", action=argparse.BooleanOptionalAction, default=True, help="include the twelve learned-codec distortion points")
    ap.add_argument("--seed", type=int, default=123, help="global RNG seed for reproducibility")
    args = ap.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device

    os.makedirs(args.out_root, exist_ok=True)
    ref_dir = os.path.join(args.out_root, "ref")
    os.makedirs(ref_dir, exist_ok=True)

    # Optional: Load learned codecs
    bmshj_models = {}
    cheng_models = {}
    if args.enable_codecs:
        if not HAS_COMPRESSAI:
            raise RuntimeError("compressai not available, but --enable_codecs was set.")
        bmshj_models = {q: bmshj2018_hyperprior(q, pretrained=True).to(device).eval() for q in [1, 2, 3, 4, 5,6]}
        cheng_models = {q: cheng2020_attn(q, pretrained=True).to(device).eval() for q in [1, 2, 3, 4, 5,6]}

    # Define ALL POINTS
    points: List[Tuple[str, callable, str]] = []

    for q in [10, 15, 20, 25, 30, 40, 50, 70]:
        points.append((f"jpeg420_q{q}", jpeg_point(q, "4:2:0"), "codec"))
        points.append((f"jpeg444_q{q}", jpeg_point(q, "4:4:4"), "codec"))
        points.append((f"webp_q{q}", webp_point(q), "codec"))

    # ---- optional learned codecs ----
    if args.enable_codecs:
        # Six quality levels per learned codec.
        for q, m in bmshj_models.items():
            points.append((f"bmshj_q{q}", bmshj_point(m, device), "codec"))
        for q, m in cheng_models.items():
            points.append((f"cheng_q{q}", cheng_point(m, device), "codec"))

    points += [
        ("resize_0.85_bicubic", resize_down_up_point(0.85, "bicubic"), "resample"),
        ("resize_0.85_bilinear", resize_down_up_point(0.85, "bilinear"), "resample"),

        ("resize_0.75_bicubic", resize_down_up_point(0.75, "bicubic"), "resample"),
        ("resize_0.75_bilinear", resize_down_up_point(0.75, "bilinear"), "resample"),

        ("resize_0.50_bicubic", resize_down_up_point(0.50, "bicubic"), "resample"),
        ("resize_0.50_bilinear", resize_down_up_point(0.50, "bilinear"), "resample"),

        ("resize_0.35_bicubic", resize_down_up_point(0.35, "bicubic"), "resample"),
        ("resize_0.35_bilinear", resize_down_up_point(0.35, "bilinear"), "resample"),
    ]

    # ---- noise / blur / color / tone ----
    points += [
        # Noise (std in [0,1] space; 0.02 ~= 5.1/255)
        ("noise_g0.010", gaussian_noise_point(0.010), "noise"),
        ("noise_g0.020", gaussian_noise_point(0.020), "noise"),
        ("noise_g0.030", gaussian_noise_point(0.030), "noise"),
        ("poisson_x1", poisson_noise_point(1.0), "noise"),
        ("poisson_x0.6", poisson_noise_point(0.6), "noise"),
        ("poisson_x0.4", poisson_noise_point(0.4), "noise"),

        # Blur (PIL radius)
        ("blur_s0.8", gaussian_blur_point(0.8), "blur"),
        ("blur_s1.2", gaussian_blur_point(1.2), "blur"),
        ("blur_s1.8", gaussian_blur_point(1.8), "blur"),
        ("blur_s2.4", gaussian_blur_point(2.4), "blur"),
        ("motion_k9", motion_blur_point(9), "blur"),
        ("motion_k13", motion_blur_point(13), "blur"),

        ("chroma_d6", chroma_jitter_point(delta=6.0, seed=args.seed + 7), "color"),
        ("chroma_d12", chroma_jitter_point(delta=12.0, seed=args.seed + 11), "color"),

        ("gamma_0.90", gamma_point(0.90), "tone"),
        ("gamma_1.10", gamma_point(1.10), "tone"),
        ("contrast_0.85", contrast_point(0.85), "tone"),
        ("contrast_1.15", contrast_point(1.15), "tone"),

        ("gamma_0.75", gamma_point(0.75), "tone"),
        ("gamma_1.25", gamma_point(1.25), "tone"),
        ("contrast_0.70", contrast_point(0.70), "tone"),
        ("contrast_1.25", contrast_point(1.25), "tone"),
    ]
    # Pre-create output dirs (faster)
    for pid, _, _ in points:
        os.makedirs(os.path.join(args.out_root, pid), exist_ok=True)

    # Run
    with open(args.img_list) as f:
        img_paths = [l.strip() for l in f if l.strip()]

    per_image_stats = []
    point_psnr: Dict[str, List[float]] = {}

    for path in tqdm(img_paths, desc="Generating points"):
        name = os.path.splitext(os.path.basename(path))[0]
        im = Image.open(path).convert("RGB")
        im.save(os.path.join(ref_dir, f"{name}.png"))

        x_ref = pil_to_tensor(im).unsqueeze(0)

        for pid, fn, ptype in points:
            rec = fn(im)
            x_hat = pil_to_tensor(rec).unsqueeze(0)

            if x_hat.shape != x_ref.shape:
                raise RuntimeError(
                    f"Shape mismatch: point={pid}, image={name}, ref={tuple(x_ref.shape)}, hat={tuple(x_hat.shape)}"
                )

            psnr = compute_psnr(x_ref, x_hat)

            out_path = os.path.join(args.out_root, pid, f"{name}.png")
            rec.save(out_path)

            per_image_stats.append({
                "image": name,
                "point_id": pid,
                "psnr": float(psnr),
                "path": out_path,
                "type": ptype,
            })
            point_psnr.setdefault(pid, []).append(psnr)

    # Save outputs
    with open(os.path.join(args.out_root, "per_image_stats.jsonl"), "w") as f:
        for r in per_image_stats:
            f.write(json.dumps(r) + "\n")

    point_stats = {}
    for pid, v in point_psnr.items():
        point_stats[pid] = {
            "n": int(len(v)),
            "mean_psnr": float(np.mean(v)),
            "std_psnr": float(np.std(v)),
            "p5_psnr": percentile(v, 5),
            "p50_psnr": percentile(v, 50),
            "p95_psnr": percentile(v, 95),
        }

    with open(os.path.join(args.out_root, "point_stats.json"), "w") as f:
        json.dump(point_stats, f, indent=2)

    print("[OK] Generated distortion points.")
    print("[OK] per_image_stats.jsonl / point_stats.json written.")


if __name__ == "__main__":
    main()