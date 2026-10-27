"""Load and export ML-CLIPSim metric checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

try:
    from .metric import CLIPMultiLayerMetric
except ImportError:  # Support direct script execution.
    from metric import CLIPMultiLayerMetric

LEARNABLE_KEYS = ("group_weight", "global_weight", "attn_mix_logit")



def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    raise ValueError("Unrecognized checkpoint format; expected keys 'model' or 'state_dict'.")


def is_learnable_only_checkpoint(state_dict: Dict[str, torch.Tensor]) -> bool:
    return set(state_dict.keys()).issubset(set(LEARNABLE_KEYS))


def load_mlcs_metric(
    ckpt_path: Optional[Union[str, Path]] = None,
    device: str = "cuda",
    use_attn_weight: bool = False,
    freeze: bool = True,
) -> torch.nn.Module:
    """
    Load pretrained ML-CLIPSim (primary API).

    Example
    -------
    >>> metric = load_mlcs_metric(device="cuda")
    >>> score = metric(ref, dist)  # [B], higher = better quality

    Parameters
    ----------
    ckpt_path : path or None
        Lightweight .pt (~2KB, default) or full training .pth (~571MB).
    device : str
        "cuda" or "cpu".
    use_attn_weight : bool
        Must be False for the released metric checkpoint.

    Inputs (forward)
    ----------------
    ref, dist : FloatTensor [B, 3, H, W], RGB, values in [0, 1].
    """
    if ckpt_path is None:
        if use_attn_weight:
            raise ValueError("the released checkpoint uses use_attn_weight=False")
        from mlclipsim import load_metric
        return load_metric(device=device, freeze=freeze)
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Download weights from the README, or run:\n"
            "  python mlcs_metric/export_metric_weights.py --src <full_checkpoint.pth>"
        )

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)

    model = CLIPMultiLayerMetric(device=device, use_attn_weight=use_attn_weight).to(device)
    if is_learnable_only_checkpoint(state_dict):
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # CLIP backbone stays from OpenAI pretrained weights
        allowed_missing = [k for k in missing if k.startswith("clip_model.")]
        if len(allowed_missing) != len(missing):
            raise RuntimeError(f"Unexpected missing keys when loading learnable weights: {missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
    else:
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    if freeze:
        # This still permits gradients with respect to the distorted input.
        model.requires_grad_(False)
    return model


def export_learnable_weights(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    extra_meta: Optional[dict] = None,
) -> Path:
    """Extract learnable tensors from a full training checkpoint into a small .pt file."""
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    try:
        ckpt = torch.load(src_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(src_path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)
    learnable = {k: state_dict[k] for k in LEARNABLE_KEYS if k in state_dict}
    if not learnable:
        raise ValueError(f"No learnable keys found in {src_path}")

    payload = {
        "format": "mlcs_learnable_v1",
        "model": "ML-CLIPSim",
        "model_class": "CLIPMultiLayerMetric",
        "checkpoint_file": "ML-CLIPSim_ViTB16.pt",
        "clip_backbone": "ViT-B/16",
        "use_attn_weight": False,
        "group_mode": "fixed12",
        "source_checkpoint": src_path.name,
        "validation": ckpt.get("metrics") if isinstance(ckpt, dict) else None,
        "best_acc": ckpt.get("best_acc") if isinstance(ckpt, dict) else None,
        "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "state_dict": learnable,
    }
    if extra_meta:
        payload.update(extra_meta)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dst_path)
    return dst_path
