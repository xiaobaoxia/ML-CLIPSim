"""Checkpoint loading for ML-CLIPSim."""

from __future__ import annotations

import json
import hashlib
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import torch

from .metric import MLCLIPSim


LEARNABLE_KEYS = ("group_weight", "global_weight", "attn_mix_logit")


def _default_asset(name: str):
    return resources.files("mlclipsim").joinpath("weights", name)


def _torch_load(path, *, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location=map_location)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("unsupported ML-CLIPSim checkpoint format")


def load_metric(
    checkpoint: str | Path | None = None,
    *,
    device: str | torch.device | None = None,
    freeze: bool = True,
    clip_checkpoint: str | Path | None = None,
    download_root: str | Path | None = None,
) -> MLCLIPSim:
    """Load the released metric with its versioned architecture configuration.

    ``checkpoint`` contains the lightweight aggregation weights. For offline
    use, pass the separate OpenAI ViT-B/16 backbone as ``clip_checkpoint``;
    otherwise CLIP uses its normal cache/download mechanism.
    """
    target_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    config_asset = _default_asset("ML-CLIPSim_ViTB16.config.json")
    with config_asset.open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"ML-CLIPSim checkpoint not found: {checkpoint_path}")
        payload = _torch_load(checkpoint_path, map_location="cpu")
    else:
        checkpoint_asset = _default_asset("ML-CLIPSim_ViTB16.pt")
        with resources.as_file(checkpoint_asset) as checkpoint_path:
            actual_hash = _sha256(checkpoint_path)
            expected_hash = config["checkpoint_sha256"]
            if actual_hash != expected_hash:
                raise RuntimeError(
                    "bundled ML-CLIPSim checkpoint failed SHA256 verification: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
            payload = _torch_load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(payload)
    learnable = {key: state_dict[key] for key in LEARNABLE_KEYS if key in state_dict}
    missing_learnable = set(LEARNABLE_KEYS) - set(learnable)
    if missing_learnable:
        raise ValueError(f"checkpoint is missing parameters: {sorted(missing_learnable)}")
    model = MLCLIPSim(
        device=target_device,
        backbone=config["backbone"],
        input_size=config["input_size"],
        layers=config["layers"],
        tau_group=config["tau_group"],
        tau_gate=config["tau_gate"],
        clip_checkpoint=clip_checkpoint,
        download_root=download_root,
    )
    model.load_state_dict(learnable, strict=False)
    model.to(target_device)
    if freeze:
        model.freeze()
    else:
        model.eval()
    return model
