"""Load pretrained MLIC++ codecs fine-tuned with ML-CLIPSim."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import os

import torch

from MLIC_pp.config.config import model_config
from MLIC_pp.models import MLICPlusPlus

WEIGHTS_ROOT = Path(__file__).resolve().parent.parent / "weights" / "codec"
MANIFEST_PATH = WEIGHTS_ROOT / "manifest.json"
DEFAULT_REVISION = "main"

# Rate point index -> filename suffix (q1 = lowest bitrate in this release set)
RATE_POINTS = (1, 2, 3, 4, 5)


def _checkpoint_path(quality: int, weights_root: Optional[Path] = None) -> Path:
    if quality not in RATE_POINTS:
        raise ValueError(f"quality must be one of {RATE_POINTS}, got {quality}")
    root = Path(weights_root) if weights_root is not None else Path(
        os.environ.get("MLCLIPSIM_CODEC_WEIGHTS", WEIGHTS_ROOT)
    )
    return root / f"mlicpp_ml_clipsim_q{quality}.pt"


def _load_checkpoint_file(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _resolve_checkpoint(
    quality: int,
    *,
    weights_root: Optional[Union[str, Path]],
    download: bool,
    repo_id: Optional[str],
    revision: str,
    cache_dir: Optional[Union[str, Path]],
) -> Path:
    local_path = _checkpoint_path(
        quality,
        Path(weights_root) if weights_root is not None else None,
    )
    if local_path.is_file():
        return local_path
    if not download:
        raise FileNotFoundError(f"Codec checkpoint not found: {local_path}")

    if not repo_id:
        raise ValueError("download=True requires an explicit repo_id; otherwise download "
                         "the GitHub Release assets into weights_root.")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Automatic codec download requires `pip install -r requirements-research.txt`."
        ) from exc

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=local_path.name,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not find {local_path.name} locally or download it from "
            f"https://huggingface.co/{repo_id}. The codec release may not be published yet."
        ) from exc
    return Path(downloaded)


def load_mlicpp_mlclipsim(
    quality: int = 3,
    device: Optional[str] = None,
    weights_root: Optional[Union[str, Path]] = None,
    strict: bool = True,
    download: bool = False,
    repo_id: Optional[str] = None,
    revision: str = DEFAULT_REVISION,
    cache_dir: Optional[Union[str, Path]] = None,
    checkpoint: Optional[Union[str, Path]] = None,
) -> MLICPlusPlus:
    """
    Load MLIC++ fine-tuned with ML-CLIPSim.

    Parameters
    ----------
    quality : int
        Rate point in {1, 2, 3, 4, 5}. Higher index = higher bitrate.
    device : str
        Target device for the codec.
    weights_root : path, optional
        Directory containing ``mlicpp_ml_clipsim_q*.pt`` files.
    strict : bool
        Passed to ``load_state_dict``.

    Returns
    -------
    MLICPlusPlus
        Eval-mode codec with loaded weights.
    """
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if quality not in RATE_POINTS:
        raise ValueError(f"quality must be one of {RATE_POINTS}, got {quality}")
    ckpt_path = Path(checkpoint) if checkpoint is not None else _resolve_checkpoint(
        quality,
        weights_root=weights_root,
        download=download,
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
    )

    model = MLICPlusPlus(model_config()).eval()
    ckpt = _load_checkpoint_file(ckpt_path, map_location="cpu")

    if "lic_state" in ckpt:
        state_dict = ckpt["lic_state"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif ckpt and all(isinstance(value, torch.Tensor) for value in ckpt.values()):
        state_dict = ckpt
    else:
        raise ValueError(f"Unrecognized codec checkpoint format: {ckpt_path}")

    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Codec checkpoint {ckpt_path} does not match this MLIC++ architecture. "
            "Use the released ML-CLIPSim codecs, matching MSE initialization, or a "
            "checkpoint produced by scripts/train_mlic.py. Different MLIC++ synthesis/LRP "
            f"variants are not interchangeable. Original error: {exc}"
        ) from exc
    return model.to(target_device)


def list_available_rate_points(weights_root: Optional[Union[str, Path]] = None) -> list[int]:
    """Return rate point indices whose checkpoint files exist on disk."""
    root = _checkpoint_path(1, Path(weights_root) if weights_root is not None else None).parent
    found = []
    for q in RATE_POINTS:
        if (root / f"mlicpp_ml_clipsim_q{q}.pt").is_file():
            found.append(q)
    return found


def write_manifest(weights_root: Optional[Union[str, Path]] = None) -> Path:
    """Refresh file sizes and SHA256 values, preserving existing model metadata."""
    import hashlib
    root = Path(weights_root) if weights_root is not None else WEIGHTS_ROOT
    target = root / "manifest.json"
    manifest = json.loads(target.read_text()) if target.is_file() else {
        "model": "MLIC++", "objective": "ML-CLIPSim", "checkpoints": []}
    previous = {row["filename"]: row for row in manifest["checkpoints"]}
    entries = []
    for quality in RATE_POINTS:
        path = _checkpoint_path(quality, root)
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        row = dict(previous.get(path.name, {}))
        row.update(quality=quality, filename=path.name, size_bytes=path.stat().st_size,
                   sha256=digest.hexdigest())
        entries.append(row)
    if not entries:
        raise FileNotFoundError(f"No released codec files found in {root}")
    manifest["checkpoints"] = entries
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    return target
