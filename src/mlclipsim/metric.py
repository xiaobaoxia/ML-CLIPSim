"""Differentiable ML-CLIPSim metric.

This module implements the released ViT-B/16 metric configuration. The CLIP
backbone is frozen; only the lightweight aggregation parameters were learned.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _l2_normalize(x: Tensor, eps: float = 1e-6) -> Tensor:
    x = x.float()
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class MLCLIPSim(nn.Module):
    """Multi-layer CLIP similarity for full-reference image quality.

    Inputs are RGB tensors shaped ``[B, 3, H, W]`` with values in ``[0, 1]``.
    The output has shape ``[B]`` and a higher score means the candidate better
    preserves machine-relevant information from the reference.

    The released checkpoint uses all 12 ViT-B/16 layers, three contiguous layer
    groups, uniform patch aggregation, and no attention weighting.
    """

    def __init__(
        self,
        *,
        device: str | torch.device = "cpu",
        backbone: str = "ViT-B/16",
        input_size: int = 224,
        layers: Optional[Iterable[int]] = None,
        tau_group: float = 2.0,
        tau_gate: float = 2.0,
        clip_model: Optional[nn.Module] = None,
        clip_checkpoint: str | Path | None = None,
        download_root: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.input_size = int(input_size)
        self.tau_group = float(tau_group)
        self.tau_gate = float(tau_gate)
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if self.tau_group <= 0 or self.tau_gate <= 0:
            raise ValueError("tau_group and tau_gate must be positive")
        if clip_checkpoint is not None and not Path(clip_checkpoint).is_file():
            raise FileNotFoundError(f"CLIP backbone checkpoint not found: {clip_checkpoint}")

        if clip_model is None:
            try:
                import clip
            except ImportError as exc:  # pragma: no cover - dependency error path
                raise ImportError(
                    "OpenAI CLIP is required. Install the project with `pip install -e .`."
                ) from exc
            clip_model, _ = clip.load(
                str(clip_checkpoint) if clip_checkpoint is not None else backbone,
                device=device,
                jit=False,
                download_root=str(download_root) if download_root is not None else None,
            )

        self.clip_model = clip_model.float().eval()
        for parameter in self.clip_model.parameters():
            parameter.requires_grad_(False)

        self.num_layers = len(self.clip_model.visual.transformer.resblocks)
        self.layers = list(range(self.num_layers)) if layers is None else list(layers)
        if not self.layers:
            raise ValueError("layers must contain at least one transformer layer")
        if min(self.layers) < 0 or max(self.layers) >= self.num_layers:
            raise ValueError(
                f"layers must be within [0, {self.num_layers - 1}], got {self.layers}"
            )
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("layers must not contain duplicates")
        if max(self.layers) > 11:
            raise ValueError("the released fixed12 grouping supports layers 0 through 11")

        self.layer_groups = self._make_fixed_groups()
        self.group_weight = nn.Parameter(torch.zeros(len(self.layer_groups)))
        self.global_weight = nn.Parameter(torch.zeros(1))

        # Kept for checkpoint compatibility. It is unused by the released model.
        self.attn_mix_logit = nn.Parameter(torch.tensor([0.4054651081081644]))

        self.register_buffer(
            "clip_mean",
            torch.tensor(CLIP_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_std",
            torch.tensor(CLIP_STD).view(1, 3, 1, 1),
            persistent=False,
        )

        self._feature_buffer: list[Optional[Tensor]] = [None] * self.num_layers
        self._hook_handles = self._register_hooks()
        self.to(device)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path | None = None,
        *,
        device: str | torch.device | None = None,
        freeze: bool = True,
        clip_checkpoint: str | Path | None = None,
        download_root: str | Path | None = None,
    ) -> "MLCLIPSim":
        """Load the released ML-CLIPSim checkpoint."""
        from .checkpoint import load_metric

        return load_metric(
            checkpoint, device=device, freeze=freeze,
            clip_checkpoint=clip_checkpoint, download_root=download_root,
        )

    def _make_fixed_groups(self) -> list[list[int]]:
        groups: list[list[int]] = []
        for lower, upper in ((0, 3), (4, 7), (8, 11)):
            selected_indices = [
                index
                for index, layer_id in enumerate(self.layers)
                if lower <= layer_id <= upper
            ]
            if selected_indices:
                groups.append(selected_indices)
        if not groups:
            raise ValueError("released fixed12 grouping does not cover the selected layers")
        return groups

    def _register_hooks(self):
        handles = []

        def make_hook(layer_index: int):
            def hook(_module, _inputs, output):
                if output is not None and output.dim() == 3:
                    self._feature_buffer[layer_index] = output.permute(1, 0, 2).contiguous()

            return hook

        for index, block in enumerate(self.clip_model.visual.transformer.resblocks):
            handles.append(block.register_forward_hook(make_hook(index)))
        return handles

    def train(self, mode: bool = True):
        """Set aggregation mode while always keeping the frozen backbone in eval mode."""
        super().train(mode)
        self.clip_model.eval()
        return self

    def freeze(self) -> "MLCLIPSim":
        """Freeze every metric parameter while preserving gradients to input images."""
        self.requires_grad_(False)
        self.eval()
        return self

    def _validate_inputs(self, reference: Tensor, candidate: Tensor) -> None:
        if not isinstance(reference, Tensor) or not isinstance(candidate, Tensor):
            raise TypeError("reference and candidate must be torch tensors")
        if reference.ndim != 4 or candidate.ndim != 4:
            raise ValueError("reference and candidate must be 4D tensors [B, 3, H, W]")
        if reference.shape[0] != candidate.shape[0]:
            raise ValueError("reference and candidate must have the same batch size")
        if reference.shape[1] != 3 or candidate.shape[1] != 3:
            raise ValueError("reference and candidate must contain three RGB channels")
        if not reference.is_floating_point() or not candidate.is_floating_point():
            raise TypeError("reference and candidate must be floating-point tensors")
        if any(size == 0 for image in (reference, candidate) for size in image.shape):
            raise ValueError("reference and candidate dimensions must be nonempty")
        if reference.device != candidate.device or reference.device != self.clip_mean.device:
            raise ValueError("reference, candidate, and metric must be on the same device")

    def _preprocess(self, image: Tensor) -> Tensor:
        image = image.float()
        if image.shape[-2:] != (self.input_size, self.input_size):
            image = F.interpolate(
                image,
                size=(self.input_size, self.input_size),
                mode="bicubic",
                align_corners=False,
            )
        image = image.clamp(0, 1)
        return (image - self.clip_mean) / self.clip_std

    def _encode(self, image: Tensor, *, detach: bool) -> tuple[Tensor, list[Tensor]]:
        for index in range(self.num_layers):
            self._feature_buffer[index] = None

        autocast_context = (
            torch.autocast(device_type="cuda", enabled=False)
            if image.device.type == "cuda"
            else nullcontext()
        )
        try:
            with autocast_context:
                if detach:
                    with torch.no_grad():
                        embedding = self.clip_model.encode_image(image.float())
                else:
                    embedding = self.clip_model.encode_image(image.float())
            features = [self._feature_buffer[index] for index in self.layers]
        finally:
            # Do not retain the candidate's autograd graph between training steps.
            self._feature_buffer[:] = [None] * self.num_layers
        if any(feature is None for feature in features):
            raise RuntimeError("failed to capture one or more CLIP transformer features")
        return embedding, [feature for feature in features if feature is not None]

    def _layer_weights(self, device: torch.device) -> Tensor:
        group_weights = torch.softmax(self.group_weight / self.tau_group, dim=0)
        layer_weights = torch.zeros(len(self.layers), device=device)
        for group_index, indices in enumerate(self.layer_groups):
            per_layer = group_weights[group_index] / len(indices)
            for index in indices:
                layer_weights[index] = per_layer
        # Preserve the released implementation exactly.
        return layer_weights / (layer_weights.sum() + 1e-6)

    def forward(self, reference: Tensor, candidate: Tensor) -> Tensor:
        self._validate_inputs(reference, candidate)
        reference = self._preprocess(reference)
        candidate = self._preprocess(candidate)

        reference_embedding, reference_features = self._encode(reference, detach=True)
        candidate_embedding, candidate_features = self._encode(candidate, detach=False)

        layer_weights = self._layer_weights(reference.device)
        token_score = torch.zeros(reference.shape[0], device=reference.device)
        for index, (reference_tokens, candidate_tokens) in enumerate(
            zip(reference_features, candidate_features)
        ):
            reference_tokens = _l2_normalize(reference_tokens.float())
            candidate_tokens = _l2_normalize(candidate_tokens.float())
            patch_similarity = (
                reference_tokens[:, 1:] * candidate_tokens[:, 1:]
            ).sum(dim=-1)
            token_score = token_score + layer_weights[index] * patch_similarity.mean(dim=1)

        reference_global = _l2_normalize(reference_embedding.float())
        candidate_global = _l2_normalize(candidate_embedding.float())
        global_score = (reference_global.detach() * candidate_global).sum(dim=-1)

        gate = torch.sigmoid(self.global_weight / self.tau_gate)
        return gate * token_score + (1.0 - gate) * global_score

    @torch.no_grad()
    def score(self, reference: Tensor, candidate: Tensor) -> Tensor:
        """Inference-only convenience method."""
        return self(reference, candidate)
