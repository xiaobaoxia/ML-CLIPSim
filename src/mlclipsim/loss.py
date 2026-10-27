"""Loss wrapper for training image models with ML-CLIPSim."""

from __future__ import annotations

from typing import Literal, Optional
from pathlib import Path

from torch import Tensor, nn

from .checkpoint import load_metric
from .metric import MLCLIPSim


class MLCLIPSimLoss(nn.Module):
    """Compute ``1 - MLCLIPSim(reference, reconstruction)``.

    Metric parameters are frozen by default. Gradients still propagate through
    ``reconstruction`` to the model being trained.
    """

    def __init__(
        self,
        metric: Optional[MLCLIPSim] = None,
        *,
        reduction: Literal["none", "mean", "sum"] = "mean",
        device: str | None = None,
        checkpoint: str | Path | None = None,
        clip_checkpoint: str | Path | None = None,
        download_root: str | Path | None = None,
    ) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: none, mean, sum")
        self.metric = metric if metric is not None else load_metric(
            checkpoint, device=device, freeze=True,
            clip_checkpoint=clip_checkpoint, download_root=download_root,
        )
        self.metric.freeze()
        self.reduction = reduction

    def forward(self, reference: Tensor, reconstruction: Tensor) -> Tensor:
        loss = 1.0 - self.metric(reference, reconstruction)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
