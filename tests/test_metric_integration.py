"""Opt-in tests using the real CLIP ViT-B/16 encoder.

Set MLCLIPSIM_CLIP_CHECKPOINT to a local OpenAI ViT-B-16.pt to run offline.
These tests compare the public package against the original research metric.
"""

import os
from unittest.mock import patch

import pytest
import torch

from mlclipsim import MLCLIPSim, MLCLIPSimLoss


@pytest.mark.skipif(
    not os.environ.get("MLCLIPSIM_CLIP_CHECKPOINT"),
    reason="set MLCLIPSIM_CLIP_CHECKPOINT to run with the real CLIP backbone",
)
def test_real_clip_identity_research_parity_and_gradient():
    from mlcs_metric.metric import CLIPMultiLayerMetric

    torch.manual_seed(123)
    metric = MLCLIPSim.from_pretrained(
        device="cpu", clip_checkpoint=os.environ["MLCLIPSIM_CLIP_CHECKPOINT"]
    )
    reference = torch.rand(1, 3, 32, 48)
    candidate = torch.rand(1, 3, 32, 48, requires_grad=True)
    # Reuse the same frozen backbone so parity checks need no second download.
    with patch("clip.load", return_value=(metric.clip_model, None)):
        research_metric = CLIPMultiLayerMetric(device="cpu", use_attn_weight=False)
    research_metric.load_state_dict(
        {key: metric.state_dict()[key] for key in
         ("group_weight", "global_weight", "attn_mix_logit")}, strict=False
    )
    research_metric.requires_grad_(False).eval()
    score = metric(reference, candidate)
    with torch.no_grad():
        expected = research_metric(reference, candidate)
        identity = metric(reference, reference)
    torch.testing.assert_close(score.detach(), expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(identity, torch.ones_like(identity), atol=1e-5, rtol=0)
    MLCLIPSimLoss(metric)(reference, candidate).backward()
    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()
    assert candidate.grad.abs().sum() > 0
    assert reference.grad is None
    assert all(parameter.grad is None for parameter in metric.parameters())
