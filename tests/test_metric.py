import pytest


torch = pytest.importorskip("torch")
nn = torch.nn

from mlclipsim.loss import MLCLIPSimLoss
from mlclipsim.metric import MLCLIPSim


class _FakeBlock(nn.Module):
    def forward(self, tokens):
        return tokens + 0.01


class _FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.resblocks = nn.ModuleList([_FakeBlock() for _ in range(12)])


class _FakeVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = _FakeTransformer()


class _FakeCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _FakeVisual()

    def encode_image(self, image):
        batch = image.shape[0]
        pooled = image.mean(dim=(-2, -1))
        cls = pooled.unsqueeze(0)
        patches = pooled.unsqueeze(0).repeat(4, 1, 1)
        tokens = torch.cat([cls, patches], dim=0)
        for block in self.visual.transformer.resblocks:
            tokens = block(tokens)
        return tokens[0].reshape(batch, -1)


def _metric():
    return MLCLIPSim(device="cpu", clip_model=_FakeCLIP())


def test_identity_score_is_close_to_one():
    metric = _metric().freeze()
    image = torch.rand(2, 3, 16, 16)
    score = metric.score(image, image)
    assert score.shape == (2,)
    assert torch.all(score > 0.99)


def test_loss_freezes_metric_but_preserves_reconstruction_gradient():
    criterion = MLCLIPSimLoss(metric=_metric())
    reference = torch.rand(2, 3, 16, 16)
    reconstruction = torch.rand(2, 3, 16, 16, requires_grad=True)
    loss = criterion(reference, reconstruction)
    loss.backward()
    assert reconstruction.grad is not None
    assert torch.isfinite(reconstruction.grad).all()
    assert all(parameter.grad is None for parameter in criterion.metric.parameters())
    assert all(not parameter.requires_grad for parameter in criterion.metric.parameters())


def test_rejects_invalid_channels():
    metric = _metric()
    with pytest.raises(ValueError, match="RGB"):
        metric(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8))


def test_load_bundled_weights_and_keep_backbone_frozen(monkeypatch):
    import clip

    monkeypatch.setattr(clip, "load", lambda *args, **kwargs: (_FakeCLIP(), None))
    metric = MLCLIPSim.from_pretrained(device="cpu", freeze=False)
    assert metric.group_weight.requires_grad
    assert metric.group_weight[1] > metric.group_weight[0]
    metric.train()
    assert not metric.clip_model.training


def test_missing_checkpoint_fails_before_loading_backbone(tmp_path, monkeypatch):
    import clip

    def unexpected_download(*args, **kwargs):
        pytest.fail("an invalid metric checkpoint must fail before CLIP is loaded")

    monkeypatch.setattr(clip, "load", unexpected_download)
    with pytest.raises(FileNotFoundError):
        MLCLIPSim.from_pretrained(tmp_path / "missing.pt", device="cpu")


def test_losses_support_all_reductions_and_clear_feature_graph():
    reference = torch.rand(2, 3, 16, 16)
    reconstruction = torch.rand(2, 3, 16, 16, requires_grad=True)
    metric = _metric().freeze()
    vector = MLCLIPSimLoss(metric, reduction="none")(reference, reconstruction)
    total = MLCLIPSimLoss(metric, reduction="sum")(reference, reconstruction)
    assert vector.shape == (2,)
    torch.testing.assert_close(total, vector.sum())
    assert all(feature is None for feature in metric._feature_buffer)
    total.backward()
    assert reconstruction.grad.abs().sum() > 0


@pytest.mark.parametrize("shape", [(0, 3, 8, 8), (1, 3, 0, 8), (1, 3, 8)])
def test_rejects_empty_or_wrong_rank(shape):
    with pytest.raises(ValueError):
        _metric()(torch.rand(shape), torch.rand(shape))


def test_rejects_integer_images():
    with pytest.raises(TypeError, match="floating"):
        _metric()(torch.zeros(1, 3, 8, 8, dtype=torch.uint8), torch.rand(1, 3, 8, 8))


def test_different_source_resolutions_are_resized():
    score = _metric()(torch.rand(1, 3, 12, 17), torch.rand(1, 3, 9, 11))
    assert score.shape == (1,)
    assert torch.isfinite(score).all()
