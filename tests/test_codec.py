"""Compression tests use a reduced channel count for fast CPU regression coverage."""
import copy
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("compressai")
pytest.importorskip("einops")

from codec.checkpoint_utils import load_mlicpp_mlclipsim
from scripts.train_mlic import configure_optimizers, pad_image, rate_distortion, save_checkpoint, train_step
from MLIC_pp.models import MLICPlusPlus


@pytest.fixture
def model():
    torch.set_num_threads(2)
    return MLICPlusPlus(SimpleNamespace(N=16, M=32, slice_num=1, context_window=5))


def test_codec_gradients_and_entropy_update(model):
    model.train()
    optim, aux_optim = configure_optimizers(model, 1e-4, 1e-3)
    images = torch.rand(1, 3, 64, 64)
    weight = model.g_a.analysis_transform[0].conv1.weight
    before = weight.detach().clone()
    stats = train_step(model, images, optim, aux_optim, objective="mse", lmbda=.01)
    assert all(torch.isfinite(torch.tensor(value)) for value in stats.values())
    assert not torch.equal(weight, before)


def test_codec_cpu_bitstream_round_trip(model):
    model.eval()
    model.update(force=True)
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        expected = model(image)["x_hat"]
        encoded = model.compress(image)
        decoded = model.decompress(encoded["strings"], encoded["shape"])["x_hat"]
    assert decoded.shape == image.shape
    assert torch.allclose(decoded, expected, atol=1e-5, rtol=1e-5)


def test_codec_shapes_and_bitstream_batch(model):
    with pytest.raises(ValueError, match="multiples of 64"):
        model(torch.rand(1, 3, 65, 64))
    with pytest.raises(ValueError, match="one image"):
        model.compress(torch.rand(2, 3, 64, 64))
    images = torch.rand(1, 3, 37, 91)
    assert pad_image(images).shape == (1, 3, 64, 128)


def test_codec_strict_load_is_respected(model):
    state = copy.copy(model.state_dict())
    del state["g_a.analysis_transform.0.conv1.bias"]
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(state)
    result = model.load_state_dict(state, strict=False)
    assert "g_a.analysis_transform.0.conv1.bias" in result.missing_keys


def test_rate_uses_original_unpadded_pixels():
    image = torch.zeros(1, 3, 31, 45)
    output = {"x_hat": torch.zeros(1, 3, 64, 64),
              "likelihoods": {"y": torch.full((1, 2, 4, 4), .5)}}
    loss, bpp, distortion = rate_distortion(output, image, "mse", .01)
    assert torch.allclose(bpp, torch.tensor(32 / (31 * 45)))
    assert distortion == 0
    assert loss == bpp


def test_codec_invalid_rate_point():
    with pytest.raises(ValueError, match="quality must be"):
        load_mlicpp_mlclipsim(quality=0, device="cpu")


def test_training_checkpoint_can_resume_with_safe_loader(model, tmp_path):
    optimizer, auxiliary = configure_optimizers(model, 1e-4, 1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    train_step(model, torch.rand(1, 3, 64, 64), optimizer, auxiliary,
               objective="mse", lmbda=.01)
    scheduler.step()
    target = tmp_path / "last.pt"
    save_checkpoint(target, model=model, optimizer=optimizer, aux_optimizer=auxiliary,
                    scheduler=scheduler, epoch=1, best_loss=10.0,
                    args=SimpleNamespace(objective="mse", lmbda=.01))
    payload = torch.load(target, map_location="cpu", weights_only=True)
    assert payload["epoch"] == 1
    assert payload["optimizer"]["state"]
    assert payload["aux_optimizer"]["state"]
    model.load_state_dict(payload["lic_state"])
    optimizer.load_state_dict(payload["optimizer"])
    auxiliary.load_state_dict(payload["aux_optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    assert scheduler.last_epoch == 1
    assert not target.with_suffix(".pt.tmp").exists()
