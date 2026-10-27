"""Optional tests of all real released weights; never download assets in tests."""
import os
from pathlib import Path

import pytest

pytest.importorskip("compressai")
pytest.importorskip("einops")


@pytest.mark.parametrize("quality", [1, 2, 3, 4, 5])
def test_released_codec_round_trip(quality):
    root = os.environ.get("MLCLIPSIM_TEST_CODEC_WEIGHTS")
    if not root:
        pytest.skip("set MLCLIPSIM_TEST_CODEC_WEIGHTS to test the five real released checkpoints")
    import torch
    from codec.validate import verify_codec
    torch.set_num_threads(2)
    assert Path(root, f"mlicpp_ml_clipsim_q{quality}.pt").is_file()
    result = verify_codec(quality, weights_root=root, size=64)
    assert result["parameters"] == 116716928
    assert result["roundtrip_max_abs_error"] <= 1e-5
