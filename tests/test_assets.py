import json
from pathlib import Path


def test_released_asset_metadata_is_complete():
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "src/mlclipsim/weights/ML-CLIPSim_ViTB16.config.json").read_text()
    )
    assert config["format_version"] == 1
    assert config["backbone"] == "ViT-B/16"
    assert config["layers"] == list(range(12))
    assert config["tau_group"] == 2.0
    assert config["tau_gate"] == 2.0
    assert (root / "src/mlclipsim/weights/ML-CLIPSim_ViTB16.pt").stat().st_size > 0
