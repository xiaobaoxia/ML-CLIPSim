from argparse import Namespace

import numpy as np
from PIL import Image
import pytest
import torch

pytest.importorskip('compressai')
from evaluation.common import reconstruct, Results


class IdentityCodec:
    def compress(self, tensor):
        self.tensor = tensor
        return dict(strings=[[b'12345'], [b'678']], shape=tensor.shape[-2:])

    def decompress(self, strings, shape):
        return dict(x_hat=self.tensor)


@pytest.mark.parametrize('multiple,mode', [(64, 'constant'), (64, 'replicate'), (128, 'constant')])
def test_coding_padding_preserves_original_pixels_and_rate_denominator(multiple, mode):
    pixels = np.arange(31 * 45 * 3, dtype=np.uint8).reshape(31, 45, 3)
    codec = IdentityCodec()
    decoded, rate = reconstruct(codec, Image.fromarray(pixels), 'cpu', multiple=multiple, mode=mode)
    torch.testing.assert_close(decoded, torch.from_numpy(pixels.copy()).permute(2, 0, 1).float() / 255)
    assert codec.tensor.shape[-2] % multiple == codec.tensor.shape[-1] % multiple == 0
    assert rate['bytes'] == 8
    assert rate['bpp'] == pytest.approx(64 / (31 * 45))


def test_resume_rejects_a_replaced_checkpoint(tmp_path):
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'first checkpoint')
    args = Namespace(output=tmp_path / 'results', checkpoint=checkpoint, weights_root=tmp_path,
                     quality=1, device='cpu', cdf='update', limit=1)
    results = Results(args)
    results.add(dict(id=1, bpp=.1, bytes=16))
    assert Results(args).done == {'1'}
    checkpoint.write_bytes(b'different checkpoint')
    with pytest.raises(ValueError, match='different run'):
        Results(args)


def test_coco_empty_detections_still_evaluate_ground_truth():
    pytest.importorskip('pycocotools')
    from pycocotools.coco import COCO
    from scripts.evaluate_coco import coco_ap
    coco = COCO()
    coco.dataset = dict(images=[dict(id=1, width=100, height=100)],
                        categories=[dict(id=1, name='person')],
                        annotations=[dict(id=1, image_id=1, category_id=1, bbox=[1, 1, 20, 20],
                                          area=400, iscrowd=0)])
    coco.createIndex()
    metrics = coco_ap(coco, [], [1], 'bbox')
    assert metrics == dict(AP=0, AP50=0, AP75=0)


def test_grouped_question_bitrate_uses_question_multiplicity(tmp_path):
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'checkpoint')
    args = Namespace(output=tmp_path / 'results', checkpoint=checkpoint, weights_root=tmp_path,
                     quality=1, device='cpu', cdf='update', limit=2)
    results = Results(args)
    results.add(dict(id=1, bpp=.1, weight=1))
    results.add(dict(id=2, bpp=.3, weight=3))
    assert results.save()['mean_bpp'] == pytest.approx(.25)
