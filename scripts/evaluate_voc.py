"""VOC2007 Faster R-CNN evaluation using Detectron2's Pascal VOC evaluator."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shutil

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from evaluation.common import arguments, load_codec, reconstruct, as_pil, Results


def main():
    parser = arguments(__doc__)
    parser.add_argument('--data', type=Path, required=True, help='VOC2007 directory')
    parser.add_argument('--split', default='test')
    parser.add_argument('--detector', help='optional local Faster R-CNN checkpoint')
    args = parser.parse_args()
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2.data.datasets.pascal_voc import CLASS_NAMES
    from detectron2.engine import DefaultPredictor
    from detectron2.evaluation import PascalVOCDetectionEvaluator
    from detectron2.structures import Boxes, Instances
    torch.set_num_threads(2)
    ids = (args.data / 'ImageSets/Main' / f'{args.split}.txt').read_text().split()
    ids = ids[:args.limit] if args.limit else ids
    results = Results(args)
    codec = load_codec(args)
    config_name = 'PascalVOC-Detection/faster_rcnn_R_50_C4.yaml'
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_name))
    cfg.MODEL.WEIGHTS = args.detector or model_zoo.get_checkpoint_url(config_name)
    cfg.MODEL.DEVICE = args.device
    detector = DefaultPredictor(cfg)
    with torch.inference_mode():
        for image_id in tqdm(ids, desc='VOC'):
            if image_id in results.done:
                continue
            with Image.open(args.data / 'JPEGImages' / f'{image_id}.jpg') as image:
                decoded, rate = reconstruct(codec, image, args.device, multiple=64)
            rgb = np.asarray(as_pil(decoded))
            instances = detector(rgb[:, :, ::-1].copy())['instances'].to('cpu')
            results.add(dict(id=image_id, boxes=instances.pred_boxes.tensor.tolist(),
                             scores=instances.scores.tolist(), classes=instances.pred_classes.tolist(), **rate))
    # Give the evaluator the selected ground-truth split, including for --limit.
    gt_root = args.output / 'ground_truth'
    (gt_root / 'Annotations').mkdir(parents=True, exist_ok=True)
    (gt_root / 'ImageSets/Main').mkdir(parents=True, exist_ok=True)
    (gt_root / 'ImageSets/Main/selected.txt').write_text('\n'.join(ids) + '\n')
    for image_id in ids:
        shutil.copy2(args.data / 'Annotations' / f'{image_id}.xml', gt_root / 'Annotations' / f'{image_id}.xml')
    name = 'codec_voc_' + str(args.output.resolve())
    MetadataCatalog.get(name).set(thing_classes=CLASS_NAMES, dirname=str(gt_root), split='selected', year=2007)
    evaluator = PascalVOCDetectionEvaluator(name)
    evaluator.reset()
    for row in results.rows:
        instances = Instances((row['height'], row['width']))
        instances.pred_boxes = Boxes(torch.tensor(row['boxes']).reshape(-1, 4))
        instances.scores = torch.tensor(row['scores'])
        instances.pred_classes = torch.tensor(row['classes'], dtype=torch.int64)
        evaluator.process([{'image_id': row['id']}], [{'instances': instances}])
    results.save(detection_percent=evaluator.evaluate())


if __name__ == '__main__':
    main()
