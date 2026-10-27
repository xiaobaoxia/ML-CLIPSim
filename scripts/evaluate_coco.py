"""COCO YOLO, Grounding DINO, Mask R-CNN and CLIPSeg after codec reconstruction."""
from collections import defaultdict
import inspect
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from evaluation.common import arguments, load_codec, reconstruct, as_pil, Results


def coco_ap(coco, predictions, ids, kind):
    from pycocotools.cocoeval import COCOeval
    if predictions:
        detections = coco.loadRes(predictions)
    else:
        from pycocotools.coco import COCO
        detections = COCO()
        detections.dataset = dict(images=[coco.imgs[i] for i in ids],
                                  categories=coco.dataset['categories'], annotations=[])
        detections.createIndex()
    evaluator = COCOeval(coco, detections, kind)
    evaluator.params.imgIds = ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return dict(AP=100 * float(evaluator.stats[0]), AP50=100 * float(evaluator.stats[1]),
                AP75=100 * float(evaluator.stats[2]))


def main():
    parser = arguments(__doc__)
    parser.add_argument('--images', type=Path, required=True, help='COCO val2017 image directory')
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', choices=['yolo', 'groundingdino', 'maskrcnn', 'clipseg'], default=['yolo'])
    parser.add_argument('--yolo-weights', default='yolo26m.pt')
    parser.add_argument('--grounding-model', default='IDEA-Research/grounding-dino-base')
    parser.add_argument('--clipseg-model', default='CIDAS/clipseg-rd64-refined')
    parser.add_argument('--maskrcnn-weights', type=Path, help='optional local torchvision state dictionary')
    parser.add_argument('--det-threshold', type=float, default=.35)
    parser.add_argument('--maskrcnn-threshold', type=float, default=.3)
    args = parser.parse_args()
    from pycocotools.coco import COCO
    from pycocotools import mask as mask_utils
    torch.set_num_threads(2)
    coco = COCO(str(args.annotations))
    ids = coco.getImgIds()
    ids = ids[:args.limit] if args.limit else ids
    categories = {row['id']: row['name'] for row in coco.loadCats(coco.getCatIds())}
    name_to_id = {name: category for category, name in categories.items()}
    models = {}
    if 'yolo' in args.tasks:
        from ultralytics import YOLO
        models['yolo'] = YOLO(args.yolo_weights)
    if 'maskrcnn' in args.tasks:
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
        model = maskrcnn_resnet50_fpn_v2(weights=None if args.maskrcnn_weights else MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                                        weights_backbone=None)
        if args.maskrcnn_weights:
            model.load_state_dict(torch.load(args.maskrcnn_weights, map_location='cpu', weights_only=True))
        models['maskrcnn'] = model.eval().to(args.device)
    if 'clipseg' in args.tasks:
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        models['clipseg'] = (CLIPSegProcessor.from_pretrained(args.clipseg_model),
                             CLIPSegForImageSegmentation.from_pretrained(args.clipseg_model).eval().to(args.device))
    if 'groundingdino' in args.tasks:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        models['groundingdino'] = (AutoProcessor.from_pretrained(args.grounding_model),
                                   AutoModelForZeroShotObjectDetection.from_pretrained(args.grounding_model).eval().to(args.device))
    results = Results(args)
    codec = load_codec(args)
    with torch.inference_mode():
        for image_id in tqdm(ids, desc='COCO'):
            if str(image_id) in results.done:
                continue
            info = coco.imgs[image_id]
            with Image.open(args.images / info['file_name']) as source:
                decoded, rate = reconstruct(codec, source, args.device)
            image = as_pil(decoded)
            width, height = image.size
            row = dict(id=int(image_id), **rate)
            if 'yolo' in models:
                model = models['yolo']
                prediction = model.predict(np.asarray(image)[:, :, ::-1].copy(), imgsz=640, conf=.001,
                                           iou=.7, max_det=500, half=False, verbose=False,
                                           device=torch.device(args.device))[0]
                row['yolo'] = []
                for box, score, label in zip(prediction.boxes.xyxy.tolist(), prediction.boxes.conf.tolist(), prediction.boxes.cls.tolist()):
                    name = model.names[int(label)]
                    if name not in name_to_id:
                        continue
                    x1, y1, x2, y2 = box
                    row['yolo'].append(dict(image_id=int(image_id), category_id=name_to_id[name],
                                            bbox=[x1, y1, x2-x1, y2-y1], score=score))
            if 'maskrcnn' in models:
                prediction = models['maskrcnn']([to_tensor(image).to(args.device)])[0]
                row['maskrcnn'] = []
                for mask, score, label in zip(prediction['masks'], prediction['scores'], prediction['labels']):
                    if float(score) < args.maskrcnn_threshold or int(label) not in categories:
                        continue
                    binary = (mask[0].cpu().numpy() > .5).astype(np.uint8)
                    if not binary.any():
                        continue
                    rle = mask_utils.encode(np.asfortranarray(binary))
                    rle['counts'] = rle['counts'].decode('ascii')
                    row['maskrcnn'].append(dict(image_id=int(image_id), category_id=int(label),
                                                score=float(score), segmentation=rle))
            if 'clipseg' in models:
                processor, model = models['clipseg']
                annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=False))
                row['clipseg'] = []
                for category in sorted({ann['category_id'] for ann in annotations}):
                    gt = np.zeros((height, width), dtype=bool)
                    for ann in annotations:
                        if ann['category_id'] == category:
                            gt |= coco.annToMask(ann).astype(bool)
                    if not gt.any():
                        continue
                    inputs = processor(text=f'a {categories[category]}', images=image, return_tensors='pt').to(args.device)
                    logits = model(**inputs).logits
                    probability = F.interpolate(logits.unsqueeze(1), size=(height, width), mode='bilinear', align_corners=False)[0, 0].sigmoid()
                    mask = probability.cpu().numpy() > .5
                    iou = float(np.logical_and(mask, gt).sum() / np.logical_or(mask, gt).sum())
                    row['clipseg'].append(dict(category_id=int(category), iou=iou))
            if 'groundingdino' in models:
                processor, model = models['groundingdino']
                prompt = '. '.join(categories.values())
                inputs = processor(images=image, text=prompt, return_tensors='pt').to(args.device)
                prediction = model(**inputs)
                postprocess = processor.post_process_grounded_object_detection
                key = 'threshold' if 'threshold' in inspect.signature(postprocess).parameters else 'box_threshold'
                prediction = postprocess(prediction, inputs.input_ids, target_sizes=[(height, width)], **{key: args.det_threshold})[0]
                labels = prediction['text_labels'] if 'text_labels' in prediction else prediction['labels']
                row['groundingdino'] = []
                for box, score, label in zip(prediction['boxes'], prediction['scores'], labels):
                    name = label if isinstance(label, str) else list(categories.values())[int(label)]
                    if name not in name_to_id:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    row['groundingdino'].append(dict(image_id=int(image_id), category_id=name_to_id[name],
                                                     bbox=[x1, y1, x2-x1, y2-y1], score=float(score)))
            results.add(row)
    metrics = {}
    for task in args.tasks:
        if task == 'clipseg':
            scores = defaultdict(list)
            for row in results.rows:
                for value in row[task]:
                    scores[value['category_id']].append(value['iou'])
            metrics[task] = dict(mIoU_percent=100 * float(np.mean([np.mean(v) for v in scores.values()])))
        else:
            predictions = [prediction for row in results.rows for prediction in row[task]]
            (args.output / f'{task}_predictions.json').write_text(json.dumps(predictions))
            metrics[task] = coco_ap(coco, predictions, ids, 'segm' if task == 'maskrcnn' else 'bbox')
    results.save(tasks=metrics)


if __name__ == '__main__':
    main()
