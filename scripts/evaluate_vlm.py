"""POPE/SEEDBench codec evaluation through VLMEvalKit's native prompts and scorers."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vlmeval-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, help='optional VLMEvalKit dataset/cache directory')
    parser.add_argument('--dataset', choices=['POPE', 'SEEDBench_IMG'], required=True)
    parser.add_argument('--model', choices=['InternVL3-1B', 'Qwen2.5-VL-3B-Instruct'], required=True)
    parser.add_argument('--model-path', help='optional local pretrained VLM directory')
    parser.add_argument('--quality', type=int, default=1)
    parser.add_argument('--weights-root', type=Path, default=Path('weights/codec'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--cdf', choices=['update', 'saved'], default='update')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--max-new-tokens', type=int, help='optional generation limit')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = visible.split(',') if visible else None
    if devices and not 0 <= args.gpu < len(devices):
        parser.error('--gpu is outside the visible GPU list')
    os.environ['CUDA_VISIBLE_DEVICES'] = devices[args.gpu] if devices else str(args.gpu)
    if args.data_root:
        os.environ['LMUData'] = str(args.data_root.resolve())
    sys.path.insert(0, str(args.vlmeval_root.resolve()))
    import copy
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    from vlmeval.config import supported_VLM
    from vlmeval.dataset import build_dataset
    from vlmeval.smp import dump
    from evaluation.common import Results, load_codec, reconstruct, as_pil, file_hash
    args.device = 'cuda:0'
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = build_dataset(args.dataset)
    selected = dataset.data.iloc[:args.limit].copy() if args.limit else dataset.data.copy()
    results = Results(args)
    codec = load_codec(args)
    options = {'model_path': args.model_path} if args.model_path else {}
    if args.max_new_tokens:
        options['max_new_tokens'] = args.max_new_tokens
    model = supported_VLM[args.model](**options)
    model.set_dump_image(dataset.dump_image)
    cache = {}
    image_dir = args.output / 'images'
    image_dir.mkdir(exist_ok=True)
    with torch.inference_mode():
        for _, row in tqdm(selected.iterrows(), total=len(selected), desc=args.dataset):
            index = str(row['index'])
            if index in results.done:
                continue
            if getattr(dataset, 'force_use_dataset_prompt', False):
                message = dataset.build_prompt(row)
            elif hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(args.dataset):
                message = model.build_prompt(row, dataset=args.dataset)
            else:
                message = dataset.build_prompt(row)
            message = copy.deepcopy(message)
            rates = []
            for item in message:
                if item['type'] != 'image':
                    continue
                path = Path(item['value'])
                identity = file_hash(path)
                if identity not in cache:
                    with Image.open(path) as image:
                        decoded, rate = reconstruct(codec, image, args.device, numpy_division=True)
                    target = image_dir / f'{identity}.png'
                    as_pil(decoded).save(target)
                    cache[identity] = (target, rate)
                target, rate = cache[identity]
                item['value'] = str(target.resolve())
                rates.append(rate)
            if not rates:
                raise ValueError('Benchmark prompt contains no image')
            prediction = model.generate(message=message, dataset=args.dataset)
            weight = 1
            if args.dataset == 'POPE':
                # POPE scores expand comma-separated category memberships.
                categories = row.get('category', '')
                weight = len(categories) if isinstance(categories, (list, tuple)) else len(str(categories).split(','))
            results.add(dict(id=index, prediction=str(prediction), weight=weight,
                             bpp=sum(rate['bpp'] for rate in rates) / len(rates),
                             bytes=sum(rate['bytes'] for rate in rates)))
    by_id = {row['id']: row['prediction'] for row in results.rows}
    selected['prediction'] = [by_id[str(index)] for index in selected['index']]
    selected = selected.drop(columns=['image'], errors='ignore')
    predictions = args.output / f'{args.model}_{args.dataset}.xlsx'
    dump(selected, str(predictions))
    scores = dataset.evaluate(str(predictions), model='exact_matching', nproc=1)
    if hasattr(scores, 'to_dict'):
        scores = json.loads(scores.to_json(orient='records'))
    results.save(benchmark=args.dataset, model=args.model, scores=scores,
                 question_count=sum(row.get('weight', 1) for row in results.rows))


if __name__ == '__main__':
    main()
