"""CLIP zero-shot ImageNet classification after real codec reconstruction."""
import json
from importlib.resources import files
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

from evaluation.common import arguments, load_codec, reconstruct, Results


@torch.inference_mode()
def text_weights(model, device):
    import clip
    data = json.loads(files('evaluation').joinpath('imagenet_prompts.json').read_text())
    weights = []
    for name in tqdm(data['imagenet_classes'], desc='CLIP text prompts'):
        prompts = [template.format(name.replace('_', ' ')) for template in data['imagenet_templates']]
        embeddings = model.encode_text(clip.tokenize(prompts).to(device)).float()
        weights.append(F.normalize(F.normalize(embeddings, dim=-1).mean(0), dim=0))
    return torch.stack(weights, dim=1)


def main():
    parser = arguments(__doc__)
    parser.add_argument('--data', type=Path, required=True, help='ImageNet root with the val split')
    parser.add_argument('--clip-checkpoint', help='optional local OpenAI ViT-B/16 checkpoint')
    parser.add_argument('--text-weights', type=Path, help='optional saved prompt embeddings')
    args = parser.parse_args()
    import clip
    torch.set_num_threads(2)
    transform = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                                    transforms.CenterCrop(224), transforms.ToTensor()])
    dataset = datasets.ImageNet(str(args.data), split='val', transform=transform)
    results = Results(args)
    codec = load_codec(args)
    model, _ = clip.load(args.clip_checkpoint or 'ViT-B/16', device=args.device, jit=False)
    model.eval().float()
    cache = args.text_weights or args.output / 'text_weights.pt'
    if cache.exists():
        weights = torch.load(cache, map_location=args.device, weights_only=True)
    else:
        weights = text_weights(model, args.device)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(weights.cpu(), cache)
    weights = weights.float()
    if tuple(weights.shape) != (model.visual.output_dim, 1000):
        raise ValueError('Text weights do not match ViT-B/16 and 1,000 ImageNet classes')
    mean = torch.tensor((.48145466, .4578275, .40821073), device=args.device).view(1, 3, 1, 1)
    std = torch.tensor((.26862954, .26130258, .27577711), device=args.device).view(1, 3, 1, 1)
    count = min(args.limit or len(dataset), len(dataset))
    with torch.inference_mode():
        for index in tqdm(range(count), desc='ImageNet'):
            if str(index) in results.done:
                continue
            image, target = dataset[index]
            decoded, rate = reconstruct(codec, image, args.device, multiple=64, mode='replicate')
            features = F.normalize(model.encode_image((decoded.unsqueeze(0) - mean) / std).float(), dim=-1)
            prediction = int((features @ weights).argmax(dim=1).item())
            results.add(dict(id=index, target=int(target), prediction=prediction,
                             correct=prediction == target, **rate))
    results.save(top1_percent=100 * sum(row['correct'] for row in results.rows) / len(results.rows))


if __name__ == '__main__':
    main()
