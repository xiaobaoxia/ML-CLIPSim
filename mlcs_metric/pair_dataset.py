import os
import json
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PairMetricDataset(Dataset):
    """
    Pairwise dataset for metric learning.

    Features:
    - Controls ratio of neutral (mean_vote == 0.5) samples
    - Randomly swaps (p0, p1) with probability `swap_prob`
      to remove position bias (p0 always worse / p1 always better)
    """

    def __init__(
        self,
        label_json,
        dist_root,
        neutral_ratio=0,   # keep only this ratio of neutral (0.5) samples
        swap_prob=0.5,       # probability to swap (p0, p1)
        seed=42,
        image_size=224,
    ):
        super().__init__()

        # ---------- load labels ----------
        with open(label_json, "r") as f:
            labels = json.load(f)

        # ---------- split neutral / non-neutral ----------
        neutral = []
        non_neutral = []
        for x in labels:
            if abs(x["mean_vote"] - 0.5) < 1e-6:
                neutral.append(x)
            else:
                non_neutral.append(x)

        rng = random.Random(seed)
        rng.shuffle(neutral)

        keep_n = int(len(non_neutral) * neutral_ratio)
        neutral = neutral[:keep_n]

        self.labels = non_neutral + neutral
        rng.shuffle(self.labels)

        print(
            f"[Dataset] total={len(labels)} | "
            f"used={len(self.labels)} | "
            f"neutral_kept={len(neutral)} | "
            f"swap_prob={swap_prob}"
        )

        self.dist_root = dist_root
        self.swap_prob = swap_prob

        # ---------- image transform ----------
        self.tf = transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.labels)

    def _load_img(self, subdir, name):
        p = os.path.join(self.dist_root, subdir, f"{name}.png")
        with Image.open(p) as img:
            return self.tf(img.convert("RGB"))

    def __getitem__(self, idx):
        it = self.labels[idx]

        ref = self._load_img("ref", it["image"])
        p0  = self._load_img(it["p0"], it["image"])
        p1  = self._load_img(it["p1"], it["image"])

        # soft label: P(p0 better than p1)
        soft = torch.tensor(it["mean_vote"], dtype=torch.float32)

        # hard label (optional): 1 means p0 better than p1
        hard = torch.tensor(it.get("winner", float(it["mean_vote"] > 0.5)), dtype=torch.float32)

        # ---------- random swap to remove position bias ----------
        if random.random() < self.swap_prob:
            # swap images
            p0, p1 = p1, p0

            # swap labels
            soft = 1.0 - soft
            if hard is not None:
                hard = 1.0 - hard

        return ref, p0, p1, hard, soft
