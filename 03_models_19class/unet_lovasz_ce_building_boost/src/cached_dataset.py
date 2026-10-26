import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from class_mapping import REMAP_LUT

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data_index")


class CachedUAVScenesDataset(Dataset):
    """Same data as UAVScenesDataset, but decodes + resizes every image and
    label ONCE up front and holds them as plain uint8 arrays in RAM. Turns
    each epoch into pure augmentation + GPU compute, no disk I/O or JPEG
    decode, which is the dominant cost at small training resolutions.

    Only worth it when height/width are small enough that num_samples *
    (h*w*3 + h*w) bytes comfortably fits in RAM (checked by the caller).
    """

    def __init__(self, split: str, height: int, width: int, transform=None,
                 index_dir: str = INDEX_DIR):
        assert split in ("train", "val", "test")
        index_path = os.path.join(index_dir, f"{split}.json")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(
                f"Index file not found: {index_path}. Run scripts/build_index.py first."
            )
        with open(index_path) as f:
            samples = json.load(f)

        n = len(samples)
        self.images = np.empty((n, height, width, 3), dtype=np.uint8)
        self.labels = np.empty((n, height, width), dtype=np.uint8)

        for i, sample in enumerate(tqdm(samples, desc=f"Caching {split} in RAM")):
            img = cv2.imread(sample["image"], cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

            raw_label = cv2.imread(sample["label"], cv2.IMREAD_UNCHANGED)
            raw_label = cv2.resize(raw_label, (width, height), interpolation=cv2.INTER_NEAREST)
            label = REMAP_LUT[raw_label]

            self.images[i] = img
            self.labels[i] = label

        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx].astype(np.int64)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=label)
            image, label = augmented["image"], augmented["mask"]

        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)
        return image, label.long()
