import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from class_mapping import REMAP_LUT

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data_index")


class UAVScenesDataset(Dataset):
    """Loads (image, merged-label) pairs from a prebuilt index json.

    Run scripts/build_index.py once before using this dataset.
    """

    def __init__(self, split: str, transform=None, index_dir: str = INDEX_DIR):
        assert split in ("train", "val", "test")
        index_path = os.path.join(index_dir, f"{split}.json")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(
                f"Index file not found: {index_path}. "
                f"Run scripts/build_index.py first."
            )
        with open(index_path) as f:
            self.samples = json.load(f)
        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = np.array(Image.open(sample["image"]).convert("RGB"))
        raw_label = np.array(Image.open(sample["label"]))
        label = REMAP_LUT[raw_label].astype(np.int64)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=label)
            image, label = augmented["image"], augmented["mask"]

        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)
        return image, label.long()
