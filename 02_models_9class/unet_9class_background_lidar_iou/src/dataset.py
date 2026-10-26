import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from class_mapping import NO_DATA_INDEX, REMAP_LUT
from lidar_projection import calibration_for_sequence, rasterize

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data_index")


class UAVScenesDataset(Dataset):
    """Loads (image, merged-label) pairs from a prebuilt index json.

    Run scripts/build_index.py once before using this dataset.

    With lidar_out_size=(h, w) each item additionally carries the frame's
    LiDAR footprint at that resolution:

        image, label, lidar_mask, lidar_label

    lidar_mask is True at pixels hit by at least one LiDAR return, and
    lidar_label holds that point's own class there (NO_DATA_INDEX elsewhere).
    Both are built by projecting the point cloud straight onto the h x w grid
    - see lidar_projection.py for why they are not rasterized-then-resized.

    Because the LiDAR rasters bypass `transform`, lidar_out_size is only
    valid with a transform whose sole geometric step is a resize to exactly
    (h, w) - i.e. get_eval_transform. It must not be used with the training
    augmentations, whose random crops/flips would desynchronize the rasters
    from the image. The returned label shape is checked against it per item.
    """

    def __init__(self, split: str, transform=None, index_dir: str = INDEX_DIR,
                 lidar_out_size=None):
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
        self.lidar_out_size = tuple(lidar_out_size) if lidar_out_size else None

    def __len__(self):
        return len(self.samples)

    def _load_lidar(self, sample, src_size):
        """-> (mask, lidar_label) rasters at self.lidar_out_size."""
        out_h, out_w = self.lidar_out_size
        if not sample.get("lidar") or not sample.get("lidar_label"):
            return (np.zeros((out_h, out_w), dtype=bool),
                    np.full((out_h, out_w), NO_DATA_INDEX, dtype=np.uint8))

        points = np.load(sample["lidar"])
        raw_point_labels = np.load(sample["lidar_label"])
        # Same raw-id -> class-id mapping as the camera labels, so the two
        # ground truths are directly comparable.
        point_labels = REMAP_LUT[np.asarray(raw_point_labels).astype(np.uint8)]
        return rasterize(
            points, point_labels,
            calib=calibration_for_sequence(sample["sequence"]),
            src_size=src_size, out_size=self.lidar_out_size,
            no_data_index=NO_DATA_INDEX,
        )

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pil_image = Image.open(sample["image"]).convert("RGB")
        src_size = (pil_image.height, pil_image.width)
        image = np.array(pil_image)
        raw_label = np.array(Image.open(sample["label"]))
        label = REMAP_LUT[raw_label].astype(np.int64)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=label)
            image, label = augmented["image"], augmented["mask"]

        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)
        label = label.long()

        if self.lidar_out_size is None:
            return image, label

        if tuple(label.shape[-2:]) != self.lidar_out_size:
            raise ValueError(
                f"lidar_out_size {self.lidar_out_size} does not match the "
                f"transformed label shape {tuple(label.shape[-2:])}; the LiDAR "
                f"rasters would not line up with the image."
            )
        mask, lidar_label = self._load_lidar(sample, src_size)
        return image, label, torch.from_numpy(mask), torch.from_numpy(lidar_label).long()
