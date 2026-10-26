"""
Per-class pixel frequency and image-presence count on the val split.
CPU-only, read-only, no model/checkpoint needed -- just scans the label
PNGs listed in data_index/val.json through the same REMAP_LUT the
training pipeline uses. Safe to run alongside active training (no GPU use).

Purpose: confirms every trainable class has real support in val, i.e.
none get silently dropped from the mIoU's nanmean-over-present-classes
(which would otherwise inflate the reported mean).

Usage (from the project root):
    python analysis/scripts/check_val_class_distribution.py
    python analysis/scripts/check_val_class_distribution.py --split train
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from class_mapping import NUM_CLASSES, NEW_CLASS_NAMES, REMAP_LUT  # noqa: E402


def main(split: str):
    index_path = os.path.join(ROOT, "data_index", f"{split}.json")
    with open(index_path) as f:
        pairs = json.load(f)

    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    images_with_class = np.zeros(NUM_CLASSES, dtype=np.int64)
    for p in pairs:
        raw = np.array(Image.open(p["label"]))
        merged = REMAP_LUT[raw]
        binc = np.bincount(merged.ravel(), minlength=NUM_CLASSES)
        counts += binc[:NUM_CLASSES]
        images_with_class += (binc[:NUM_CLASSES] > 0).astype(np.int64)

    total = counts.sum()
    n = len(pairs)
    print(f"{'Class':<20}{'pixel_freq':>12}{'#images containing it':>28} (of {n})")
    for name, c, cnt in zip(NEW_CLASS_NAMES, counts, images_with_class):
        print(f"{name:<20}{c/total:>12.6f}{cnt:>28d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()
    main(args.split)
