"""
Sanity check for the Building-aware WeightedRandomSampler used in train.py.

Rebuilds the per-image sampling weights (1 + alpha * building_fraction,
clipped at clip_max) from the cached training labels and prints their
distribution, so the sampler settings can be checked without training:
how many images contain no Building, how many hit the clip, and how many
samples one epoch draws for the chosen epoch_length_factor.

alpha, clip_max and the 1.3 epoch factor must match configs/config_256.yaml.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cached_dataset import CachedUAVScenesDataset  # noqa: E402
from class_mapping import NEW_CLASS_NAMES  # noqa: E402
from transforms import get_train_transform_cached  # noqa: E402

train_ds = CachedUAVScenesDataset("train", 256, 256, transform=get_train_transform_cached(),
                                   index_dir=os.path.join(os.path.dirname(__file__), "..", "data_index"))

building_idx = NEW_CLASS_NAMES.index("Building")
building_frac = (train_ds.labels == building_idx).reshape(len(train_ds), -1).mean(axis=1)

alpha = 6.0
clip_max = 2.5
weights = np.clip(1.0 + alpha * building_frac, None, clip_max)

print(f"n images: {len(train_ds)}")
print(f"building_frac: mean={building_frac.mean():.4f} median={np.median(building_frac):.4f} "
      f"p90={np.percentile(building_frac, 90):.4f} max={building_frac.max():.4f}")
print(f"sample weights: mean={weights.mean():.3f} median={np.median(weights):.3f} "
      f"p90={np.percentile(weights, 90):.3f} max={weights.max():.3f}")
print(f"fraction of images with zero Building: {(building_frac == 0).mean():.3f}")
print(f"fraction of images clipped at clip_max: {(weights >= clip_max).mean():.4f}")
n_epoch = int(len(train_ds) * 1.3)
print(f"with epoch_length_factor=1.3: {n_epoch} samples/epoch vs {len(train_ds)} images "
      f"({n_epoch/len(train_ds):.2f}x)")
