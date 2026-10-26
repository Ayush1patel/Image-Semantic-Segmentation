"""
Scan the UAVScenes CAM_label / CAM_LIDAR folders, build train/val/test
index files (image path <-> label path pairs), and compute per-class
pixel frequency statistics over the training split (used for class-
balanced loss weighting).

This project uses 19 trainable classes (all named UAVScenes classes kept
separate; only the 7 unnamed/reserved raw ids are folded into Background),
so train_class_stats.json will have 19 entries.

Usage:
    python scripts/build_index.py
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from class_mapping import NUM_CLASSES, REMAP_LUT  # noqa: E402
from splits import (IMAGE_ROOT, IMAGE_SUBDIR, LABEL_ID_SUBDIR, LABEL_ROOT,  # noqa: E402
                     SPLITS)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data_index")


def build_split_index(split_name, sequences):
    pairs = []
    for seq in sequences:
        img_dir = os.path.join(IMAGE_ROOT, seq, IMAGE_SUBDIR)
        lbl_dir = os.path.join(LABEL_ROOT, seq, LABEL_ID_SUBDIR)
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Missing image dir: {img_dir}")
        if not os.path.isdir(lbl_dir):
            raise FileNotFoundError(f"Missing label dir: {lbl_dir}")

        img_files = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir)
                     if f.lower().endswith(".jpg")}
        lbl_files = {os.path.splitext(f)[0]: f for f in os.listdir(lbl_dir)
                     if f.lower().endswith(".png")}

        common = sorted(set(img_files) & set(lbl_files))
        missing_img = sorted(set(lbl_files) - set(img_files))
        missing_lbl = sorted(set(img_files) - set(lbl_files))
        if missing_img or missing_lbl:
            print(f"  [WARN] {seq}: {len(missing_img)} labels w/o image, "
                  f"{len(missing_lbl)} images w/o label (skipped)")

        for stem in common:
            pairs.append({
                "image": os.path.join(img_dir, img_files[stem]),
                "label": os.path.join(lbl_dir, lbl_files[stem]),
                "sequence": seq,
                "stem": stem,
            })
        print(f"  {seq}: {len(common)} paired samples")

    print(f"[{split_name}] total: {len(pairs)} samples")
    return pairs


def compute_class_pixel_freq(pairs, sample_stride=1):
    """Sum per-class pixel counts (after remap) over a list of pairs."""
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for p in tqdm(pairs[::sample_stride], desc="Class pixel histogram"):
        raw = np.array(Image.open(p["label"]))
        merged = REMAP_LUT[raw]
        binc = np.bincount(merged.ravel(), minlength=NUM_CLASSES)
        counts[:len(binc)] += binc[:NUM_CLASSES]
    return counts


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_indices = {}
    for split_name, sequences in SPLITS.items():
        print(f"\n=== Building index: {split_name} ===")
        pairs = build_split_index(split_name, sequences)
        all_indices[split_name] = pairs
        out_path = os.path.join(OUT_DIR, f"{split_name}.json")
        with open(out_path, "w") as f:
            json.dump(pairs, f, indent=2)
        print(f"  -> wrote {out_path}")

    print("\n=== Computing per-class pixel frequency on TRAIN split ===")
    counts = compute_class_pixel_freq(all_indices["train"])
    freqs = counts / counts.sum()
    stats = {
        "class_pixel_counts": counts.tolist(),
        "class_pixel_freq": freqs.tolist(),
    }
    stats_path = os.path.join(OUT_DIR, "train_class_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  -> wrote {stats_path}")

    from class_mapping import NEW_CLASS_NAMES
    print("\nPer-class pixel frequency (train split):")
    for name, c, fr in zip(NEW_CLASS_NAMES, counts, freqs):
        print(f"  {name:18s} count={c:>14d}  freq={fr:.6f}")

    print("\nDone. Sample counts:", {k: len(v) for k, v in all_indices.items()})


if __name__ == "__main__":
    main()
