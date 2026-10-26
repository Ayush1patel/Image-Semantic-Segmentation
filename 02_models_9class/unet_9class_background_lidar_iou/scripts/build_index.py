"""
Scan the UAVScenes CAM_label / CAM_LIDAR / LIDAR_label folders, build
train/val/test index files (image <-> label <-> LiDAR path tuples), and
compute per-class pixel frequency statistics over the training split (used
for class-balanced loss weighting).

Each index entry also carries the frame's LiDAR point cloud and per-point
label paths when they exist, so evaluation can restrict IoU to the pixels
the LiDAR actually hit. LiDAR files are named
image<cam_ts>_lidar<lidar_ts>.npy and are matched to a camera frame by
<cam_ts>; frames with no LiDAR counterpart get null paths and are simply
skipped by the LiDAR-masked metrics.

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
                     LIDAR_FILE_PREFIX, LIDAR_FILE_SEP, LIDAR_LABEL_ID_SUBDIR,
                     LIDAR_LABEL_ROOT, LIDAR_SUBDIR, SPLITS)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data_index")


def index_lidar_dir(lidar_dir):
    """Map camera-timestamp stem -> lidar filename for one sequence."""
    if not os.path.isdir(lidar_dir):
        return {}
    by_stem = {}
    for fname in os.listdir(lidar_dir):
        if not fname.endswith(".npy") or not fname.startswith(LIDAR_FILE_PREFIX):
            continue
        stem = fname[len(LIDAR_FILE_PREFIX):].split(LIDAR_FILE_SEP)[0]
        by_stem.setdefault(stem, fname)
    return by_stem


def build_split_index(split_name, sequences):
    pairs = []
    n_with_lidar = 0
    for seq in sequences:
        img_dir = os.path.join(IMAGE_ROOT, seq, IMAGE_SUBDIR)
        lbl_dir = os.path.join(LABEL_ROOT, seq, LABEL_ID_SUBDIR)
        lidar_dir = os.path.join(IMAGE_ROOT, seq, LIDAR_SUBDIR)
        lidar_lbl_dir = os.path.join(LIDAR_LABEL_ROOT, seq, LIDAR_LABEL_ID_SUBDIR)
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Missing image dir: {img_dir}")
        if not os.path.isdir(lbl_dir):
            raise FileNotFoundError(f"Missing label dir: {lbl_dir}")

        img_files = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir)
                     if f.lower().endswith(".jpg")}
        lbl_files = {os.path.splitext(f)[0]: f for f in os.listdir(lbl_dir)
                     if f.lower().endswith(".png")}
        lidar_files = index_lidar_dir(lidar_dir)
        lidar_lbl_files = index_lidar_dir(lidar_lbl_dir)

        common = sorted(set(img_files) & set(lbl_files))
        missing_img = sorted(set(lbl_files) - set(img_files))
        missing_lbl = sorted(set(img_files) - set(lbl_files))
        if missing_img or missing_lbl:
            print(f"  [WARN] {seq}: {len(missing_img)} labels w/o image, "
                  f"{len(missing_lbl)} images w/o label (skipped)")

        seq_lidar = 0
        for stem in common:
            lidar_name = lidar_files.get(stem)
            lidar_lbl_name = lidar_lbl_files.get(stem)
            has_lidar = lidar_name is not None and lidar_lbl_name is not None
            seq_lidar += has_lidar
            pairs.append({
                "image": os.path.join(img_dir, img_files[stem]),
                "label": os.path.join(lbl_dir, lbl_files[stem]),
                "lidar": os.path.join(lidar_dir, lidar_name) if lidar_name else None,
                "lidar_label": (os.path.join(lidar_lbl_dir, lidar_lbl_name)
                                if lidar_lbl_name else None),
                "sequence": seq,
                "stem": stem,
            })
        n_with_lidar += seq_lidar
        note = "" if seq_lidar == len(common) else f"  [{len(common) - seq_lidar} without LiDAR]"
        print(f"  {seq}: {len(common)} paired samples{note}")

    print(f"[{split_name}] total: {len(pairs)} samples, {n_with_lidar} with LiDAR")
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
