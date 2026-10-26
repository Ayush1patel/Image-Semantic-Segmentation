"""
Scan every UAVScenes camera-label PNG (interval5_CAM_label_id) across ALL
sequences on disk and record a per-image histogram of raw class ids
(0-25).

Raw-id granularity is kept -- NOT pre-merged into the 19-class scheme
(unet_dice_ce/src/class_mapping.py) or the 9-class scheme
(../../data_utils.py) -- so this single scan backs both downstream
taxonomies without re-reading ~24k images twice. Re-run only if new
sequences are added or label files change.

Covers all 20 sequence folders found under LABEL_ROOT, including ones not
currently used by any train/val/test split (e.g. the *_GNSS03 sequences
and *_GNSS_Evening), because deciding a new spatial split is exactly the
point of this scan -- restricting it to the current split would beg the
question.

Output: 01_dataset_analysis/outputs/per_image_raw_class_counts.csv
  columns: sequence, stem, label_path, width, height, total_pixels,
           raw_0 .. raw_25

Usage:
    python 01_dataset_analysis/scripts/scan_class_pixel_distribution.py
"""
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import time

import numpy as np
from PIL import Image

LABEL_ROOT = r"D:\Saqib\Multimodal_segmentation\UAVScenes\interval5_CAM_label"
LABEL_ID_SUBDIR = "interval5_CAM_label_id"
NUM_RAW_CLASSES = 26

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
OUT_CSV = os.path.join(OUT_DIR, "per_image_raw_class_counts.csv")


def discover_sequences():
    return sorted(
        d for d in os.listdir(LABEL_ROOT)
        if os.path.isdir(os.path.join(LABEL_ROOT, d))
    )


def discover_label_files(seq):
    lbl_dir = os.path.join(LABEL_ROOT, seq, LABEL_ID_SUBDIR)
    if not os.path.isdir(lbl_dir):
        print(f"[WARN] missing label dir for {seq}: {lbl_dir}")
        return []
    files = sorted(f for f in os.listdir(lbl_dir) if f.lower().endswith(".png"))
    return [(seq, os.path.splitext(f)[0], os.path.join(lbl_dir, f)) for f in files]


def _process_one(item):
    seq, stem, label_path = item
    arr = np.array(Image.open(label_path))
    h, w = arr.shape[:2]
    flat = arr.ravel()
    counts = np.bincount(flat, minlength=NUM_RAW_CLASSES)[:NUM_RAW_CLASSES]
    over_range = int(flat.shape[0]) - int(counts.sum())
    return seq, stem, label_path, w, h, int(arr.size), counts.tolist(), over_range


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sequences = discover_sequences()
    print(f"Found {len(sequences)} sequences under {LABEL_ROOT}")

    all_items = []
    for seq in sequences:
        items = discover_label_files(seq)
        print(f"  {seq}: {len(items)} label files")
        all_items.extend(items)
    print(f"Total label files to scan: {len(all_items)}")

    rows = []
    total_over_range = 0
    n_done = 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = [ex.submit(_process_one, item) for item in all_items]
        for fut in as_completed(futures):
            seq, stem, label_path, w, h, total_px, counts, over_range = fut.result()
            if over_range:
                total_over_range += over_range
            rows.append((seq, stem, label_path, w, h, total_px, counts))
            n_done += 1
            if n_done % 1000 == 0 or n_done == len(futures):
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                print(f"  {n_done}/{len(futures)} scanned "
                      f"({elapsed:.1f}s elapsed, {rate:.1f} img/s)")

    if total_over_range:
        print(f"[WARN] {total_over_range} pixels across the dataset had a raw id "
              f">= {NUM_RAW_CLASSES} (outside the documented 0-25 range) -- "
              f"excluded from all counts. Investigate before trusting stats.")

    rows.sort(key=lambda r: (r[0], r[1]))

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["sequence", "stem", "label_path", "width", "height", "total_pixels"]
            + [f"raw_{i}" for i in range(NUM_RAW_CLASSES)]
        )
        for seq, stem, label_path, w, h, total_px, counts in rows:
            writer.writerow([seq, stem, label_path, w, h, total_px] + counts)

    print(f"Wrote {len(rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
