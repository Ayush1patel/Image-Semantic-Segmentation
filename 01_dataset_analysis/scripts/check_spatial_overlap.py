"""
Quantify actual ground-distance overlap between the 01 (train), 02 (val),
and 03 (test) flights at each site, using each sequence's own
rtk_positions_raw.csv (absolute UTM easting/northing, NOT the per-sequence-
relative T4x4 pose -- that one can't be safely compared across sequences
without first confirming a shared origin, which we haven't).

For each image (identified by its filename timestamp, which IS the
capture time), find the RTK row with the closest headerstamp in that same
sequence's log and use its (easting, northing) as the image's ground
position. Then, per site: build a KD-tree of the train (01) sequence's
image positions, and for every val (02) / test (03) image find the
distance to the nearest train image. Also compute the median consecutive-
frame spacing WITHIN the train sequence itself, as a reference scale for
"how close is close" -- if cross-sequence nearest-neighbor distances are
in the same ballpark as the within-sequence frame spacing, the flights are
retracing the same ground, not just the same general area.

Usage:
    python 01_dataset_analysis/scripts/check_spatial_overlap.py
"""
import csv
import os

import numpy as np
from scipy.spatial import cKDTree

CAM_LIDAR_ROOT = r"D:\Saqib\Multimodal_segmentation\UAVScenes\interval5_CAM_LIDAR"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
COUNTS_CSV = os.path.join(OUT_DIR, "per_image_raw_class_counts.csv")

SITES = {
    "AMtown": ["interval5_AMtown01", "interval5_AMtown02", "interval5_AMtown03"],
    "AMvalley": ["interval5_AMvalley01", "interval5_AMvalley02", "interval5_AMvalley03"],
    "HKairport": ["interval5_HKairport01", "interval5_HKairport02", "interval5_HKairport03"],
    "HKisland": ["interval5_HKisland01", "interval5_HKisland02", "interval5_HKisland03"],
}


def load_image_stems():
    """sequence -> sorted list of float timestamps (from stems already scanned)."""
    seq_to_stems = {}
    with open(COUNTS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            seq_to_stems.setdefault(r["sequence"], []).append(float(r["stem"]))
    return seq_to_stems


def load_rtk_track(seq):
    """Return sorted (headerstamp[N], easting[N], northing[N]) arrays."""
    path = os.path.join(CAM_LIDAR_ROOT, seq, "rtk_positions_raw.csv")
    ts, e, n = [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ts.append(float(r["headerstamp"]))
            e.append(float(r["easting"]))
            n.append(float(r["northing"]))
    order = np.argsort(ts)
    return np.array(ts)[order], np.array(e)[order], np.array(n)[order]


def positions_for_images(image_ts, rtk_ts, rtk_e, rtk_n):
    """Nearest-timestamp match: for each image timestamp, the RTK (e, n) of
    the closest-in-time RTK fix in the same sequence's log."""
    idx = np.searchsorted(rtk_ts, image_ts)
    idx = np.clip(idx, 1, len(rtk_ts) - 1)
    left = idx - 1
    choose_left = np.abs(rtk_ts[left] - image_ts) <= np.abs(rtk_ts[idx] - image_ts)
    final_idx = np.where(choose_left, left, idx)
    dt = np.abs(rtk_ts[final_idx] - image_ts)
    return rtk_e[final_idx], rtk_n[final_idx], dt


def consecutive_spacing(e, n):
    order = np.argsort(np.arange(len(e)))  # already time-ordered by construction
    d = np.hypot(np.diff(e[order]), np.diff(n[order]))
    return d


def main():
    seq_to_stems = load_image_stems()

    for site, (train_seq, val_seq, test_seq) in SITES.items():
        print(f"\n{'=' * 70}\n{site}: train={train_seq}  val={val_seq}  test={test_seq}\n{'=' * 70}")

        positions = {}
        max_dt = 0.0
        for seq in (train_seq, val_seq, test_seq):
            rtk_ts, rtk_e, rtk_n = load_rtk_track(seq)
            img_ts = np.array(sorted(seq_to_stems[seq]))
            e, n, dt = positions_for_images(img_ts, rtk_ts, rtk_e, rtk_n)
            positions[seq] = (e, n)
            max_dt = max(max_dt, dt.max())
            print(f"  {seq}: {len(img_ts)} images, RTK match dt max={dt.max():.3f}s median={np.median(dt):.3f}s")

        train_e, train_n = positions[train_seq]
        train_xy = np.stack([train_e, train_n], axis=1)
        tree = cKDTree(train_xy)

        train_spacing = consecutive_spacing(train_e, train_n)
        print(f"  Reference: consecutive-frame spacing WITHIN {train_seq} "
              f"(train's own flight path resolution): "
              f"median={np.median(train_spacing):.2f}m  p90={np.percentile(train_spacing, 90):.2f}m")

        for seq in (val_seq, test_seq):
            e, n = positions[seq]
            xy = np.stack([e, n], axis=1)
            dist, _ = tree.query(xy, k=1)
            print(f"\n  {seq} -> nearest {train_seq} frame, distance distribution:")
            print(f"    min={dist.min():.2f}m  p5={np.percentile(dist, 5):.2f}m  "
                  f"median={np.median(dist):.2f}m  p95={np.percentile(dist, 95):.2f}m  max={dist.max():.2f}m")
            for thresh in (5, 10, 25, 50, 100):
                frac = float((dist < thresh).mean())
                print(f"    fraction of {seq} images within {thresh:>3d}m of a {train_seq} image: {frac * 100:5.1f}%")


if __name__ == "__main__":
    main()
