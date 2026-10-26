"""
Counts how many test images got better, worse or stayed the same after
Building mask post-processing, using building_postprocess_comparison.csv
(output of postprocess_building_masks.py).

An image counts as "improved" when its Building error (FP + FN pixels)
decreased after post-processing.
"""
import csv
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "building_postprocess_comparison.csv")

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

improved, worsened, same = 0, 0, 0
for r in rows:
    raw_err = int(r["raw_fp"]) + int(r["raw_fn"])
    post_err = int(r["post_fp"]) + int(r["post_fn"])
    if post_err < raw_err:
        improved += 1
    elif post_err > raw_err:
        worsened += 1
    else:
        same += 1
print(f"images with any Building gt or pred activity compared: {len(rows)}")
print(f"improved: {improved}  worsened: {worsened}  unchanged: {same}")
