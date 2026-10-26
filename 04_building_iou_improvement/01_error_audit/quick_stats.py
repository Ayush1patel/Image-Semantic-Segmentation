"""
Summary statistics over building_error_per_image.csv (output of
audit_building_errors.py).

Reports how concentrated the Building error is (share of total error held
by the 20 worst images) and ranks images by per-image Building IoU, which
is a fairer "worst case" measure than raw error pixel count because it is
not dominated by images that simply contain a lot of Building.
"""
import csv
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "building_error_per_image.csv")

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

# Only images that actually contain Building ground truth are meaningful here.
has_building = [r for r in rows if int(r["building_gt_px"]) > 0]
print("total test images:", len(rows))
print("images with any Building gt pixels:", len(has_building))
errs = [int(r["total_error"]) for r in has_building]
print("mean total_error (images with building):", sum(errs) / len(errs))
print("median total_error:", sorted(errs)[len(errs) // 2])
top20_err = sum(sorted(errs, reverse=True)[:20])
print("top-20 share of all error:", top20_err, "/", sum(errs), "=", top20_err / sum(errs))

# Per-image IoU = TP / (TP + FP + FN); an empty union counts as a perfect score.
ious = []
for r in has_building:
    tp = int(r["tp"]); fp = int(r["fp"]); fn = int(r["fn"])
    union = tp + fp + fn
    iou = tp / union if union > 0 else 1.0
    ious.append((iou, r))
ious.sort(key=lambda x: x[0])
print()
print("Worst 15 by per-image Building IoU (not raw error count):")
for iou, r in ious[:15]:
    print(f"{iou:.3f}  {r['sequence']:<28} {r['stem']:<26} gt_px={r['building_gt_px']:>6} fp={r['fp']:>5} fn={r['fn']:>5}")
