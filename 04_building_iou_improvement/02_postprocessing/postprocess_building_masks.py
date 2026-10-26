r"""
Phase 1 -- post-processing experiment on the EXISTING CE+Lovasz checkpoint's
Building predictions. No training, no model changes: this only asks "how
much of the remaining Building error can cheap, inference-time mask
cleanup recover?" before spending a ~4hr retrain on Phase 2.

Pipeline per predicted Building mask:
  1. Morphological opening then closing (3x3 kernel) -- strips small
     false-positive speckles, fills small false-negative holes.
  2. Connected-component area filter (min_area=30px) -- drops tiny blobs,
     consistent with the audit's finding that sub-30px Building regions
     are mostly annotation noise, not real structure.
  3. Contour simplification (cv2.approxPolyDP, epsilon=1% of perimeter)
     -- turns the jagged pixel-level boundary into a small-vertex-count
     polygon, then rasterizes it back to a mask for a fair pixel-level
     IoU comparison against the raw prediction.

Outputs (this folder):
  building_postprocess_comparison.csv
      every test image: raw (tp,fp,fn) vs post-processed (tp,fp,fn) for
      the Building class.
  images/before_after_XX_<sequence>_<stem>.png
      the same 20 worst-case images from ../01_error_audit/audit_summary.md, now showing
      raw vs. post-processed Building error side by side.
  phase1_summary.md
      headline before/after IoU/precision/recall + reading, written after
      this script runs.

Usage (training venv, not the base interpreter):
    & "E:\Multimodal\NamanAyush\\.venv\Scripts\python.exe" postprocess_building_masks.py
"""
import csv
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT = os.path.join(os.path.dirname(__file__), "..", "..", "03_models_19class", "unet_lovasz_ce")
sys.path.insert(0, os.path.join(PROJECT, "src"))

from class_mapping import NEW_CLASS_NAMES, colorize  # noqa: E402
from dataset import UAVScenesDataset  # noqa: E402
from model import build_model  # noqa: E402
from transforms import denormalize, get_eval_transform  # noqa: E402

OUT_DIR = os.path.dirname(__file__)
IMAGES_DIR = os.path.join(OUT_DIR, "images")
BUILDING_IDX = NEW_CLASS_NAMES.index("Building")

MIN_AREA = 30
EPSILON_RATIO = 0.01
MORPH_KERNEL = 3

# Same 20 worst-by-raw-error images as the original audit, for direct
# before/after comparability -- (sequence, stem) pairs, hand-copied from
# ../01_error_audit/audit_summary.md's source data (../01_error_audit/building_error_per_image.csv).
WORST_20 = [
    ("interval5_HKisland03", "1669702867.899781942"),
    ("interval5_AMtown03", "1658132106.637086680"),
    ("interval5_AMtown03", "1658131898.147640013"),
    ("interval5_HKisland03", "1669702896.899803877"),
    ("interval5_AMtown03", "1658132110.139440595"),
    ("interval5_AMtown03", "1658132370.627472504"),
    ("interval5_AMtown03", "1658131906.147291445"),
    ("interval5_AMtown03", "1658132371.129440994"),
    ("interval5_AMtown03", "1658132249.631514739"),
    ("interval5_AMtown03", "1658132107.140170715"),
    ("interval5_AMtown03", "1658132110.638159539"),
    ("interval5_AMtown03", "1658132250.133362730"),
    ("interval5_HKisland03", "1669702887.399913073"),
    ("interval5_AMtown03", "1658132108.141117443"),
    ("interval5_AMtown03", "1658131960.144448554"),
    ("interval5_AMtown03", "1658132114.140213705"),
    ("interval5_AMtown03", "1658132255.133536131"),
    ("interval5_AMtown03", "1658132250.631183520"),
    ("interval5_AMtown03", "1658131962.644038919"),
    ("interval5_AMtown03", "1658132249.133634473"),
]


def postprocess_mask(mask_bool):
    mask_u8 = (mask_bool.astype(np.uint8)) * 255
    kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask_u8)
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, EPSILON_RATIO * peri, True)
        cv2.fillPoly(out, [approx], 255)
    return out > 0


def error_overlay(gt_building, pred_building):
    overlay = np.zeros((*gt_building.shape, 3), dtype=np.uint8)
    overlay[:, :] = (235, 235, 235)
    overlay[gt_building & pred_building] = (34, 139, 34)
    overlay[~gt_building & pred_building] = (220, 20, 60)
    overlay[gt_building & ~pred_building] = (30, 90, 220)
    return overlay


@torch.no_grad()
def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    config_path = os.path.join(PROJECT, "configs", "config_256.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    checkpoint = os.path.join(PROJECT, "results_256", "checkpoints", "best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h, w = cfg["data"]["height"], cfg["data"]["width"]

    test_ds = UAVScenesDataset("test", transform=get_eval_transform(h, w),
                                index_dir=os.path.join(PROJECT, "data_index"))
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0)

    model = build_model(encoder_name=cfg["model"]["encoder_name"], encoder_weights=None,
                         num_classes=len(NEW_CLASS_NAMES)).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {checkpoint} (epoch {ckpt.get('epoch')}, best_miou={ckpt.get('best_miou')})")

    rows = []
    sample_idx = 0
    for images, labels in tqdm(test_loader, desc="Scoring raw + post-processed"):
        images_d = images.to(device, non_blocking=True)
        logits = model(images_d)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        labels_np = labels.numpy()

        for i in range(images.size(0)):
            gt_building = labels_np[i] == BUILDING_IDX
            raw_building = preds[i] == BUILDING_IDX
            post_building = postprocess_mask(raw_building)

            def stats(pred_mask):
                tp = int(np.sum(gt_building & pred_mask))
                fp = int(np.sum(~gt_building & pred_mask))
                fn = int(np.sum(gt_building & ~pred_mask))
                return tp, fp, fn

            raw_tp, raw_fp, raw_fn = stats(raw_building)
            post_tp, post_fp, post_fn = stats(post_building)

            sample = test_ds.samples[sample_idx]
            rows.append({
                "sequence": sample["sequence"], "stem": sample["stem"], "index": sample_idx,
                "raw_tp": raw_tp, "raw_fp": raw_fp, "raw_fn": raw_fn,
                "post_tp": post_tp, "post_fp": post_fp, "post_fn": post_fn,
            })
            sample_idx += 1

    csv_path = os.path.join(OUT_DIR, "building_postprocess_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} images)")

    def agg_metrics(tp_key, fp_key, fn_key):
        tp = sum(r[tp_key] for r in rows)
        fp = sum(r[fp_key] for r in rows)
        fn = sum(r[fn_key] for r in rows)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return iou, prec, rec

    raw_iou, raw_prec, raw_rec = agg_metrics("raw_tp", "raw_fp", "raw_fn")
    post_iou, post_prec, post_rec = agg_metrics("post_tp", "post_fp", "post_fn")
    print(f"\nBuilding IoU:       raw={raw_iou:.4f}  post-processed={post_iou:.4f}  delta={post_iou - raw_iou:+.4f}")
    print(f"Building Precision: raw={raw_prec:.4f}  post-processed={post_prec:.4f}  delta={post_prec - raw_prec:+.4f}")
    print(f"Building Recall:    raw={raw_rec:.4f}  post-processed={post_rec:.4f}  delta={post_rec - raw_rec:+.4f}")

    # ---- Before/after visuals for the same worst-20 images as the audit ----
    index_by_key = {(r["sequence"], r["stem"]): r["index"] for r in rows}
    print(f"\nRendering {len(WORST_20)} before/after visualizations...")
    for rank, (seq, stem) in enumerate(tqdm(WORST_20), start=1):
        idx = index_by_key.get((seq, stem))
        if idx is None:
            print(f"  [WARN] {seq}/{stem} not found, skipping")
            continue
        image, label = test_ds[idx]
        image_d = image.unsqueeze(0).to(device)
        logits = model(image_d)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy()
        gt = label.numpy()

        img_rgb = denormalize(image)
        gt_color = colorize(gt)
        gt_building = gt == BUILDING_IDX
        raw_building = pred == BUILDING_IDX
        post_building = postprocess_mask(raw_building)

        raw_overlay = error_overlay(gt_building, raw_building)
        post_overlay = error_overlay(gt_building, post_building)

        row = rows[idx]
        raw_err = row["raw_fp"] + row["raw_fn"]
        post_err = row["post_fp"] + row["post_fn"]

        fig, axes = plt.subplots(1, 4, figsize=(22, 6))
        axes[0].imshow(img_rgb); axes[0].set_title("Image"); axes[0].axis("off")
        axes[1].imshow(gt_color); axes[1].set_title("Ground Truth (all classes)"); axes[1].axis("off")
        axes[2].imshow(raw_overlay); axes[2].set_title(f"Raw prediction (error={raw_err})"); axes[2].axis("off")
        axes[3].imshow(post_overlay); axes[3].set_title(f"Post-processed (error={post_err})"); axes[3].axis("off")
        fig.suptitle(f"#{rank}  {seq} / {stem}  (green=correct red=FP blue=FN)")
        plt.tight_layout()
        out_name = f"before_after_{rank:02d}_{seq.replace('interval5_', '')}_{stem}.png"
        plt.savefig(os.path.join(IMAGES_DIR, out_name), dpi=110)
        plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()
