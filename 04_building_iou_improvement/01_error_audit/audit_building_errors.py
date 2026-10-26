r"""
Qualitative + quantitative error audit for the Building class, on the
CE+Lovasz UNet's test-set predictions (../../03_models_19class/unet_lovasz_ce,
checkpoints/best.pt -- the current best model). No training, read-only
scoring of an already-trained checkpoint against the held-out test split.

Purpose: before picking a fix (architecture, data, resolution, ...) for
pushing Building accuracy further, find out what the model's Building
errors actually look like. Building's confusion matrix is already clean
(only ~4-5% bleed into Background) -- this digs into the WORST individual
images to find the actual failure pattern (missed small buildings,
false-positive blobs on confusable roof-like classes, systematic
under/over-segmentation, etc.) rather than guessing.

Outputs (this folder):
  building_error_per_image.csv
      every test image: sequence, stem, building ground-truth pixel count,
      TP/FP/FN pixel counts, total error (FP+FN).
  images/worst_XX_<sequence>_<stem>.png
      the 20 highest-error images, each a 4-panel figure:
      Image | Ground Truth | Prediction | Building error overlay
      (green=correct Building, red=false positive, blue=false negative).
  audit_summary.md
      the worst-20 table + headline stats, written after this script runs.

Usage (run with the project's venv, not the base interpreter):
    & "E:\Multimodal\NamanAyush\\.venv\Scripts\python.exe" audit_building_errors.py
"""
import csv
import os
import sys

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
TOP_K = 20


@torch.no_grad()
def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    config_path = os.path.join(PROJECT, "configs", "config_256.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    checkpoint = os.path.join(PROJECT, "results_256", "checkpoints", "best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h, w = cfg["data"]["height"], cfg["data"]["width"]

    test_ds = UAVScenesDataset("test", transform=get_eval_transform(h, w), index_dir=os.path.join(PROJECT, "data_index"))
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0)

    model = build_model(encoder_name=cfg["model"]["encoder_name"], encoder_weights=None,
                         num_classes=len(NEW_CLASS_NAMES)).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {checkpoint} (epoch {ckpt.get('epoch')}, best_miou={ckpt.get('best_miou')})")

    rows = []
    sample_idx = 0
    for images, labels in tqdm(test_loader, desc="Scoring test set"):
        images_d = images.to(device, non_blocking=True)
        logits = model(images_d)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        labels_np = labels.numpy()

        for i in range(images.size(0)):
            gt = labels_np[i]
            pr = preds[i]
            gt_building = gt == BUILDING_IDX
            pr_building = pr == BUILDING_IDX
            tp = int(np.sum(gt_building & pr_building))
            fp = int(np.sum(~gt_building & pr_building))
            fn = int(np.sum(gt_building & ~pr_building))
            sample = test_ds.samples[sample_idx]
            rows.append({
                "sequence": sample["sequence"], "stem": sample["stem"],
                "index": sample_idx,
                "building_gt_px": int(gt_building.sum()),
                "tp": tp, "fp": fp, "fn": fn, "total_error": fp + fn,
            })
            sample_idx += 1

    csv_path = os.path.join(OUT_DIR, "building_error_per_image.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} images)")

    rows_sorted = sorted(rows, key=lambda r: r["total_error"], reverse=True)[:TOP_K]

    # Re-run inference just for the worst-K images to render visualizations
    # (cheap -- 20 single-image forward passes, not worth batching).
    print(f"Rendering {TOP_K} worst-error visualizations...")
    for rank, row in enumerate(tqdm(rows_sorted), start=1):
        image, label = test_ds[row["index"]]
        image_d = image.unsqueeze(0).to(device)
        logits = model(image_d)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy()
        gt = label.numpy()

        img_rgb = denormalize(image)
        gt_color = colorize(gt)
        pred_color = colorize(pred)

        gt_building = gt == BUILDING_IDX
        pred_building = pred == BUILDING_IDX
        overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)
        overlay[:, :] = (235, 235, 235)  # light gray background
        overlay[gt_building & pred_building] = (34, 139, 34)    # green: correct
        overlay[~gt_building & pred_building] = (220, 20, 60)   # red: false positive
        overlay[gt_building & ~pred_building] = (30, 90, 220)   # blue: false negative

        fig, axes = plt.subplots(1, 4, figsize=(22, 6))
        axes[0].imshow(img_rgb); axes[0].set_title("Image"); axes[0].axis("off")
        axes[1].imshow(gt_color); axes[1].set_title("Ground Truth (all classes)"); axes[1].axis("off")
        axes[2].imshow(pred_color); axes[2].set_title("Prediction (all classes)"); axes[2].axis("off")
        axes[3].imshow(overlay)
        axes[3].set_title("Building error: green=correct red=FP blue=FN")
        axes[3].axis("off")
        fig.suptitle(
            f"#{rank}  {row['sequence']} / {row['stem']}  "
            f"(gt_px={row['building_gt_px']}, fp={row['fp']}, fn={row['fn']})"
        )
        plt.tight_layout()
        out_name = f"worst_{rank:02d}_{row['sequence'].replace('interval5_', '')}_{row['stem']}.png"
        plt.savefig(os.path.join(IMAGES_DIR, out_name), dpi=110)
        plt.close(fig)

    print("Worst-20 summary:")
    print(f"{'rank':<5}{'sequence':<28}{'stem':<28}{'gt_px':>8}{'fp':>8}{'fn':>8}")
    for rank, row in enumerate(rows_sorted, start=1):
        print(f"{rank:<5}{row['sequence']:<28}{row['stem']:<28}{row['building_gt_px']:>8}"
              f"{row['fp']:>8}{row['fn']:>8}")


if __name__ == "__main__":
    main()
