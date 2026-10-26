"""
Test-set evaluation.

Reports three views of the same predictions, all at the config's eval
resolution and all using the same 9-class taxonomy:

  1. FULL FRAME      - every pixel, scored against the camera label PNG.
                       This is the unchanged, headline metric.
  2. LIDAR PIXELS    - only the pixels that a LiDAR return projects onto,
                       still scored against the camera label. Same model,
                       same ground truth, restricted support - so it is
                       directly comparable to (1) and isolates how the model
                       does where the point cloud actually has coverage.
  3. LIDAR POINTS    - the same pixels, but scored against the LiDAR point's
                       own label from interval5_LIDAR_label instead of the
                       camera label. This is the LiDAR modality's own ground
                       truth.

(2) is the like-for-like comparison against (1): same model, same ground
truth, smaller support. Read (3) with care - the two ground truths do not
agree with each other. Measured over the masked pixels of a 41-frame sample
of the test split, camera label vs LiDAR point label scores 95.1% pixel
agreement but only 0.69 mIoU between them. Rare classes are where they
diverge, so (3) carries a hard per-class ceiling well under 1.0 that has
nothing to do with the model; it is a second opinion, not a stricter
version of (2).

See lidar_projection.py for the projection geometry and its validation.
"""
import argparse
import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from class_mapping import (IGNORE_INDEX, NEW_CLASS_NAMES, NO_DATA_INDEX,
                            NUM_CLASSES, colorize)
from dataset import UAVScenesDataset
from losses import DiceCELoss, compute_class_weights
from metrics import ConfusionMatrix
from model import build_model
from transforms import denormalize, get_eval_transform

ROOT = os.path.join(os.path.dirname(__file__), "..")


def save_confusion_matrix(cm, title, path):
    """Row-normalized confusion matrix heatmap."""
    mat = cm.mat.astype(np.float64)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm_mat = mat / row_sums

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm_mat, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(NEW_CLASS_NAMES, rotation=90)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(NEW_CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
    ax.set_title(title)
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def save_vis(img_rgb, gt_color, pred_color, lidar_gt_color, lidar_pred_color, path):
    """Image | GT | Pred on top, LiDAR-restricted GT | Pred underneath."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    panels = [
        (img_rgb, "Image"),
        (gt_color, "Ground Truth (camera)"),
        (pred_color, "Prediction (full frame)"),
        (None, None),
        (lidar_gt_color, "Ground Truth @ LiDAR pixels"),
        (lidar_pred_color, "Prediction @ LiDAR pixels"),
    ]
    for ax, (arr, title) in zip(axes.ravel(), panels):
        ax.axis("off")
        if arr is None:
            continue
        ax.imshow(arr)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def main(checkpoint: str, config: str, num_vis: int, out_dir: str = None):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    results_dir = os.path.join(ROOT, cfg["train"]["results_dir"])
    if checkpoint is None:
        checkpoint = os.path.join(results_dir, "checkpoints", "best.pt")
    if out_dir is None:
        out_dir = os.path.join(results_dir, "test_eval")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h, w = cfg["data"]["height"], cfg["data"]["width"]

    # lidar_out_size makes each item carry its LiDAR footprint at (h, w).
    test_ds = UAVScenesDataset("test", transform=get_eval_transform(h, w),
                                lidar_out_size=(h, w))
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"],
                              shuffle=False, num_workers=cfg["data"]["num_workers"])

    model = build_model(encoder_name=cfg["model"]["encoder_name"],
                         encoder_weights=None, num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint {checkpoint} (epoch {ckpt.get('epoch')}, "
          f"best_miou={ckpt.get('best_miou')})")

    # Same loss as training (class-weighted Dice + CE), so "test loss" is
    # directly comparable to the train/val loss curves.
    stats_path = os.path.join(ROOT, "data_index", "train_class_stats.json")
    class_weights = None
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        class_weights = compute_class_weights(
            stats["class_pixel_freq"], NUM_CLASSES,
            clip_max=cfg["train"]["class_weight_clip_max"],
        ).to(device)
    criterion = DiceCELoss(
        NUM_CLASSES, class_weights=class_weights,
        dice_weight=cfg["train"]["dice_weight"], ce_weight=cfg["train"]["ce_weight"],
        ignore_index=IGNORE_INDEX,
    )

    # Fresh output folder every run - wipe any stale images/results from a
    # previous checkpoint or a previous num_vis setting.
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # Spread visualized samples evenly across the whole test set (index order
    # is sequential flight frames, so the first N would all look near-identical).
    vis_indices = set(np.linspace(0, len(test_ds) - 1, min(num_vis, len(test_ds)), dtype=int).tolist())

    cm_full = ConfusionMatrix(NUM_CLASSES, NEW_CLASS_NAMES, ignore_index=IGNORE_INDEX)
    cm_lidar_px = ConfusionMatrix(NUM_CLASSES, NEW_CLASS_NAMES, ignore_index=IGNORE_INDEX)
    cm_lidar_pts = ConfusionMatrix(NUM_CLASSES, NEW_CLASS_NAMES, ignore_index=IGNORE_INDEX)
    total_loss = 0.0
    vis_saved = 0
    sample_idx = 0
    lidar_px_total = 0
    frame_px_total = 0
    frames_with_lidar = 0

    for images, labels, lidar_masks, lidar_labels in tqdm(test_loader, desc="[test]"):
        images_d = images.to(device, non_blocking=True)
        labels_d = labels.to(device, non_blocking=True)
        lidar_masks_d = lidar_masks.to(device, non_blocking=True)
        lidar_labels_d = lidar_labels.to(device, non_blocking=True)

        logits = model(images_d)
        total_loss += criterion(logits, labels_d).item()
        preds = torch.argmax(logits, dim=1)

        # (1) full frame, camera ground truth
        cm_full.update(preds, labels_d)

        # (2)/(3) LiDAR-hit pixels only. Selecting with the boolean mask
        # flattens to a 1-D vector of just those pixels, which is exactly the
        # support we want the IoU computed over.
        if lidar_masks_d.any():
            cm_lidar_px.update(preds[lidar_masks_d], labels_d[lidar_masks_d])
            cm_lidar_pts.update(preds[lidar_masks_d], lidar_labels_d[lidar_masks_d])

        lidar_px_total += int(lidar_masks_d.sum().item())
        frame_px_total += int(lidar_masks_d.numel())
        frames_with_lidar += int(lidar_masks_d.flatten(1).any(dim=1).sum().item())

        for i in range(images.size(0)):
            if sample_idx in vis_indices:
                mask_i = lidar_masks[i].numpy()
                gt_i = labels[i].numpy()
                pred_i = preds[i].cpu().numpy()
                # Outside the LiDAR footprint there is nothing to show.
                lidar_gt_i = np.where(mask_i, gt_i, NO_DATA_INDEX)
                lidar_pred_i = np.where(mask_i, pred_i, NO_DATA_INDEX)
                save_vis(
                    denormalize(images[i]), colorize(gt_i), colorize(pred_i),
                    colorize(lidar_gt_i), colorize(lidar_pred_i),
                    os.path.join(out_dir, f"vis_{vis_saved:03d}.png"),
                )
                vis_saved += 1
            sample_idx += 1

    test_loss = total_loss / max(1, len(test_loader))
    coverage = lidar_px_total / max(1, frame_px_total)

    summary_text = "\n".join([
        f"Test Loss:       {test_loss:.4f}",
        f"Eval resolution: {h}x{w}",
        f"Frames:          {sample_idx} ({frames_with_lidar} with LiDAR returns)",
        f"LiDAR coverage:  {100 * coverage:.2f}% of pixels "
        f"({lidar_px_total} of {frame_px_total})",
        "",
        "=" * 72,
        "(1) FULL FRAME - all pixels vs camera label",
        "=" * 72,
        cm_full.summary(),
        "",
        "=" * 72,
        "(2) LIDAR PIXELS - LiDAR-hit pixels only, vs camera label",
        "=" * 72,
        cm_lidar_px.summary(),
        "",
        "=" * 72,
        "(3) LIDAR POINTS - LiDAR-hit pixels only, vs LiDAR point label",
        "=" * 72,
        cm_lidar_pts.summary(),
        "",
        "=" * 72,
        "mIoU comparison",
        "=" * 72,
        f"  full frame                     {cm_full.mean_iou():.4f}",
        f"  LiDAR pixels (camera GT)       {cm_lidar_px.mean_iou():.4f}",
        f"  LiDAR points (LiDAR GT)        {cm_lidar_pts.mean_iou():.4f}",
    ])
    print("\n=== Test set results ===")
    print(summary_text)

    save_confusion_matrix(cm_full, "Row-normalized confusion matrix (test, full frame)",
                           os.path.join(out_dir, "confusion_matrix.png"))
    save_confusion_matrix(cm_lidar_px, "Row-normalized confusion matrix (test, LiDAR pixels)",
                           os.path.join(out_dir, "confusion_matrix_lidar_pixels.png"))
    save_confusion_matrix(cm_lidar_pts, "Row-normalized confusion matrix (test, LiDAR points)",
                           os.path.join(out_dir, "confusion_matrix_lidar_points.png"))

    with open(os.path.join(out_dir, "results.txt"), "w") as f:
        f.write(summary_text)

    results_json = {
        "checkpoint": checkpoint,
        "checkpoint_epoch": ckpt.get("epoch"),
        "test_loss": test_loss,
        "eval_height": h,
        "eval_width": w,
        "num_frames": sample_idx,
        "num_frames_with_lidar": frames_with_lidar,
        "lidar_pixel_coverage": coverage,
        "lidar_pixels_scored": lidar_px_total,
        "full_frame": cm_full.as_dict(),
        "lidar_pixels_camera_gt": cm_lidar_px.as_dict(),
        "lidar_points_lidar_gt": cm_lidar_pts.as_dict(),
        # Flattened copies of the headline full-frame numbers, so existing
        # readers of results.json keep working unchanged.
        **cm_full.as_dict(),
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\nSaved visualizations, confusion matrices, results.txt, and results.json to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                         help="Defaults to <results_dir>/checkpoints/best.pt from --config")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    parser.add_argument("--num_vis", type=int, default=10)
    parser.add_argument("--out_dir", default=None,
                         help="Defaults to <results_dir>/test_eval from --config")
    args = parser.parse_args()
    main(args.checkpoint, args.config, args.num_vis, args.out_dir)
