"""
Evaluate a trained checkpoint on the held-out test split.

Computes the confusion matrix and per-class / mean IoU, then writes
results.json, a confusion-matrix plot and a set of prediction
visualisations (image | ground truth | prediction) into
<results_dir>/test_eval/. Defaults to results_dir/checkpoints/best.pt.

Usage:
    python src/evaluate.py --config configs/config_256.yaml
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

from class_mapping import BACKGROUND_INDEX, IGNORE_INDEX, NEW_CLASS_NAMES, NUM_CLASSES, colorize
from dataset import UAVScenesDataset
from losses import DiceCELoss, compute_class_weights
from metrics import ConfusionMatrix
from model import build_model
from transforms import denormalize, get_eval_transform

ROOT = os.path.join(os.path.dirname(__file__), "..")


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

    test_ds = UAVScenesDataset("test", transform=get_eval_transform(h, w))
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"],
                              shuffle=False, num_workers=cfg["data"]["num_workers"])

    model = build_model(encoder_name=cfg["model"]["encoder_name"],
                         encoder_weights=None, num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint {checkpoint} (epoch {ckpt.get('epoch')}, "
          f"best_miou={ckpt.get('best_miou')})")

    # Same loss as training (class-weighted Dice + CE, ignore_index-aware),
    # so "test loss" is directly comparable to the train/val loss curves.
    stats_path = os.path.join(ROOT, "data_index", "train_class_stats.json")
    class_weights = None
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        class_weights = compute_class_weights(
            stats["class_pixel_freq"], NUM_CLASSES,
            clip_max=cfg["train"]["class_weight_clip_max"],
        ).to(device)

    # IGNORE_INDEX is None in this project; convert to -100 for PyTorch.
    _loss_ignore = IGNORE_INDEX if IGNORE_INDEX is not None else -100
    criterion = DiceCELoss(
        NUM_CLASSES, class_weights=class_weights,
        dice_weight=cfg["train"]["dice_weight"], ce_weight=cfg["train"]["ce_weight"],
        ignore_index=_loss_ignore,
    )

    # Fresh output folder every run - wipe any stale images/results from a
    # previous checkpoint or a previous num_vis setting.
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # Spread visualized samples evenly across the whole test set (index order
    # is sequential flight frames, so the first N would all look near-identical).
    vis_indices = set(np.linspace(0, len(test_ds) - 1, min(num_vis, len(test_ds)), dtype=int).tolist())

    cm = ConfusionMatrix(NUM_CLASSES, NEW_CLASS_NAMES, ignore_index=IGNORE_INDEX,
                         exclude_indices=[BACKGROUND_INDEX])
    total_loss = 0.0
    vis_saved = 0
    sample_idx = 0

    for images, labels in tqdm(test_loader, desc="[test]"):
        images_d = images.to(device, non_blocking=True)
        labels_d = labels.to(device, non_blocking=True)
        logits = model(images_d)
        total_loss += criterion(logits, labels_d).item()
        preds = torch.argmax(logits, dim=1)
        cm.update(preds, labels_d)

        for i in range(images.size(0)):
            if sample_idx in vis_indices:
                img_rgb = denormalize(images[i])
                gt_color = colorize(labels[i].numpy())
                pred_color = colorize(preds[i].cpu().numpy())

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                axes[0].imshow(img_rgb); axes[0].set_title("Image"); axes[0].axis("off")
                axes[1].imshow(gt_color); axes[1].set_title("Ground Truth"); axes[1].axis("off")
                axes[2].imshow(pred_color); axes[2].set_title("Prediction"); axes[2].axis("off")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"vis_{vis_saved:03d}.png"), dpi=120)
                plt.close(fig)
                vis_saved += 1
            sample_idx += 1

    test_loss = total_loss / max(1, len(test_loader))

    summary_text = f"Test Loss:       {test_loss:.4f}\n" + cm.summary()
    print("\n=== Test set results ===")
    print(summary_text)

    # Confusion matrix heatmap (row-normalized), diagonal and off-diagonal
    # rendered on SEPARATE color scales. A single linear 0-1 scale makes
    # every off-diagonal cell look like the same near-black "nothing here"
    # color, because the true diagonal values (70-95%) dwarf the
    # off-diagonal ones (usually <15%) -- exactly the cells that matter
    # for diagnosing false positives / confusions. Splitting the scales,
    # capping the off-diagonal one at its own 99th percentile (not 1.0),
    # and labeling cells numerically makes both "how correct" and "what
    # it's confused with" readable without relying on color perception.
    mat = cm.mat.astype(np.float64)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm_mat = mat / row_sums

    diag_vals = np.diag(norm_mat).copy()
    off_diag = norm_mat.copy()
    np.fill_diagonal(off_diag, np.nan)
    finite_off = off_diag[~np.isnan(off_diag)]
    off_vmax = max(float(np.percentile(finite_off, 99)), 0.02)

    fig, ax = plt.subplots(figsize=(15, 11))

    off_cmap = plt.cm.Reds.copy()
    off_cmap.set_bad(color="white")
    ax.imshow(off_diag, cmap=off_cmap, vmin=0.0, vmax=off_vmax)

    diag_cmap = plt.cm.Greens
    diag_norm = plt.Normalize(vmin=0.4, vmax=1.0)
    for i in range(NUM_CLASSES):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                    facecolor=diag_cmap(diag_norm(diag_vals[i])),
                                    edgecolor="none"))

    annotate_thresh = 0.01  # label any cell holding >=1% of that true class's pixels
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = norm_mat[i, j]
            if v < annotate_thresh:
                continue
            if i == j:
                txt_color = "white" if diag_norm(v) > 0.6 else "black"
            else:
                txt_color = "white" if (v / off_vmax) > 0.6 else "black"
            ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center",
                     fontsize=6.5, color=txt_color)

    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(NEW_CLASS_NAMES, rotation=90)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(NEW_CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")

    # Short, factual title: model / resolution / loss -- not a description
    # of how to read the chart (that belongs in the surrounding doc/slide,
    # not baked into the image).
    loss_name = "CE + Lovasz-Softmax" if "lovasz_weight" in cfg["train"] else "CE + Dice"
    h, w = cfg["data"]["height"], cfg["data"]["width"]
    ax.set_title(
        f"Confusion Matrix — UNet ({cfg['model']['encoder_name']}), "
        f"{h}×{w}, {loss_name}"
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, NUM_CLASSES, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, NUM_CLASSES, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Two colorbars on MANUALLY placed axes (figure-fraction coordinates),
    # not fig.colorbar(..., ax=ax, pad=...) -- that approach recomputes the
    # main axes' box on every call and then tight_layout() re-shuffles
    # everything again afterwards, which is what produced both problems
    # last time: a large empty gap before the bars, and the off-diagonal
    # bar's label drifting into the diagonal bar's tick numbers. Fixed
    # positions sidestep both: main plot occupies a fixed box, the two
    # colorbars sit immediately right of it with a small gap between the
    # plot and the bars and a deliberately wide gap between the two bars
    # so each one's label has clear space of its own.
    fig.subplots_adjust(left=0.14, right=0.79, bottom=0.22, top=0.93)

    sm_off = plt.cm.ScalarMappable(cmap=off_cmap, norm=plt.Normalize(vmin=0, vmax=off_vmax))
    cax_off = fig.add_axes((0.805, 0.22, 0.02, 0.71))
    cbar2 = fig.colorbar(sm_off, cax=cax_off)
    cbar2.set_label("Confused with (off-diagonal)", fontsize=9)
    cbar2.ax.tick_params(labelsize=8)

    sm_diag = plt.cm.ScalarMappable(cmap=diag_cmap, norm=diag_norm)
    cax_diag = fig.add_axes((0.91, 0.22, 0.02, 0.71))
    cbar1 = fig.colorbar(sm_diag, cax=cax_diag)
    cbar1.set_label("Correct (diagonal)", fontsize=9)
    cbar1.ax.tick_params(labelsize=8)

    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(out_dir, "results.txt"), "w") as f:
        f.write(summary_text)

    results_json = {
        "checkpoint": checkpoint,
        "checkpoint_epoch": ckpt.get("epoch"),
        "test_loss": test_loss,
        **cm.as_dict(),
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\nSaved visualizations, confusion matrix, results.txt, and results.json to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                         help="Defaults to <results_dir>/checkpoints/best.pt from --config")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "config_256.yaml"))
    parser.add_argument("--num_vis", type=int, default=10)
    parser.add_argument("--out_dir", default=None,
                         help="Defaults to <results_dir>/test_eval from --config")
    args = parser.parse_args()
    main(args.checkpoint, args.config, args.num_vis, args.out_dir)
