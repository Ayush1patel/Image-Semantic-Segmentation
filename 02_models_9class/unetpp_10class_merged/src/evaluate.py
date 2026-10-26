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

from class_mapping import NEW_CLASS_NAMES, NUM_CLASSES, colorize
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

    # Same loss as training (class-weighted Dice + CE), so "test loss" is
    # directly comparable to the train/val loss curves in metrics.csv.
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
    )

    # Fresh output folder every run - wipe any stale images/results from a
    # previous checkpoint or a previous num_vis setting.
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # Spread visualized samples evenly across the whole test set (index order
    # is sequential flight frames, so the first N would all look near-identical).
    vis_indices = set(np.linspace(0, len(test_ds) - 1, min(num_vis, len(test_ds)), dtype=int).tolist())

    cm = ConfusionMatrix(NUM_CLASSES, NEW_CLASS_NAMES)
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

    # Confusion matrix heatmap (row-normalized)
    mat = cm.mat.astype(np.float64)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm_mat = mat / row_sums

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm_mat, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(NEW_CLASS_NAMES, rotation=90)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(NEW_CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
    ax.set_title("Row-normalized confusion matrix (test)")
    fig.colorbar(im)
    plt.tight_layout()
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
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    parser.add_argument("--num_vis", type=int, default=10)
    parser.add_argument("--out_dir", default=None,
                         help="Defaults to <results_dir>/test_eval from --config")
    args = parser.parse_args()
    main(args.checkpoint, args.config, args.num_vis, args.out_dir)
