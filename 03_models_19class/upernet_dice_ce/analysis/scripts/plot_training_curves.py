"""
Post-training analysis plots for the UPerNet+ResNet34 19-class run.

Reads results_256/metrics.csv (one row/epoch: losses, val_mIoU, val pixel
accuracy, val mean precision/recall/F1, lr, epoch time) and
results_256/per_class_iou.csv (one row/epoch, one column per class) --
both written automatically by src/train.py during training -- and renders
a set of PNGs to analysis/images/ for use directly in a PPT/report.

Usage (after training has produced at least a few epochs of metrics.csv):
    python analysis/scripts/plot_training_curves.py

Only needs matplotlib/numpy (already in requirements.txt) - no pandas.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

# Classes that were the clear weak points in the sibling vanilla-UNet
# 19-class run (see ../final_results.md in that project) - broken out into
# their own plot here so it's easy to see whether UPerNet struggles with
# the same classes or a different set.
WEAK_CLASSES = ["Truck", "Sedan", "Traffic_Barrier", "Umbrella", "Bridge"]


def read_csv_dict(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def mark_best(ax, epochs, values, label_fmt="best: epoch {ep} ({val:.4f})"):
    """Vertical dashed line + annotation at the epoch with the max value."""
    values = np.asarray(values, dtype=float)
    if np.all(np.isnan(values)):
        return
    best_idx = int(np.nanargmax(values))
    best_ep = epochs[best_idx]
    best_val = values[best_idx]
    ax.axvline(best_ep, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.annotate(label_fmt.format(ep=best_ep, val=best_val),
                xy=(best_ep, best_val), xytext=(5, 5),
                textcoords="offset points", fontsize=8, color="dimgray")


def plot_loss_curves(rows, out_dir):
    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [to_float(r["train_loss"]) for r in rows]
    val_loss = [to_float(r["val_loss"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="train_loss", color="tab:blue")
    ax.plot(epochs, val_loss, label="val_loss", color="tab:orange")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (0.5 Dice + 0.5 weighted CE)")
    ax.set_title("Training vs. validation loss")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)


def plot_miou_and_accuracy(rows, out_dir):
    epochs = [int(r["epoch"]) for r in rows]
    miou = [to_float(r["val_mIoU"]) for r in rows]
    acc = [to_float(r["val_pixel_accuracy"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, miou, label="val mIoU (excl. Background)", color="tab:green")
    ax.plot(epochs, acc, label="val pixel accuracy", color="tab:purple")
    mark_best(ax, epochs, miou, "best mIoU: epoch {ep} ({val:.4f})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Validation mIoU & pixel accuracy per epoch")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "val_miou_accuracy.png"), dpi=150)
    plt.close(fig)


def plot_precision_recall_f1(rows, out_dir):
    epochs = [int(r["epoch"]) for r in rows]
    precision = [to_float(r["val_mean_precision"]) for r in rows]
    recall = [to_float(r["val_mean_recall"]) for r in rows]
    f1 = [to_float(r["val_mean_f1"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, precision, label="mean precision", color="tab:red")
    ax.plot(epochs, recall, label="mean recall", color="tab:brown")
    ax.plot(epochs, f1, label="mean F1", color="tab:cyan")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Validation mean precision / recall / F1 per epoch (excl. Background)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "val_precision_recall_f1.png"), dpi=150)
    plt.close(fig)


def plot_lr_schedule(rows, out_dir):
    epochs = [int(r["epoch"]) for r in rows]
    lr = [to_float(r["lr"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, lr, color="tab:olive")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate")
    ax.set_title("LR schedule (3-epoch warmup + cosine decay) - sanity check")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "lr_schedule.png"), dpi=150)
    plt.close(fig)


def plot_epoch_time(rows, out_dir):
    epochs = [int(r["epoch"]) for r in rows]
    epoch_time = [to_float(r["epoch_time_sec"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, epoch_time, color="tab:gray", marker=".")
    mean_t = np.nanmean(epoch_time)
    ax.axhline(mean_t, color="black", linestyle=":", linewidth=1,
               label=f"mean {mean_t:.1f}s/epoch")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Train time (s)")
    ax.set_title("Epoch training time - throughput / stability check")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "epoch_time.png"), dpi=150)
    plt.close(fig)


def plot_per_class_iou(pc_rows, out_dir):
    if not pc_rows:
        return
    epochs = [int(r["epoch"]) for r in pc_rows]
    class_names = [k for k in pc_rows[0].keys() if k != "epoch"]

    # All classes - full picture, legend outside the axes since 19 lines.
    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = plt.colormaps.get_cmap("tab20")
    for i, name in enumerate(class_names):
        vals = [to_float(r[name]) for r in pc_rows]
        ax.plot(epochs, vals, label=name, color=cmap(i % 20), linewidth=1.3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("IoU")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class validation IoU per epoch (all 19 classes)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_class_iou_all.png"), dpi=150)
    plt.close(fig)

    # Weak-class focus - easier to read, and directly comparable to the
    # sibling vanilla-UNet run's known weak points (Truck, Sedan,
    # Traffic_Barrier, Umbrella, Bridge).
    present_weak = [c for c in WEAK_CLASSES if c in class_names]
    if present_weak:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name in present_weak:
            vals = [to_float(r[name]) for r in pc_rows]
            ax.plot(epochs, vals, label=name, linewidth=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel("IoU")
        ax.set_ylim(0, 1)
        ax.set_title("Per-class validation IoU - historically weak/rare classes")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "per_class_iou_weak_classes.png"), dpi=150)
        plt.close(fig)


def plot_dashboard(rows, out_dir):
    """One combined figure (2x3 grid) - quick single-image overview,
    convenient for dropping straight into a slide."""
    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [to_float(r["train_loss"]) for r in rows]
    val_loss = [to_float(r["val_loss"]) for r in rows]
    miou = [to_float(r["val_mIoU"]) for r in rows]
    acc = [to_float(r["val_pixel_accuracy"]) for r in rows]
    precision = [to_float(r["val_mean_precision"]) for r in rows]
    recall = [to_float(r["val_mean_recall"]) for r in rows]
    f1 = [to_float(r["val_mean_f1"]) for r in rows]
    lr = [to_float(r["lr"]) for r in rows]
    epoch_time = [to_float(r["epoch_time_sec"]) for r in rows]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0, 0]
    ax.plot(epochs, train_loss, label="train"); ax.plot(epochs, val_loss, label="val")
    ax.set_title("Loss"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, miou, color="tab:green")
    mark_best(ax, epochs, miou, "best: ep{ep} ({val:.3f})")
    ax.set_ylim(0, 1); ax.set_title("Val mIoU (excl. Background)"); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.plot(epochs, acc, color="tab:purple")
    ax.set_ylim(0, 1); ax.set_title("Val pixel accuracy"); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, precision, label="precision")
    ax.plot(epochs, recall, label="recall")
    ax.plot(epochs, f1, label="F1")
    ax.set_ylim(0, 1); ax.set_title("Val precision/recall/F1"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, lr, color="tab:olive")
    ax.set_title("LR schedule"); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.plot(epochs, epoch_time, color="tab:gray", marker=".")
    ax.set_title("Epoch time (s)"); ax.grid(alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Epoch")

    fig.suptitle("UPerNet + ResNet34 (19-class, 256x256) - training summary", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_dashboard.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=os.path.join(ROOT, "results_256"))
    parser.add_argument("--out_dir", default=os.path.join(ROOT, "analysis", "images"))
    args = parser.parse_args()

    metrics_path = os.path.join(args.results_dir, "metrics.csv")
    per_class_path = os.path.join(args.results_dir, "per_class_iou.csv")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(
            f"{metrics_path} not found - run src/train.py first (this script "
            f"reads the per-epoch logs it writes)."
        )

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_csv_dict(metrics_path)
    if not rows:
        raise RuntimeError(f"{metrics_path} has no data rows yet.")

    plot_loss_curves(rows, args.out_dir)
    plot_miou_and_accuracy(rows, args.out_dir)
    plot_precision_recall_f1(rows, args.out_dir)
    plot_lr_schedule(rows, args.out_dir)
    plot_epoch_time(rows, args.out_dir)
    plot_dashboard(rows, args.out_dir)

    if os.path.isfile(per_class_path):
        pc_rows = read_csv_dict(per_class_path)
        plot_per_class_iou(pc_rows, args.out_dir)
    else:
        print(f"[WARN] {per_class_path} not found - skipping per-class IoU plots.")

    print(f"Saved plots to {args.out_dir}:")
    for f in sorted(os.listdir(args.out_dir)):
        print(" -", f)


if __name__ == "__main__":
    main()
