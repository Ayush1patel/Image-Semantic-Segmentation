import argparse
import csv
import json
import math
import os
import random
import shutil
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cached_dataset import CachedUAVScenesDataset
from class_mapping import (BACKGROUND_INDEX, IGNORE_INDEX, NEW_CLASS_NAMES, NUM_CLASSES,
                            RAW_CLASSES, RAW_TO_NEW)
from dataset import UAVScenesDataset
from losses import DiceCELoss, compute_class_weights
from metrics import ConfusionMatrix
from model import build_model
from progress_vis import build_fixed_vis_batch, save_progress_grid
from transforms import (get_eval_transform, get_eval_transform_cached,
                         get_train_transform, get_train_transform_cached)

ROOT = os.path.join(os.path.dirname(__file__), "..")


def setup_results_dir(results_dir: str, cfg_path: str, cfg: dict):
    """Create the results folder and drop everything needed to reproduce
    or interpret the run: the exact config used and the class mapping."""
    os.makedirs(results_dir, exist_ok=True)
    shutil.copy(cfg_path, os.path.join(results_dir, "config_used.yaml"))
    class_info = {
        "new_class_names": NEW_CLASS_NAMES,
        "ignore_index": IGNORE_INDEX,
        "raw_class_names": RAW_CLASSES,
        "raw_id_to_class_id_or_ignore": RAW_TO_NEW,
    }
    with open(os.path.join(results_dir, "class_info.json"), "w") as f:
        json.dump(class_info, f, indent=2)


class MetricsLogger:
    """Appends one row per epoch to results_dir/metrics.csv.

    Beyond the loss/mIoU logged by the sibling UNet project, this also logs
    val pixel accuracy and mean precision/recall/F1 every epoch, so the
    post-hoc analysis plots (analysis/scripts/plot_training_curves.py) can
    show more than just a single loss/mIoU curve."""

    FIELDS = ["epoch", "train_loss", "val_loss", "val_mIoU", "val_pixel_accuracy",
              "val_mean_precision", "val_mean_recall", "val_mean_f1",
              "lr", "epoch_time_sec", "is_best"]

    def __init__(self, results_dir: str):
        self.path = os.path.join(results_dir, "metrics.csv")
        if not os.path.isfile(self.path):
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def log(self, **row):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(row)


class PerClassIoULogger:
    """Appends one row per epoch to results_dir/per_class_iou.csv - one
    column per class. Lets the analysis plots show per-class IoU trends
    over training (e.g. do the weak/rare classes like Truck, Sedan,
    Traffic_Barrier ever catch up, or do they plateau early)."""

    def __init__(self, results_dir: str, class_names):
        self.class_names = class_names
        self.fields = ["epoch"] + list(class_names)
        self.path = os.path.join(results_dir, "per_class_iou.csv")
        if not os.path.isfile(self.path):
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()

    def log(self, epoch: int, per_class_iou: np.ndarray):
        row = {"epoch": epoch}
        for name, val in zip(self.class_names, per_class_iou):
            row[name] = "" if np.isnan(val) else round(float(val), 6)
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(row)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_lr_lambda(warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def main(cfg_path: str, resume: str = None):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    results_dir = os.path.join(ROOT, cfg["train"]["results_dir"])
    setup_results_dir(results_dir, cfg_path, cfg)
    metrics_logger = MetricsLogger(results_dir)
    per_class_logger = PerClassIoULogger(results_dir, NEW_CLASS_NAMES)
    print(f"Results folder: {results_dir}  (config_used.yaml, class_info.json, "
          f"metrics.csv, per_class_iou.csv, checkpoints/, progress/, tensorboard/)")

    h, w = cfg["data"]["height"], cfg["data"]["width"]
    cache_in_ram = cfg["data"].get("cache_in_ram", False)
    if cache_in_ram:
        print(f"Caching train+val images in RAM at {h}x{w} (one-time cost)...")
        train_ds = CachedUAVScenesDataset("train", h, w, transform=get_train_transform_cached())
        val_ds = CachedUAVScenesDataset("val", h, w, transform=get_eval_transform_cached())
    else:
        train_ds = UAVScenesDataset("train", transform=get_train_transform(h, w))
        val_ds = UAVScenesDataset("val", transform=get_eval_transform(h, w))
    print(f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

    # RAM-cached datasets hold large numpy arrays; on Windows, DataLoader
    # workers use spawn (not fork), so num_workers>0 would re-pickle/copy
    # those arrays per worker. Force single-process loading in that case --
    # RAM indexing + augmentation is already fast enough without workers.
    loader_workers = 0 if cache_in_ram else cfg["data"]["num_workers"]
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=loader_workers, pin_memory=True, drop_last=True,
        persistent_workers=loader_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=loader_workers, pin_memory=True,
        persistent_workers=loader_workers > 0,
    )

    vis_images, vis_labels, vis_indices = build_fixed_vis_batch(
        val_ds, cfg["train"]["num_progress_images"], seed=cfg["train"]["seed"],
    )
    progress_dir = os.path.join(results_dir, "progress")
    print(f"Progress snapshots: {len(vis_indices)} fixed val samples (indices {vis_indices}) "
          f"-> {progress_dir}/epoch_XXX.png every epoch")

    model = build_model(
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=cfg["model"]["encoder_weights"],
        num_classes=NUM_CLASSES,
    ).to(device)

    stats_path = os.path.join(ROOT, "data_index", "train_class_stats.json")
    class_weights = None
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        class_weights = compute_class_weights(
            stats["class_pixel_freq"], NUM_CLASSES,
            clip_max=cfg["train"]["class_weight_clip_max"],
        ).to(device)
        print("Class weights:", class_weights.tolist())
    else:
        print("[WARN] No train_class_stats.json found; using unweighted CE. "
              "Run scripts/build_index.py first.")

    # IGNORE_INDEX is None in this project (all pixels are scored).
    # CrossEntropyLoss and DiceLoss accept ignore_index=-100 as the
    # PyTorch default "no ignore" sentinel when None is passed here.
    _loss_ignore = IGNORE_INDEX if IGNORE_INDEX is not None else -100
    criterion = DiceCELoss(
        NUM_CLASSES, class_weights=class_weights,
        dice_weight=cfg["train"]["dice_weight"], ce_weight=cfg["train"]["ce_weight"],
        ignore_index=_loss_ignore,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"],
    )
    steps_per_epoch = len(train_loader)
    lr_lambda = build_lr_lambda(cfg["train"]["warmup_epochs"], cfg["train"]["epochs"], steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.amp.GradScaler(device.type, enabled=cfg["train"]["amp"])

    ckpt_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_dir = os.path.join(results_dir, "tensorboard")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    start_epoch = 0
    best_miou = -1.0
    epochs_since_improve = 0

    if resume and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt.get("best_miou", -1.0)
        print(f"Resumed from {resume} at epoch {start_epoch}, best_miou={best_miou:.4f}")

    global_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']} [train]")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=cfg["train"]["amp"]):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss /= max(1, len(train_loader))
        epoch_time = time.time() - t0
        writer.add_scalar("train/loss_epoch", epoch_loss, epoch)
        print(f"Epoch {epoch+1} train_loss={epoch_loss:.4f} time={epoch_time:.1f}s")

        if (epoch + 1) % cfg["train"]["val_every"] == 0:
            val_loss, cm = validate(model, val_loader, criterion, device, cfg["train"]["amp"])
            val_miou = cm.mean_iou()
            val_pixel_acc = cm.pixel_accuracy()
            val_precision = cm.mean_precision()
            val_recall = cm.mean_recall()
            val_f1 = cm.mean_f1()

            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/mIoU", val_miou, epoch)
            writer.add_scalar("val/pixel_accuracy", val_pixel_acc, epoch)
            writer.add_scalar("val/mean_precision", val_precision, epoch)
            writer.add_scalar("val/mean_recall", val_recall, epoch)
            writer.add_scalar("val/mean_f1", val_f1, epoch)
            for name, iou in zip(NEW_CLASS_NAMES, cm.per_class_iou()):
                if not np.isnan(iou):
                    writer.add_scalar(f"val_per_class_iou/{name}", iou, epoch)

            print(f"=== Epoch {epoch+1}/{cfg['train']['epochs']}: "
                  f"train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f}  val_mIoU={val_miou:.4f}  "
                  f"val_pixel_acc={val_pixel_acc:.4f}  val_F1={val_f1:.4f} ===")

            vis_path = save_progress_grid(model, vis_images, vis_labels, device, epoch,
                                           progress_dir, amp=cfg["train"]["amp"])
            print(f"  -> saved progress snapshot {vis_path}")

            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_miou": best_miou,
                "cfg": cfg,
            }
            torch.save(ckpt, os.path.join(ckpt_dir, "last.pt"))

            is_best = val_miou > best_miou
            if is_best:
                best_miou = val_miou
                epochs_since_improve = 0
                ckpt["best_miou"] = best_miou
                best_path = os.path.join(ckpt_dir, "best.pt")
                torch.save(ckpt, best_path)
                print(f"  -> new best mIoU {best_miou:.4f}, saved {best_path}")
            else:
                epochs_since_improve += 1

            metrics_logger.log(
                epoch=epoch + 1, train_loss=round(epoch_loss, 6),
                val_loss=round(val_loss, 6), val_mIoU=round(val_miou, 6),
                val_pixel_accuracy=round(val_pixel_acc, 6),
                val_mean_precision=round(val_precision, 6),
                val_mean_recall=round(val_recall, 6),
                val_mean_f1=round(val_f1, 6),
                lr=scheduler.get_last_lr()[0], epoch_time_sec=round(epoch_time, 1),
                is_best=is_best,
            )
            per_class_logger.log(epoch + 1, cm.per_class_iou())

            if not is_best and epochs_since_improve >= cfg["train"]["early_stop_patience"]:
                print(f"Early stopping: no val mIoU improvement in "
                      f"{cfg['train']['early_stop_patience']} epochs.")
                break

    writer.close()
    print(f"Training done. Best val mIoU: {best_miou:.4f}")
    print(f"All results saved under: {results_dir}")


@torch.no_grad()
def validate(model, val_loader, criterion, device, amp: bool):
    model.eval()
    cm = ConfusionMatrix(NUM_CLASSES, NEW_CLASS_NAMES, ignore_index=IGNORE_INDEX,
                         exclude_indices=[BACKGROUND_INDEX])
    total_loss = 0.0
    for images, labels in tqdm(val_loader, desc="[val]"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        cm.update(preds, labels)
    total_loss /= max(1, len(val_loader))
    return total_loss, cm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "config_256.yaml"))
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    main(args.config, args.resume)
