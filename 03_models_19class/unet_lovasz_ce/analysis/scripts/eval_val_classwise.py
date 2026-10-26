"""
Full 19-class IoU/Precision/Recall/F1 breakdown on the VAL split, from any
checkpoint. Same metric convention as train.py/evaluate.py (Background
excluded from the mean, not from per-class reporting).

This is deliberately separate from evaluate.py (which scores the TEST
split for the final, held-out number) -- this script is for checking
val performance mid-training without waiting for a run to finish, e.g. to
sanity-check what's driving a given epoch's headline val_mIoU.

Usage (from the project root):
    python analysis/scripts/eval_val_classwise.py
    python analysis/scripts/eval_val_classwise.py --checkpoint results_256/checkpoints/last.pt
    python analysis/scripts/eval_val_classwise.py --config configs/config_256.yaml --batch_size 8

Safe to run alongside an active training process: forces num_workers=0
(Windows spawn-safe without a __main__ guard) and a small batch size by
default so it doesn't compete heavily for GPU memory. Check `nvidia-smi`
first if training is already close to full VRAM.
"""
import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from class_mapping import BACKGROUND_INDEX, IGNORE_INDEX, NEW_CLASS_NAMES, NUM_CLASSES  # noqa: E402
from dataset import UAVScenesDataset  # noqa: E402
from metrics import ConfusionMatrix  # noqa: E402
from model import build_model  # noqa: E402
from transforms import get_eval_transform  # noqa: E402


def main(checkpoint: str, config: str, batch_size: int):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h, w = cfg["data"]["height"], cfg["data"]["width"]

    val_ds = UAVScenesDataset("val", transform=get_eval_transform(h, w),
                               index_dir=os.path.join(ROOT, "data_index"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(encoder_name=cfg["model"]["encoder_name"], encoder_weights=None,
                         num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    epoch = ckpt.get("epoch")
    print(f"Loaded {checkpoint}  (epoch {epoch + 1 if epoch is not None else '?'}, "
          f"best_miou recorded={ckpt.get('best_miou')})")

    cm = ConfusionMatrix(NUM_CLASSES, NEW_CLASS_NAMES, ignore_index=IGNORE_INDEX,
                          exclude_indices=[BACKGROUND_INDEX])

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="val inference"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=True):
                logits = model(images)
            preds = torch.argmax(logits, dim=1)
            cm.update(preds, labels)

    print()
    print(cm.summary())
    return cm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(ROOT, "results_256", "checkpoints", "best.pt"))
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "config_256.yaml"))
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args.checkpoint, args.config, args.batch_size)
