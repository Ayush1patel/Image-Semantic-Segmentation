import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from class_mapping import colorize
from transforms import denormalize


def build_fixed_vis_batch(dataset, num_images: int, seed: int = 42):
    """Pick a fixed, deterministic subset of the dataset to visualize
    every epoch, so predictions are directly comparable over training."""
    rng = np.random.RandomState(seed)
    n = min(num_images, len(dataset))
    indices = sorted(rng.choice(len(dataset), size=n, replace=False).tolist())
    images, labels = [], []
    for idx in indices:
        img, lbl = dataset[idx]
        images.append(img)
        labels.append(lbl)
    return torch.stack(images), torch.stack(labels), indices


@torch.no_grad()
def save_progress_grid(model, images, labels, device, epoch, out_dir, amp=True):
    """Run the fixed vis batch through the model and save an
    Image | Ground Truth | Prediction comparison grid for this epoch."""
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    images_d = images.to(device)
    with torch.amp.autocast(device_type=device.type, enabled=amp):
        logits = model(images_d)
    preds = torch.argmax(logits, dim=1).cpu()

    n = images.size(0)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        img_rgb = denormalize(images[i])
        gt_color = colorize(labels[i].numpy())
        pred_color = colorize(preds[i].numpy())
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(gt_color)
        axes[i, 1].axis("off")
        axes[i, 2].imshow(pred_color)
        axes[i, 2].axis("off")
        if i == 0:
            axes[i, 0].set_title("Image")
            axes[i, 1].set_title("Ground Truth")
            axes[i, 2].set_title("Prediction")

    fig.suptitle(f"Epoch {epoch + 1}")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"epoch_{epoch + 1:03d}.png")
    plt.savefig(out_path, dpi=90)
    plt.close(fig)
    return out_path
