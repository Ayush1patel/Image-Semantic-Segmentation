import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss


class DiceCELoss(nn.Module):
    """Combined Dice + (class-weighted) Cross-Entropy loss for multiclass
    segmentation. Dice directly targets IoU-like overlap; weighted CE
    stabilizes training and helps with the rare classes.

    ignore_index pixels are excluded from both terms - they never contribute
    gradient and are never predicted. Pass ignore_index=None (the current
    taxonomy's setting, since Background is now a real class) to score every
    pixel; CE then falls back to its own -100 sentinel, which no label uses."""

    CE_NO_IGNORE = -100  # torch's default sentinel; unreachable for our labels

    def __init__(self, num_classes: int, class_weights: torch.Tensor = None,
                 dice_weight: float = 0.5, ce_weight: float = 0.5,
                 ignore_index: int = None):
        super().__init__()
        self.dice = DiceLoss(mode="multiclass", from_logits=True, ignore_index=ignore_index)
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=self.CE_NO_IGNORE if ignore_index is None else ignore_index,
        )
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (self.dice_weight * self.dice(logits, target)
                + self.ce_weight * self.ce(logits, target))


def compute_class_weights(class_pixel_freq, num_classes: int, clip_max: float = 10.0):
    """Median-frequency balancing: weight_c = median(freq) / freq_c.
    class_pixel_freq comes from build_index.py's histogram over the
    num_classes trainable ids - which now covers every pixel, Background
    included, so the frequencies sum to 1 over the whole train split."""
    freq = torch.tensor(class_pixel_freq, dtype=torch.float32)
    freq = torch.clamp(freq, min=1e-8)
    median_freq = torch.median(freq)
    weights = median_freq / freq
    weights = torch.clamp(weights, max=clip_max)
    return weights
