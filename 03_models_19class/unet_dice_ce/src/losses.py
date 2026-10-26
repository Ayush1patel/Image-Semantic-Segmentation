import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss


class DiceCELoss(nn.Module):
    """Combined Dice + (class-weighted) Cross-Entropy loss for multiclass
    segmentation. Dice directly targets IoU-like overlap; weighted CE
    stabilizes training and helps with the rare classes.

    With IGNORE_INDEX=None (all pixels scored), every pixel contributes
    gradient - no ignore bucket."""

    def __init__(self, num_classes: int, class_weights: torch.Tensor = None,
                 dice_weight: float = 0.5, ce_weight: float = 0.5,
                 ignore_index: int = -100):
        super().__init__()
        self.dice = DiceLoss(mode="multiclass", from_logits=True, ignore_index=ignore_index)
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (self.dice_weight * self.dice(logits, target)
                + self.ce_weight * self.ce(logits, target))


def compute_class_weights(class_pixel_freq, num_classes: int, clip_max: float = 10.0):
    """Median-frequency balancing: weight_c = median(freq) / freq_c.
    class_pixel_freq should cover all num_classes trainable ids (including
    Background at index 0 in this project)."""
    freq = torch.tensor(class_pixel_freq, dtype=torch.float32)
    freq = torch.clamp(freq, min=1e-8)
    median_freq = torch.median(freq)
    weights = median_freq / freq
    weights = torch.clamp(weights, max=clip_max)
    return weights
