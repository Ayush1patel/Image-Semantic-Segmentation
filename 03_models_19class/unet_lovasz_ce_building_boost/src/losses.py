import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import LovaszLoss


class CELovaszLoss(nn.Module):
    """Combined class-weighted Cross-Entropy + Lovasz-Softmax loss for
    multiclass segmentation.

    Replaces the sibling project's `DiceCELoss` (0.5 Dice + 0.5 weighted
    CE) with 0.5 Lovasz-Softmax + 0.5 weighted CE. Everything else in this
    project (architecture, split, optimizer, schedule, batch size, seed)
    is unchanged -- this is the only deliberate difference.

    Why Lovasz instead of Dice here: Dice and CE are both effectively
    per-pixel objectives (Dice is a smooth, count-weighted overlap score;
    it doesn't identify *which* pixels are costing IoU the most).
    Lovasz-Softmax is a convex surrogate built directly from the Jaccard
    index: it ranks pixels by prediction error and weights gradient by
    each one's marginal effect on IoU, so it concentrates gradient on the
    specific errors that actually hurt IoU -- solid false-positive blobs
    and boundary noise -- rather than averaging uniformly like Dice/CE do.
    Goal: reduce false positives and raise IoU on Building and its
    visually-confusable neighbors (Container, Transparent_Roof, Car_Park,
    Paved_Walk), where the sibling run's confusion matrix already showed
    over-prediction.

    Why combined with CE, not Lovasz alone: Lovasz's gradient is only
    informative once predictions are non-trivial (it ranks errors -- at
    random init there's no meaningful ranking yet), so it warms up poorly
    on its own and doesn't shape calibrated confidence the way a
    likelihood-based loss does. More importantly for this dataset: Lovasz
    only computes a per-class Jaccard term where that class has pixels
    present, so a class entirely absent from a given batch gets literally
    zero gradient from Lovasz that step. The dataset-wide scan in
    ../../01_dataset_analysis found several trainable classes (Bridge, Pool,
    Solar_Board, Container, Traffic_Barrier, ...) are entirely absent from
    most sequences -- with batch_size=32 it's entirely plausible for a
    batch to contain none of them. CE always contributes dense, weighted
    gradient for every class in every batch (reinforced by the existing
    median-frequency class weights below), so it keeps those sparse
    classes learnable throughout training while Lovasz sharpens the
    IoU-costing errors on top.

    With IGNORE_INDEX=None (all pixels scored), every pixel contributes
    gradient - no ignore bucket, same convention as the sibling project.
    """

    def __init__(self, num_classes: int, class_weights: torch.Tensor = None,
                 lovasz_weight: float = 0.5, ce_weight: float = 0.5,
                 ignore_index: int = -100):
        super().__init__()
        # per_image=False: Lovasz computed over the whole flattened batch,
        # not averaged per-image -- more sample-efficient for rare classes
        # (a class needs only batch-level presence, not per-image presence,
        # to get a gradient contribution) and smp's documented default.
        self.lovasz = LovaszLoss(mode="multiclass", per_image=False, ignore_index=ignore_index)
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.lovasz_weight = lovasz_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (self.lovasz_weight * self.lovasz(logits, target)
                + self.ce_weight * self.ce(logits, target))


def compute_class_weights(class_pixel_freq, num_classes: int, clip_max: float = 10.0):
    """Median-frequency balancing: weight_c = median(freq) / freq_c.
    Identical to the sibling project's version -- same split, same
    data_index/train_class_stats.json, same class distribution."""
    freq = torch.tensor(class_pixel_freq, dtype=torch.float32)
    freq = torch.clamp(freq, min=1e-8)
    median_freq = torch.median(freq)
    weights = median_freq / freq
    weights = torch.clamp(weights, max=clip_max)
    return weights
