# Vanilla UNet — 512×512 — Full Run Report

UAVScenes aerial semantic segmentation, plain UNet (no nested/dense skip
pathways), trained and evaluated entirely at 512×512 resolution. Same
configuration as the 128×128 run in every respect except resolution and
batch size (compute-bound at this resolution, so batch size doesn't
materially change epoch time).

## Model

| | |
|---|---|
| Architecture | `segmentation_models_pytorch.Unet` (vanilla UNet) |
| Encoder | ResNet-34, ImageNet-pretrained |
| Parameters | 24,437,384 (identical to the 128×128 run — only input resolution changed) |
| Input resolution | 512×512 (train, val, and test — no resolution change at any stage) |
| Output classes | 8 (+ `ignore_index`, never predicted) |
| Checkpoint size | 293,506,948 bytes (~280 MB) |

## Configuration used

```yaml
data:
  height: 512
  width: 512
  cache_in_ram: true    # ~17GB for train+val combined, fits in system RAM

model:
  encoder_name: resnet34
  encoder_weights: imagenet

train:
  batch_size: 32         # benchmarked: epoch time ~flat across batch sizes (compute-bound)
  epochs: 70
  lr: 0.0003
  weight_decay: 0.0001
  warmup_epochs: 3
  dice_weight: 0.5
  ce_weight: 0.5
  class_weight_clip_max: 10.0
  grad_clip_norm: 5.0
  amp: true
  early_stop_patience: 15
  seed: 42
```
Loss: 0.5 × Dice + 0.5 × class-weighted Cross-Entropy, both `ignore_index`-aware.
Optimizer: AdamW, cosine LR schedule with 3-epoch warmup.

## Class taxonomy (from `class distribution.txt`) — identical to the 128×128 run

| ID | Class | Raw UAVScenes classes merged into it |
|---|---|---|
| 0 | Building | roof, transparent_roof |
| 1 | Road | dirt_motor_road, paved_motor_road, airstrip, car_park, paved_walk |
| 2 | Water | river, pool |
| 3 | Vegetation | green_field, wild_field |
| 4 | Bridge | bridge |
| 5 | Container | container |
| 6 | Traffic_Barrier | traffic_barrier |
| 7 | Vehicle | sedan, truck |
| `ignore_index=255` | — (excluded from loss & metrics) | background, solar_board, umbrella, + unnamed/reserved raw ids |

## Dataset split (by flight sequence) — identical to the 128×128 run

| Split | Sequences | Images |
|---|---|---|
| Train | AMtown01, AMvalley01, HKairport01, HKisland01, HKairport_GNSS01, HKisland_GNSS01, HKairport_GNSS02, HKisland_GNSS02 | 11,825 |
| Val | AMtown02, AMvalley02, HKairport02, HKisland02 | 4,207 |
| Test | AMtown03, AMvalley03, HKairport03, HKisland03 | 3,285 |

**⚠️ Known limitation** (same as the 128×128 run — this is a property of
the split, not the resolution): train/val/test sequences at the same site
have 100% GPS bounding-box overlap. See "Caveats" below.

## Training summary

- Ran the full 70 epochs (no early stopping triggered).
- Train loss: → 0.0091 by epoch 70 (roughly half the final train loss of the 128×128 run — the model fits the (larger, more detailed) training images more precisely at this resolution).
- Val loss: settled around 0.115–0.121 for the last ~10 epochs.
- Val mIoU: peaked at **0.9349** (epoch 68, 0-indexed epoch 67), ending at 0.9347 (epoch 70) — very stable in the final stretch.
- Best checkpoint: **epoch 68**, val mIoU = 0.934872 → saved to `results_512/checkpoints/best.pt`.
- Per-epoch time: ~217–219s (≈3.6 min/epoch — compute-bound at this resolution; batch size does not meaningfully change this).
- Total training time: ~4+ hours (70 × ~218s ≈ 254 min, plus RAM-cache build).

## Test-set results (held-out, 3,285 images, checkpoint epoch 67)

### Overall

| Metric | Value |
|---|---|
| Test Loss | 0.1416 |
| Pixel Accuracy | 99.37% |
| **Mean IoU** | **92.83%** |
| Mean Precision | 95.46% |
| Mean Recall | 96.93% |
| Mean F1 | 96.18% |

### Per-class (sorted best → worst by IoU)

| Class | IoU | Precision | Recall | F1 | Support (px) |
|---|---|---|---|---|---|
| Vegetation | 99.29% | 99.71% | 99.58% | 99.64% | 591,473,000 |
| Water | 99.05% | 99.40% | 99.64% | 99.52% | 42,452,217 |
| Building | 96.87% | 97.71% | 99.11% | 98.41% | 32,210,135 |
| Road | 95.06% | 97.45% | 97.49% | 97.47% | 57,488,543 |
| Container | 94.80% | 96.00% | 98.69% | 97.33% | 1,516,537 |
| Bridge | 88.84% | 93.98% | 94.20% | 94.09% | 102,579 |
| Traffic_Barrier | 85.39% | 89.56% | 94.82% | 92.12% | 1,288,595 |
| Vehicle | 83.32% | 89.89% | 91.93% | 90.90% | 1,435,330 |

## Comparison to the 128×128 run

Resolution is the *only* thing that changed between these two runs (same
split, same classes, same architecture, same hyperparameters, same seed) —
so the gains below isolate the resolution effect specifically.

| Class | 128×128 IoU | 512×512 IoU | Gain |
|---|---|---|---|
| **Vehicle** | 59.94% | **83.32%** | **+23.4 pts** |
| Bridge | 79.72% | 88.84% | +9.1 pts |
| Road | 90.29% | 95.06% | +4.8 pts |
| Building | 91.91% | 96.87% | +5.0 pts |
| Traffic_Barrier | 80.36% | 85.39% | +5.0 pts |
| Container | 90.59% | 94.80% | +4.2 pts |
| Water | 98.58% | 99.05% | +0.5 pts |
| Vegetation | 98.66% | 99.29% | +0.6 pts |
| **Overall mIoU** | 86.25% | **92.83%** | **+6.6 pts** |

Vehicle — the smallest, most fine-grained class — benefited far more than
any other class from the resolution increase, as expected; the already
near-saturated large classes (Vegetation, Water) barely moved.

## Caveats — read before quoting these numbers

1. **Geographic/spatial leakage in the split** (same finding as the 128×128
   run — this is a property of the dataset split, unaffected by
   resolution). Confirmed via GPS bounding-box analysis and direct visual
   comparison: train (e.g. AMtown01) and test (AMtown03) sequences at the
   same site are different flights over the *identical* physical area (a
   matched GPS pair 0.57m apart showed the same buildings, roads, and
   static structures). Not a pipeline bug — inherent to the
   sequence-numbering convention used for the split, and the official
   UAVScenes paper does not address it either. These numbers reflect
   performance under repeated-flight-same-location conditions, not
   generalization to genuinely unseen terrain.
2. No exact duplicate images or filename overlap between splits (verified
   by exhaustive MD5 hash comparison across all 19,317 images).
3. `ignore_index` excludes ~15–20% of pixels (background, solar panels,
   umbrellas, unlabeled regions) from both training and scoring — a
   deliberate, disclosed design choice, but it makes the 8-class task
   strictly easier than scoring the full scene.
4. For reference, the official UAVScenes paper's own best baseline
   (DeiT3-m transformer, 19 classes, UPerNet head) reports 68.3% mIoU —
   well below the numbers here, consistent with the effects above.