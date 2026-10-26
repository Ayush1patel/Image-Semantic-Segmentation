# Vanilla UNet — 128×128 — Full Run Report

UAVScenes aerial semantic segmentation, plain UNet (no nested/dense skip
pathways), trained and evaluated entirely at 128×128 resolution.

## Model

| | |
|---|---|
| Architecture | `segmentation_models_pytorch.Unet` (vanilla UNet) |
| Encoder | ResNet-34, ImageNet-pretrained |
| Parameters | 24,437,384 |
| Input resolution | 128×128 (train, val, and test — no resolution change at any stage) |
| Output classes | 8 (+ `ignore_index`, never predicted) |
| Checkpoint size | 293,506,948 bytes (~280 MB) |

## Configuration used

```yaml
data:
  height: 128
  width: 128
  cache_in_ram: true

model:
  encoder_name: resnet34
  encoder_weights: imagenet

train:
  batch_size: 128
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

## Class taxonomy (from `class distribution.txt`)

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

## Dataset split (by flight sequence)

| Split | Sequences | Images |
|---|---|---|
| Train | AMtown01, AMvalley01, HKairport01, HKisland01, HKairport_GNSS01, HKisland_GNSS01, HKairport_GNSS02, HKisland_GNSS02 | 11,825 |
| Val | AMtown02, AMvalley02, HKairport02, HKisland02 | 4,207 |
| Test | AMtown03, AMvalley03, HKairport03, HKisland03 | 3,285 |

**⚠️ Known limitation**: verified via GPS track analysis that train/val/test
sequences at the same site (e.g. AMtown01/02/03) have 100% geographic
bounding-box overlap — they are repeated flights over the same physical
area, not independent locations. See "Caveats" below.

## Training summary

- Ran the full 70 epochs (no early stopping triggered).
- Train loss: 1.3711 (epoch 1) → 0.0179 (epoch 70) — steady, smooth decrease throughout.
- Val loss: bottomed out around epoch 13 (0.1830), then gradually rose to ~0.29 by epoch 70 while val mIoU kept improving (see class-review discussion below — this reflects the class-weighted CE component, not a genuine generalization regression).
- Val mIoU: 0.1026 (epoch 1) → peaked at **0.8718** (epoch 69, 0-indexed epoch 68).
- Best checkpoint: **epoch 69**, val mIoU = 0.87176 → saved to `results/checkpoints/best.pt`.
- Per-epoch time: ~13.5–14.6s (caching all images in RAM up front eliminates disk I/O during training).

## Test-set results (held-out, 3,285 images, checkpoint epoch 68)

### Overall

| Metric | Value |
|---|---|
| Test Loss | 0.3506 |
| Pixel Accuracy | 98.73% |
| **Mean IoU** | **86.25%** |
| Mean Precision | 91.42% |
| Mean Recall | 92.97% |
| Mean F1 | 92.14% |

### Per-class (sorted best → worst by IoU)

| Class | IoU | Precision | Recall | F1 | Support (px) |
|---|---|---|---|---|---|
| Vegetation | 98.66% | 99.54% | 99.11% | 99.33% | 36,968,432 |
| Water | 98.58% | 99.04% | 99.53% | 99.29% | 2,659,602 |
| Building | 91.91% | 93.95% | 97.68% | 95.78% | 2,014,093 |
| Container | 90.59% | 92.69% | 97.56% | 95.06% | 95,640 |
| Road | 90.29% | 94.13% | 95.67% | 94.90% | 3,592,647 |
| Traffic_Barrier | 80.36% | 84.75% | 93.94% | 89.11% | 80,548 |
| Bridge | 79.72% | 90.02% | 87.44% | 88.71% | 6,409 |
| Vehicle | 59.94% | 77.21% | 72.82% | 74.95% | 88,463 |

Vehicle is the clear weak point — smallest, rarest, most fine-grained
class, and 128×128 shrinks individual vehicles to just a few pixels.

## Caveats — read before quoting these numbers

1. **Geographic/spatial leakage in the split.** Confirmed via GPS bounding-box
   analysis and direct visual comparison: train (e.g. AMtown01) and test
   (AMtown03) sequences at the same site are different flights over the
   *identical* physical area (a matched GPS pair 0.57m apart showed the same
   buildings, roads, and static structures). This is not a pipeline bug —
   it's inherent to the sequence-numbering convention used for the split,
   and the official UAVScenes paper does not address it either. It means
   these numbers reflect performance under repeated-flight-same-location
   conditions, not generalization to genuinely unseen terrain.
2. No exact duplicate images or filename overlap between splits (verified
   by exhaustive MD5 hash comparison across all 19,317 images) — the
   leakage above is geographic/contextual, not literal duplication.
3. `ignore_index` excludes ~15–20% of pixels (background, solar panels,
   umbrellas, unlabeled regions) from both training and scoring — a
   deliberate, disclosed design choice, but it makes the 8-class task
   strictly easier than scoring the full scene.
4. For reference, the official UAVScenes paper's own best baseline
   (DeiT3-m transformer, 19 classes, UPerNet head) reports 68.3% mIoU —
   well below the numbers here, consistent with the effects above.