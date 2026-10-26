# Training progress summary

**Training is now complete (100/100 epochs, finished normally without
early-stopping) — see [final_results.md](final_results.md) for the
converged test-set numbers.** This file was written mid-training (last
updated at epoch 66/100) and is kept as-is below for the trend history;
the milestones table now includes the final rows too. Source of truth for
the full per-epoch history is always `../results_256/metrics.csv`.

## Run configuration

- Config: `../configs/config_256.yaml`
- Model: vanilla UNet (`segmentation_models_pytorch.Unet`), resnet34
  encoder, ImageNet-pretrained
- Classes: 19 (see `../src/class_mapping.py`) — all UAVScenes paper
  classes kept separate, unnamed/reserved raw ids merged into Background
- Resolution: 256x256, `cache_in_ram: true`
- Loss: 0.5 Dice + 0.5 class-weighted CE (median-frequency, clip 10x)
- Optimizer: AdamW, lr 3e-4, weight_decay 1e-4, cosine schedule, 3-epoch warmup
- Batch size: 32, epochs: 100 (early_stop_patience: 15), seed: 42

## Milestones (val_mIoU, excl. Background)

| Epoch | val_mIoU | val_loss | Note |
|---|---|---|---|
| 1 | 0.1631 | 0.884 | |
| 5 | 0.6598 | 0.231 | |
| 10 | 0.7301 | 0.227 | |
| 20 | 0.7215 | 0.217 | dip (not monotonic epoch-to-epoch, expected) |
| 30 | 0.7791 | 0.209 | |
| 40 | 0.7949 | 0.218 | |
| 45 | 0.7975 | 0.234 | **first full class-wise breakdown taken here** |
| 50 | 0.8027 | 0.246 | |
| 60 | 0.8092 | 0.264 | |
| 66 | 0.8149 | 0.285 | second class-wise breakdown taken here (superseded) |
| 80 | 0.8188 | 0.336 | |
| 90 | 0.8197 | 0.359 | |
| **98** | **0.8201** | 0.359 | **final best checkpoint** (used for test evaluation) |
| 100 | 0.8199 | 0.361 | run end (not best; best.pt stayed at epoch 98) |

Full per-epoch history: `../results_256/metrics.csv`.
Live curves: `tensorboard --logdir ../results_256/tensorboard`.

## Pattern worth noting

val_loss has been drifting *up* since around epoch 30 (0.209 -> 0.285)
while val_mIoU keeps climbing. This divergence is a known characteristic
of combined Dice+CE training — Dice is IoU-shaped and keeps improving
even as the CE term's calibration loosens (predictions get more confident
in ways that raise per-pixel CE loss without hurting overlap-based IoU).
Not itself a red flag, but worth watching if val_mIoU ever plateaus or
reverses while val_loss keeps rising.

## What happened after this file was first written

- Training ran all the way to epoch 100 without early-stopping — val_mIoU
  kept finding small new bests right up to epoch 99, so the 15-epoch
  no-improvement patience never triggered.
- `../src/evaluate.py` was run automatically on the held-out **test**
  split as soon as training stopped. See
  [final_results.md](final_results.md) for that converged, final number
  (test mIoU 0.8055) — that's the fair comparison point, not the val
  snapshots in this file or in `classwise_val_iou_results.md`.
