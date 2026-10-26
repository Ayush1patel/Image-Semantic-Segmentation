# UAVScenes Vanilla UNet Semantic Segmentation

Plain UNet (`segmentation_models_pytorch.Unet`, no nested/dense skip
pathways) trained on the same UAVScenes camera dataset and train/val/test
split as the sibling `unetpp_10class_merged` project, with three deliberate
differences:

1. **Class taxonomy** — from `class distribution.txt` (project root), not
   the 10-class UNet++ scheme. 8 trainable classes; background,
   solar_board, umbrella, and unnamed/reserved raw ids are a genuine
   `ignore_index` (255) — excluded from both the loss and the metrics,
   never predicted, never scored. Note Airstrip is folded into Road here
   (it was its own class in the UNet++ project).
2. **Resolution** — fixed at 128x128 for training, evaluation, and all
   reported metrics.
3. **Model** — vanilla UNet instead of UNet++.

## Class mapping

| PMNet ID | Class | Raw UAVScenes ids merged into it |
|---|---|---|
| 0 | Building | roof, transparent_roof |
| 1 | Road | dirt_motor_road, paved_motor_road, airstrip, car_park, paved_walk |
| 2 | Water | river, pool |
| 3 | Vegetation | green_field, wild_field |
| 4 | Bridge | bridge |
| 5 | Container | container |
| 6 | Traffic_Barrier | traffic_barrier |
| 7 | Vehicle | sedan, truck |
| ignore_index=255 | — | background, solar_board, umbrella, + unnamed/reserved raw ids |

See [src/class_mapping.py](src/class_mapping.py) for the exact raw-id table.

## Split (identical to unetpp_10class_merged, by sequence)

- train: AMtown01, AMvalley01, HKairport01, HKisland01, HKairport_GNSS01, HKisland_GNSS01, HKairport_GNSS02, HKisland_GNSS02 (11,825 images)
- val: AMtown02, AMvalley02, HKairport02, HKisland02 (4,207 images)
- test: AMtown03, AMvalley03, HKairport03, HKisland03 (3,285 images)

## Setup

```powershell
cd E:\Multimodal\NamanAyush\02_models_9class\unet_8class_ignore_background
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Usage

1. Build the train/val/test index + class-frequency stats (scans the same
   raw dataset folders on D:\, one-time):
   ```powershell
   python scripts/build_index.py
   ```
2. Train:
   ```powershell
   python src/train.py --config configs/config.yaml
   ```
   Prints train loss, val loss, and val mIoU every epoch. All results
   (config snapshot, class_info.json, metrics.csv, checkpoints, per-epoch
   progress snapshots, tensorboard logs) saved under `results/`.
3. Evaluate on the held-out test split:
   ```powershell
   python src/evaluate.py --config configs/config.yaml
   ```
   Saves per-class IoU/precision/recall/F1, mIoU, pixel accuracy, test
   loss, a confusion matrix heatmap, and sample predictions to
   `results/test_eval/`.

## Notes

- Loss = 0.5 Dice + 0.5 class-weighted Cross-Entropy, both `ignore_index`-aware
  (`src/losses.py`), so ignored pixels never contribute gradient.
- `ConfusionMatrix` (`src/metrics.py`) excludes `ignore_index` pixels
  entirely from IoU/precision/recall/F1/pixel-accuracy — they are neither
  counted as correct nor incorrect.
- At 128x128 the whole train+val set fits trivially in RAM (~1GB), so
  `cache_in_ram: true` gives very fast epochs with no disk I/O.
- Since this is intentionally a fixed-128x128 / vanilla-UNet comparison
  point against the UNet++ project, treat any mIoU difference between the
  two as reflecting *both* the resolution drop and the architecture
  change, not architecture alone.
