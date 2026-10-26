# UAVScenes Vanilla UNet — 19-class, 256x256

Same dataset, split, architecture, loss, optimizer, and training
hyperparameters as `unet_9class_baseline`, with exactly
two changes:

1. **Class taxonomy** — all 19 classes named in the UAVScenes paper (1
   background + 16 static + 2 dynamic) are kept as separate trainable
   classes, instead of being merged into 9 semantic super-classes. The
   only merging that happens is folding the 7 unnamed/reserved raw ids
   (7, 8, 12, 21, 22, 23, 25) into Background. Background is trained as a
   real class (`IGNORE_INDEX=None`, every pixel scored) but excluded from
   the headline mIoU, same convention as the sibling project.
2. **Resolution** — 256x256 for training, evaluation, and all reported
   metrics (between the sibling project's 128x128 and 512x512 configs).

Everything else — model (vanilla `segmentation_models_pytorch.Unet`,
resnet34/imagenet encoder), loss (0.5 Dice + 0.5 class-weighted CE),
optimizer (AdamW, cosine schedule w/ 3-epoch warmup), batch size (32), seed
(42), and the train/val/test sequence split are unchanged from
`config_512.yaml` in the sibling project. One deliberate deviation from
that config: epochs raised from 70 to 100 (early-stop patience stays 15).

## Class mapping

| New ID | Class | Raw UAVScenes id |
|---|---|---|
| 0 | Background | 0, + unnamed/reserved 7, 8, 12, 21, 22, 23, 25 |
| 1 | Building | 1 (roof) |
| 2 | Dirt_Motor_Road | 2 |
| 3 | Paved_Motor_Road | 3 |
| 4 | River | 4 |
| 5 | Pool | 5 |
| 6 | Bridge | 6 |
| 7 | Container | 9 |
| 8 | Airstrip | 10 |
| 9 | Traffic_Barrier | 11 |
| 10 | Green_Field | 13 |
| 11 | Wild_Field | 14 |
| 12 | Solar_Board | 15 |
| 13 | Umbrella | 16 |
| 14 | Transparent_Roof | 17 |
| 15 | Car_Park | 18 |
| 16 | Paved_Walk | 19 |
| 17 | Sedan | 20 |
| 18 | Truck | 24 |

See [src/class_mapping.py](src/class_mapping.py) for the exact raw-id table.

## Split (identical to the sibling projects, by sequence)

- train: AMtown01, AMvalley01, HKairport01, HKisland01, HKairport_GNSS01, HKisland_GNSS01, HKairport_GNSS02, HKisland_GNSS02 (11,825 images)
- val: AMtown02, AMvalley02, HKairport02, HKisland02 (4,207 images)
- test: AMtown03, AMvalley03, HKairport03, HKisland03 (3,285 images)

## Setup

No dedicated `.venv` — this project (like `unet_9class_baseline`)
runs against the shared Python environment already on PATH
(`unetpp_10class_merged\.venv`, which has torch 2.5.1+cu121 and all other
dependencies from `requirements.txt` already installed and CUDA-verified).
If that environment isn't on PATH for you, either add
`unetpp_10class_merged\.venv\Scripts` to PATH or activate it directly:

```powershell
E:\Multimodal\NamanAyush\02_models_9class\unetpp_10class_merged\.venv\Scripts\Activate.ps1
```

The `data_index/` folder (train/val/test index + `train_class_stats.json`
19-class pixel-frequency stats) has already been built — training can
start directly.

## Usage

1. Train:
   ```powershell
   cd E:\Multimodal\NamanAyush\03_models_19class\unet_dice_ce
   python src/train.py --config configs/config_256.yaml
   ```
   Prints train loss, val loss, and val mIoU every epoch. All results
   (config snapshot, class_info.json, metrics.csv, checkpoints, per-epoch
   progress snapshots, tensorboard logs) saved under `results_256/`.
2. Evaluate on the held-out test split:
   ```powershell
   python src/evaluate.py --config configs/config_256.yaml
   ```
   Saves per-class IoU/precision/recall/F1, mIoU, pixel accuracy, test
   loss, a confusion matrix heatmap, and sample predictions to
   `results_256/test_eval/`.

If the raw dataset ever moves or `data_index/` needs rebuilding:
```powershell
python scripts/build_index.py
```

## Notes

- Loss = 0.5 Dice + 0.5 class-weighted Cross-Entropy (`src/losses.py`);
  class weights come from `data_index/train_class_stats.json`
  (median-frequency balancing, clipped at 10x).
- `ConfusionMatrix` (`src/metrics.py`) excludes Background from the mean
  IoU/precision/recall/F1 (it's still trained on and still reported
  per-class) — same convention as `unet_9class_baseline`.
- At 256x256 the whole train+val set (~16k images) fits easily in RAM
  (~4.2GB), so `cache_in_ram: true` gives fast epochs with no disk I/O
  after the one-time cache-build pass at the start of training.
- Checkpoints from this project are NOT compatible with any other project
  in this workspace — 19 classes is a different `NUM_CLASSES` from the
  9-class `unet_9class_baseline`, 10-class `unetpp_10class_merged`, or 8-class
  `unet_8class_ignore_background` runs.
