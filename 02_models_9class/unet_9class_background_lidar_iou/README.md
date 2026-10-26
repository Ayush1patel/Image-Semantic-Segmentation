# UAVScenes Vanilla UNet — 9 classes (+Background) + LiDAR-restricted IoU

Fork of the sibling `unet_8class_ignore_background` project. Same vanilla UNet
(`segmentation_models_pytorch.Unet`, ResNet-34 ImageNet encoder), same
dataset, same train/val/test split, same hyperparameters — with two
deliberate changes:

1. **Background is a real class.** The 8 named classes keep their exact ids
   and order; `Background` is appended as **class 8**. Nothing is ignored
   any more.
2. **LiDAR-restricted evaluation.** Alongside the usual full-frame metrics,
   `src/evaluate.py` computes IoU over only the pixels that a LiDAR point
   projects onto.

`unet_8class_ignore_background` is untouched and still holds the 8-class
`ignore_index` taxonomy and its trained results — the two projects are
independent and can be compared side by side.

## 1. Class mapping (9 classes)

| ID | Class | Raw UAVScenes ids merged into it |
|---|---|---|
| 0 | Building | roof, transparent_roof |
| 1 | Road | dirt_motor_road, paved_motor_road, airstrip, car_park, paved_walk |
| 2 | Water | river, pool |
| 3 | Vegetation | green_field, wild_field |
| 4 | Bridge | bridge |
| 5 | Container | container |
| 6 | Traffic_Barrier | traffic_barrier |
| 7 | Vehicle | sedan, truck |
| **8** | **Background** | **background, solar_board, umbrella, + unnamed/reserved raw ids** |

Every raw id maps to a real class, so `IGNORE_INDEX` is `None` and **no
pixel is ignored** by the loss or the metrics. Both loss terms and
`ConfusionMatrix` still accept an `ignore_index`; it is simply unset here.

Background is a large class, not a rounding error — measured over the full
train split:

| Class | pixel freq | loss weight | | Class | pixel freq | loss weight |
|---|---|---|---|---|---|---|
| Building | 0.0280 | 1.000 | | Container | 0.0037 | 7.504 |
| Road | 0.0590 | 0.475 | | Traffic_Barrier | 0.0015 | 10.0 |
| Water | 0.0964 | 0.291 | | Vehicle | 0.0014 | 10.0 |
| Vegetation | 0.6592 | 0.043 | | **Background** | **0.1507** | **0.186** |
| Bridge | 0.0001 | 10.0 | | | | |

Because the output head is 9-wide, **checkpoints from
`unet_8class_ignore_background` cannot be loaded here.** This project trains from
scratch.

See [src/class_mapping.py](src/class_mapping.py) for the exact raw-id table.

## 2. LiDAR-restricted evaluation

`src/evaluate.py` reports three views of the *same* predictions:

| View | Pixels scored | Ground truth |
|---|---|---|
| **(1) Full frame** | all | camera label PNG (`interval5_CAM_label`) |
| **(2) LiDAR pixels** | only pixels a LiDAR return projects onto | camera label PNG |
| **(3) LiDAR points** | same pixels as (2) | LiDAR point label (`interval5_LIDAR_label`) |

(2) is the headline addition: identical model and ground truth as (1), just
restricted support, so the two mIoUs are directly comparable and the gap
isolates how the model does where the point cloud has coverage.

**Read (3) with care.** The two ground truths do not agree with each other:
over the masked pixels of a 41-frame sample of the test split, camera label
vs LiDAR point label scores 95.1% pixel agreement but only **0.69 mIoU**
between them — they diverge mostly on the rare classes. So (3) has a hard
per-class ceiling well below 1.0 that has nothing to do with the model. It
is a second opinion, not a stricter version of (2).

### Projection geometry

Points are projected with `P_cam = R @ P_lidar + t` followed by the pinhole
+ Brown-Conrady distortion model, using the per-scene camera–LiDAR
calibration ([src/lidar_projection.py](src/lidar_projection.py)). That
direction was verified against the data: 99.1% of a frame's points land in
frame, and per-point LiDAR labels match the camera label PNG at the
projected pixel 99.0% of the time on non-background points (93.2% overall).
The inverse transform puts under 4% of points in frame, so the convention is
not ambiguous.

Projected coordinates are scaled straight onto the eval grid rather than
rasterized at full resolution and downsampled — at 2448x2048 a frame's
~24.5k points cover only 0.48% of pixels, so a nearest-neighbour downsample
to 512x512 would discard ~95% of them. Resulting coverage is **~8.3% of
pixels at 512x512** and ~49% at 128x128.

All 19,317 frames across the 16 split sequences have a paired point cloud
and per-point labels, so no frame is dropped from the LiDAR metrics.

## Split (identical to the sibling projects, by sequence)

- train: AMtown01, AMvalley01, HKairport01, HKisland01, HKairport_GNSS01, HKisland_GNSS01, HKairport_GNSS02, HKisland_GNSS02 (11,825 images)
- val: AMtown02, AMvalley02, HKairport02, HKisland02 (4,207 images)
- test: AMtown03, AMvalley03, HKairport03, HKisland03 (3,285 images)

## Setup

This project has no `.venv` of its own. Either reuse the sibling one:

```powershell
cd E:\Multimodal\NamanAyush\02_models_9class\unet_9class_background_lidar_iou
$py = "..\unet_8class_ignore_background\.venv\Scripts\python.exe"
```

or create a fresh one:

```powershell
cd E:\Multimodal\NamanAyush\02_models_9class\unet_9class_background_lidar_iou
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Usage

`data_index/` is **already built** for this taxonomy (index files carry the
LiDAR paths, and `train_class_stats.json` holds the 9-class histogram), so
step 1 can be skipped unless the class mapping or split changes.

1. Rebuild the index + class-frequency stats (scans the raw dataset on D:\,
   ~6 min — only needed if the class mapping or split changes):
   ```powershell
   & $py scripts\build_index.py
   ```
2. Train:
   ```powershell
   & $py src\train.py --config configs\config_512.yaml
   ```
   Results (config snapshot, class_info.json, metrics.csv, checkpoints,
   per-epoch progress snapshots, tensorboard logs) go to `results_512/`.
   Use `configs\config.yaml` for the 128x128 variant, which writes to
   `results/`.
3. Evaluate on the held-out test split:
   ```powershell
   & $py src\evaluate.py --config configs\config_512.yaml
   ```
   Writes `results.txt` / `results.json` covering all three views above,
   three confusion-matrix heatmaps
   (`confusion_matrix.png`, `_lidar_pixels.png`, `_lidar_points.png`) and
   sample visualizations — each showing image / GT / prediction on top and
   the LiDAR-restricted GT / prediction underneath.

## Notes

- Loss = 0.5 Dice + 0.5 class-weighted Cross-Entropy (`src/losses.py`).
  With `ignore_index=None` every pixel contributes gradient; CE falls back
  to torch's `-100` sentinel, which no label can take.
- The LiDAR-restricted metrics work by passing `ConfusionMatrix` only the
  masked pixels, not by ignoring the rest — so per-class support counts read
  as true pixel counts.
- `lidar_out_size` on `UAVScenesDataset` is only valid with a transform
  whose sole geometric step is a resize (i.e. `get_eval_transform`); the
  rasters bypass albumentations, so random crops/flips would desynchronize
  them. The dataset checks the resulting label shape per item.
- The LiDAR rasters are built for evaluation only — training and validation
  use the plain camera path and are unaffected.
