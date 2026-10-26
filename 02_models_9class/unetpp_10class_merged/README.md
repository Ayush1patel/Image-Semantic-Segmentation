# UAVScenes UNet++ Semantic Segmentation

UNet++ (via `segmentation_models_pytorch`) trained on the UAVScenes camera
dataset, with the original 19 annotated classes merged down to 10 classes.

## Data layout expected

```
D:\Saqib\Multimodal_segmentation\UAVScenes\interval5_CAM_label\<sequence>\interval5_CAM_label_id\*.png   (raw class-id, uint8)
D:\Saqib\Multimodal_segmentation\UAVScenes\interval5_CAM_LIDAR\<sequence>\interval5_CAM\*.jpg              (RGB camera image)
```
Paths are set in [src/splits.py](src/splits.py).

## Class merging

Raw UAVScenes labels use 26 ids (see the dataset's `cmap.py`), 19 of which are
named in the paper (1 background + 16 static + 2 dynamic). We merge these to
10 classes for training — see [src/class_mapping.py](src/class_mapping.py)
for the full raw-id -> merged-id table and rationale:

| id | class | merged from |
|----|-------|-------------|
| 0 | Background | background, unused/reserved raw ids, solar_board (removed), umbrella (removed) |
| 1 | Building | roof, transparent_roof |
| 2 | Road | dirt_motor_road, paved_motor_road, paved_walk, car_park |
| 3 | Airstrip | airstrip |
| 4 | Water | river, pool |
| 5 | Bridge | bridge |
| 6 | Container | container |
| 7 | Traffic_Barrier | traffic_barrier |
| 8 | Vegetation | green_field, wild_field |
| 9 | Vehicle | sedan, truck |

## Split (fixed, by sequence)

- train: AMtown01, AMvalley01, HKairport01, HKisland01, HKairport_GNSS01, HKisland_GNSS01, HKairport_GNSS02, HKisland_GNSS02 (11,825 images)
- val: AMtown02, AMvalley02, HKairport02, HKisland02 (4,207 images)
- test: AMtown03, AMvalley03, HKairport03, HKisland03 (3,285 images)

## Setup

```powershell
cd E:\Multimodal\NamanAyush\02_models_9class\unetpp_10class_merged
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Usage

1. Build the train/val/test index + class-frequency stats (one-time, scans all folders):
   ```powershell
   python scripts/build_index.py
   ```
2. Train:
   ```powershell
   python src/train.py --config configs/config.yaml
   ```
   Prints train loss, val loss, and val mIoU every epoch. TensorBoard logs go to
   `outputs/tensorboard`; checkpoints to `checkpoints/best.pt` / `checkpoints/last.pt`.
   A fixed set of 10 validation images is re-rendered (Image | Ground Truth |
   Prediction) after every epoch to `outputs/progress/epoch_XXX.png` so you can
   watch qualitative progress as training goes.
   Resume an interrupted run with `--resume checkpoints/last.pt`.
3. Evaluate on the held-out **test** split (per-class IoU, mIoU, confusion
   matrix, sample visualizations) — only run this once, after training is
   done, to get an unbiased final number:
   ```powershell
   python src/evaluate.py --checkpoint checkpoints/best.pt
   ```

### Faster config, full 50 epochs (~2.2 hour run)

`configs/config_fast.yaml` trades some resolution for speed vs.
`config.yaml` while still running the full 50 epochs: 288x352, all images
decoded once and cached in RAM (`data.cache_in_ram: true`, see
[src/cached_dataset.py](src/cached_dataset.py)) so each epoch is pure
augmentation + GPU compute with no disk I/O, batch size 32. Benchmarked
end-to-end on this machine at ~7 min cache build + ~127 min training =
**~134 min (~2.23h) total**, well under a 2.5h cap, peak VRAM ~7.3/20GB.
Run it the same way:
```powershell
python src/train.py --config configs/config_fast.yaml
python src/evaluate.py --checkpoint checkpoints_fast/best.pt --config configs/config_fast.yaml
```
This is faster than the full-res run but not a replacement for it — at
288x352 the rarest classes (Vehicle, Bridge, Traffic_Barrier, Container;
all <0.4% of pixels even at native resolution) still lose a lot of detail.
Use `configs/config.yaml` (960x800, ~16h) when you want the best achievable
mIoU with no time constraint.

## Notes on getting good mIoU

- Loss = 0.5 Dice + 0.5 class-weighted Cross-Entropy (median-frequency
  balancing from `data_index/train_class_stats.json`) to handle the heavy
  class imbalance (Background/Road dominate; Vehicle/Bridge/Container are rare).
- Default encoder is `resnet34` (ImageNet-pretrained) for fast iteration.
  Once the pipeline is validated, bump `model.encoder_name` in
  `configs/config.yaml` to `resnet50` or `timm-efficientnet-b4` (larger
  capacity) for a further mIoU boost if training time/VRAM allow.
- Training resolution is 960x800 (config `data.height`/`data.width`),
  matching the native 2448x2048 aspect ratio; benchmarked at batch size 8
  to ~13.4/20GB peak VRAM and ~19 min/epoch on an RTX 4000 Ada. 50 epochs
  with early stopping (patience 15) is the default budget (~16h worst case).
  Raise resolution/batch if VRAM allows — small structures (Traffic_Barrier,
  Container) benefit from resolution.
- Augmentation includes scale jitter + crop, flip, and color jitter to
  fight overfitting given the correlated frames within each flight sequence.
