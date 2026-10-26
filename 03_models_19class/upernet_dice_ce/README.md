# UAVScenes UPerNet + ResNet34 — 19-class, 256x256

Architecture-only comparison against `unet_dice_ce`. Same
dataset, split, class taxonomy (19 classes), input resolution, optimizer,
batch size, and loss — the **only** difference is the decoder:
[UPerNet](https://arxiv.org/abs/1807.10221) (Xiao et al. 2018) instead of a
vanilla UNet decoder. UPerNet+ResNet is the exact decoder/backbone pairing
used in the UAVScenes paper's own image-segmentation benchmark (Table 3),
so this run is also the closest architecture match to that paper baseline
available in this workspace (see `../unet_dice_ce/analysis/paper_comparison.md`
for why the *training recipe* still differs from the paper's — that
analysis applies here too, since input size/optimizer/loss are matched to
the sibling UNet project, not to the paper's 224x224-crop/plain-Adam recipe).

## What's identical to `unet_dice_ce`

- Dataset, sequence split, `data_index/` (copied as-is — same images, same
  labels, same 19-class remap, so no need to rebuild)
- Class taxonomy: 19 named UAVScenes classes kept separate; only the 7
  unnamed/reserved raw ids folded into Background (`src/class_mapping.py`)
- Resolution: 256×256, full-frame resize (no crop)
- Encoder: ResNet-34, ImageNet-pretrained
- Optimizer: AdamW, lr 3e-4, weight decay 1e-4, cosine schedule w/ 3-epoch warmup
- Batch size: 32, epochs: 100 (early-stop patience 15)
- Loss: 0.5 Dice + 0.5 class-weighted Cross-Entropy (median-frequency
  weighting from `data_index/train_class_stats.json`, clipped at 10x)
- Background: trained as a real class (`IGNORE_INDEX=None`), excluded only
  from the headline mean metrics

## What's different

- **Decoder**: `smp.UPerNet` instead of `smp.Unet` (`src/model.py`) — pyramid
  pooling module (PPM) on the encoder's deepest stage, fused top-down
  (FPN-style) with intermediate encoder features into one map before the
  classifier head. No UNet-style same-resolution skip concatenation.
- **Extra per-epoch logging/analysis** (not present in the sibling project):
  - `results_256/metrics.csv` also logs `val_pixel_accuracy`,
    `val_mean_precision`, `val_mean_recall`, `val_mean_f1` every epoch
    (sibling project only logs `val_mIoU`).
  - `results_256/per_class_iou.csv` — one row per epoch, one column per
    class — tracks every class's val IoU trajectory over training.
  - TensorBoard also gets a `val_per_class_iou/<class>` scalar per class,
    per epoch, in addition to the aggregate scalars.
  - `analysis/scripts/plot_training_curves.py` — turns the above CSVs into
    PNGs: loss curves, val mIoU + pixel accuracy, val precision/recall/F1,
    LR schedule sanity check, epoch-time/throughput, per-class IoU (all 19
    classes, plus a focused plot on the classes that were weakest in the
    sibling UNet run — Truck, Sedan, Traffic_Barrier, Umbrella, Bridge),
    and a single combined dashboard image. Run after training finishes (or
    at any point mid-training to see progress so far).

## Class mapping

Same as `unet_dice_ce` — see [src/class_mapping.py](src/class_mapping.py)
for the full raw-id table.

## Split (identical to the sibling projects, by sequence)

- train: AMtown01, AMvalley01, HKairport01, HKisland01, HKairport_GNSS01, HKisland_GNSS01, HKairport_GNSS02, HKisland_GNSS02 (11,825 images)
- val: AMtown02, AMvalley02, HKairport02, HKisland02 (4,207 images)
- test: AMtown03, AMvalley03, HKairport03, HKisland03 (3,285 images)

## Setup

No dedicated `.venv` — runs against the shared Python environment already
on PATH (`unetpp_10class_merged\.venv`, which has torch 2.5.1+cu121,
`segmentation-models-pytorch` (with its `UPerNet` decoder), and every other
dependency from `requirements.txt` already installed and CUDA-verified).
If that environment isn't on PATH, activate it directly:

```powershell
E:\Multimodal\NamanAyush\02_models_9class\unetpp_10class_merged\.venv\Scripts\Activate.ps1
```

`data_index/` (train/val/test index + `train_class_stats.json`) is already
built (copied from the sibling project — same underlying data) — training
can start directly, no need to run `scripts/build_index.py`.

## Usage

1. Train:
   ```powershell
   cd E:\Multimodal\NamanAyush\upernet_dice_ce
   python src/train.py --config configs/config_256.yaml
   ```
   Prints train loss, val loss, val mIoU, val pixel accuracy, and val F1
   every epoch. All results (config snapshot, class_info.json, metrics.csv,
   per_class_iou.csv, checkpoints/, per-epoch progress snapshots,
   tensorboard logs) saved under `results_256/`.

2. Evaluate on the held-out test split:
   ```powershell
   python src/evaluate.py --config configs/config_256.yaml
   ```
   Saves per-class IoU/precision/recall/F1, mIoU, pixel accuracy, test
   loss, a confusion matrix heatmap, and sample predictions to
   `results_256/test_eval/`.

3. Generate the extra analysis plots (any time after a few epochs exist,
   or once training is done):
   ```powershell
   python analysis/scripts/plot_training_curves.py
   ```
   Saves PNGs to `analysis/images/`.

If the raw dataset ever moves or `data_index/` needs rebuilding:
```powershell
python scripts/build_index.py
```

## Notes

- Checkpoints from this project are NOT compatible with any other project
  in this workspace — different decoder architecture (state dict keys
  won't match `unet_dice_ce`'s vanilla-UNet checkpoints even
  though `NUM_CLASSES` is the same).
- At 256×256 the whole train+val set (~16k images) fits easily in RAM
  (~4.2GB), so `cache_in_ram: true` gives fast epochs with no disk I/O
  after the one-time cache-build pass at the start of training.
