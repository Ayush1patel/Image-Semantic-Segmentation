# UAVScenes Vanilla UNet — 19-class, 256x256, CE + Lovasz-Softmax loss

Sibling of [`../unet_dice_ce`](../unet_dice_ce),
identical in every respect **except the loss function**:

| | `unet_dice_ce` | `unet_lovasz_ce` (this project) |
|---|---|---|
| Loss | 0.5 Dice + 0.5 class-weighted CE | 0.5 Lovasz-Softmax + 0.5 class-weighted CE |
| Architecture | vanilla `segmentation_models_pytorch.Unet`, resnet34/imagenet encoder | unchanged |
| Optimizer / schedule | AdamW, cosine w/ 3-epoch warmup | unchanged |
| Batch size / epochs / seed | 32 / 100 / 42 | unchanged |
| Class taxonomy | 19 classes (see below) | unchanged |
| Resolution | 256x256 | unchanged |
| Train/val/test split | by sequence (below) | unchanged — `data_index/*.json` copied verbatim, not rebuilt |

Nothing else was changed. `data_index/train_class_stats.json` (used for
median-frequency class weighting in the CE term) is the same file as the
sibling project's — the split didn't change, so the class-pixel-frequency
stats it was built from don't either.

## Why this variant exists

Goal: reduce false positives and raise IoU specifically on Building and
its visually-confusable neighbors (Container, Transparent_Roof, Car_Park,
Paved_Walk) — the sibling run's confusion matrix already showed
over-prediction on these ("recall leads precision throughout"). Dice is a
smooth, per-pixel-averaged overlap score; it can't distinguish "a solid
false-positive blob" from "the same number of scattered boundary
mistakes." Lovasz-Softmax is a direct convex surrogate of the Jaccard
index whose gradient is concentrated on whichever pixels are actually
costing IoU right now, so it should specifically shrink the blob-shaped
false positives Dice/CE under-penalize.

Lovasz is combined with CE rather than used alone deliberately: see the
docstring in [`src/losses.py`](src/losses.py) for the full reasoning —
short version, Lovasz's gradient needs a non-trivial starting prediction
to rank errors against, warms up poorly from random init on its own, and
gives zero gradient to any class entirely absent from a batch. Several
trainable classes in this dataset (Bridge, Pool, Solar_Board, Container,
Traffic_Barrier, ...) are absent from most sequences entirely (see
`../../01_dataset_analysis/`), so CE stays in the loss to keep those classes
learnable throughout training while Lovasz sharpens the IoU-costing
errors on top.

## Class mapping

Identical to the sibling project — see
[`src/class_mapping.py`](src/class_mapping.py) for the exact raw-id table.

## Split (identical to the sibling project, by sequence)

- train: AMtown01, AMvalley01, HKairport01, HKisland01, HKairport_GNSS01, HKisland_GNSS01, HKairport_GNSS02, HKisland_GNSS02 (11,825 images)
- val: AMtown02, AMvalley02, HKairport02, HKisland02 (4,207 images)
- test: AMtown03, AMvalley03, HKairport03, HKisland03 (3,285 images)

**Known caveat, carried over from the sibling project unchanged:** this
split assigns whole flight runs to train/val/test. A GPS-based check
(`../../01_dataset_analysis/scripts/check_spatial_overlap.py`) found the
01/02/03 runs at each site are the same flight path re-flown, not
independent surveys — median distance from a val/test frame to its
nearest train frame is well under 1 meter at every site. Results from
this project inherit that same leakage until a spatially-separated split
replaces it; see `../../01_dataset_analysis/` for the full analysis and
planned fix.

## Setup

Same shared environment as the sibling project — no dedicated `.venv`
here either. Activate the virtual environment kept in
`02_models_9class\unetpp_10class_merged`:

```powershell
E:\Multimodal\NamanAyush\02_models_9class\unetpp_10class_merged\.venv\Scripts\Activate.ps1
```

`requirements.txt` is unchanged; `segmentation_models_pytorch>=0.3.4`
already provides `LovaszLoss` (`src/losses.py`), no new dependency needed.

`data_index/` is already built (copied from the sibling project — the
split is identical) — training can start directly, no need to re-run
`scripts/build_index.py`.

## Usage

1. Train:
   ```powershell
   cd E:\Multimodal\NamanAyush\03_models_19class\unet_lovasz_ce
   python src/train.py --config configs/config_256.yaml
   ```
   All results (config snapshot, class_info.json, metrics.csv,
   checkpoints, per-epoch progress snapshots, tensorboard logs) saved
   under `results_256/` — a fresh directory, does not touch the sibling
   project's `results_256/`.
2. Evaluate on the held-out test split:
   ```powershell
   python src/evaluate.py --config configs/config_256.yaml
   ```
   Saves per-class IoU/precision/recall/F1, mIoU, pixel accuracy, test
   loss, a confusion matrix heatmap, and sample predictions to
   `results_256/test_eval/` — directly comparable to the sibling
   project's `results_256/test_eval/` for the confusion-matrix diff.

If the raw dataset ever moves or `data_index/` needs rebuilding:
```powershell
python scripts/build_index.py
```

## Notes

- Loss = 0.5 Lovasz-Softmax + 0.5 class-weighted Cross-Entropy
  (`src/losses.py`, class `CELovaszLoss`); class weights come from
  `data_index/train_class_stats.json` (median-frequency balancing,
  clipped at 10x) — identical weighting to the sibling project.
- `ConfusionMatrix` (`src/metrics.py`) excludes Background from the mean
  IoU/precision/recall/F1 (it's still trained on and still reported
  per-class) — same convention as the sibling project.
- At 256x256 the whole train+val set (~16k images) fits easily in RAM
  (~4.2GB), so `cache_in_ram: true` gives fast epochs with no disk I/O
  after the one-time cache-build pass at the start of training.
- Checkpoints from this project are compatible with the sibling
  `unet_dice_ce` project (identical architecture and
  `NUM_CLASSES=19`) but not with any other project in this workspace.
