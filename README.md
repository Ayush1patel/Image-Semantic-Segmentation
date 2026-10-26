# UAVScenes Semantic Segmentation

Semantic segmentation of UAV camera imagery from the
[UAVScenes](UAVScenesPaper.pdf) dataset. The project studies how three design
choices affect segmentation quality, and then targets the weakest important
class (Building) with dedicated error analysis and inference-time fixes:

1. **Class taxonomy** — merged label sets (8, 9 and 10 classes) versus the full
   19-class label set.
2. **Loss function** — Dice + class-weighted cross-entropy versus
   Lovász-Softmax + class-weighted cross-entropy.
3. **Architecture** — UNet, UNet++ and UPerNet.

## Dataset

- 20 UAVScenes sequences (about 24,000 labelled camera frames, `interval5`
  sampling) from four sites: AMtown, AMvalley, HKairport and HKisland.
- Raw labels use 26 ids, 19 of which are named in the paper (1 background,
  16 static classes, 2 dynamic classes: Sedan and Truck). The remaining ids are
  unnamed or reserved and are folded into Background.
- The split is **by sequence**, never by frame, so no scene appears in more
  than one split:

  | Split | Sequences | Images |
  |---|---|---|
  | Train | AMtown01, AMvalley01, HKairport01, HKisland01, HKairport_GNSS01/02, HKisland_GNSS01/02 | 11,825 |
  | Validation | AMtown02, AMvalley02, HKairport02, HKisland02 | 4,207 |
  | Test | AMtown03, AMvalley03, HKairport03, HKisland03 | 3,285 |

- The class distribution is heavily imbalanced (Green_Field and Wild_Field
  cover about 64% of pixels, while Sedan, Transparent_Roof and Truck are well
  under 1%), which motivates the median-frequency class weights used in the
  loss. See `01_dataset_analysis/class_distribution_report/`.

## Repository structure

```
01_dataset_analysis/            Class-distribution scan, plots and report for the full dataset
    scripts/                        scan, summarise and plot scripts
    outputs/                        per-image and aggregated class-count CSVs
    class_distribution_report/      written report and figures

02_models_9class/               Merged-taxonomy models
    unetpp_10class_merged/              UNet++, 19 raw classes merged to 10
    unet_8class_ignore_background/      UNet, 8 classes, background ignored (128px and 512px runs)
    unet_9class_background_lidar_iou/   UNet, 9 classes (background as a class), LiDAR-restricted IoU
    unet_9class_baseline/               UNet, 9 classes, 256px (baseline for the 9 vs 19-class comparison)

03_models_19class/              Native 19-class taxonomy, all at 256x256
    unet_dice_ce/                       UNet, Dice + CE loss (19-class baseline)
    unet_lovasz_ce/                     UNet, Lovász-Softmax + CE loss
    unet_lovasz_ce_building_boost/      UNet, Lovász + CE with Building-focused sampling and weighting
    upernet_dice_ce/                    UPerNet, Dice + CE loss (architecture comparison)

04_building_iou_improvement/    Building-class error analysis and fixes (uses unet_lovasz_ce)
    01_error_audit/                     Per-image error audit and worst-case visualisations
    02_postprocessing/                  Morphological / connected-component mask post-processing
    03_tta_threshold_sweep/             Test-time augmentation and decision-threshold sweep
```

Every model folder has the same internal layout:

| Path | Contents |
|---|---|
| `configs/` | YAML hyperparameters (resolution, batch size, epochs, loss weights) |
| `data_index/` | Train/val/test index files and `train_class_stats.json` (class pixel frequencies) |
| `src/` | `train.py`, `evaluate.py`, `model.py`, `losses.py`, `metrics.py`, `dataset.py`, `transforms.py`, `class_mapping.py` |
| `scripts/` | Helper scripts, e.g. `build_index.py` to rebuild the split index |
| `results*/` | Checkpoints, training curves, TensorBoard logs and `test_eval/` metrics |
| `README.md` | Setup, class mapping and run instructions specific to that model |

## Experiments and results

Test-set results on the held-out sequences. mIoU is not strictly comparable
across the 9-class and 19-class rows: the class sets differ, and some runs
exclude Background from the mean (see each model's README for its convention).

### Taxonomy and architecture

| Model | Classes | Resolution | Loss | Test mIoU | Pixel acc. |
|---|---|---|---|---|---|
| `unet_8class_ignore_background` | 8 | 128 | Dice + CE | 0.863 | 0.987 |
| `unet_8class_ignore_background` | 8 | 512 | Dice + CE | 0.928 | 0.994 |
| `unetpp_10class_merged` | 10 | 288x352 | Dice + CE | 0.828 | 0.954 |
| `unet_9class_baseline` | 9 | 256 | Dice + CE | 0.807 | 0.951 |
| `unet_dice_ce` | 19 | 256 | Dice + CE | 0.805 | 0.943 |
| `upernet_dice_ce` | 19 | 256 | Dice + CE | 0.815 | 0.949 |

`unet_9class_baseline` and `unet_dice_ce` use identical hyperparameters, so the
difference between them isolates the effect of the class taxonomy.
`upernet_dice_ce` is matched to `unet_dice_ce` for an architecture-only
comparison.

### Loss function and Building focus (19-class UNet, 256px)

| Model | Change | Test mIoU |
|---|---|---|
| `unet_dice_ce` | baseline: 0.5 Dice + 0.5 weighted CE | 0.805 |
| `unet_lovasz_ce` | Lovász-Softmax replaces Dice, aimed at Building false positives | 0.808 |
| `unet_lovasz_ce_building_boost` | plus higher Building CE weight and a Building-aware weighted sampler | 0.810 |

### Building IoU improvement (on `unet_lovasz_ce`, no retraining)

| Stage | Building IoU |
|---|---|
| Plain argmax prediction | 0.850 |
| Morphological post-processing | 0.835 (worse; recall dropped) |
| Horizontal-flip TTA + validation-tuned threshold | 0.863 |

The audit in `01_error_audit` shows the Building error is spread across most
images rather than concentrated in a few outliers. The post-processing
experiment is a documented negative result: it removed correct Building pixels.
The final configuration (horizontal-flip TTA, Gaussian smoothing sigma 0.5,
probability threshold 0.6) was chosen on the validation split only, then
applied to the test split; it improved 2,091 of 3,285 test images and worsened
199.

## Reproducing the results

All projects share one Python environment (PyTorch with CUDA,
`segmentation_models_pytorch`, albumentations, OpenCV, PyYAML, TensorBoard).
Each model folder lists its dependencies in `requirements.txt`.

```powershell
# from inside a model folder, e.g. 03_models_19class/unet_dice_ce
pip install -r requirements.txt
python scripts/build_index.py                        # only if data_index/ needs rebuilding
python src/train.py    --config configs/config_256.yaml
python src/evaluate.py --config configs/config_256.yaml
```

Dataset locations are set in each project's `src/splits.py`; edit them to point
at your local copy of UAVScenes. The Building improvement scripts in
`04_building_iou_improvement/` read the trained `unet_lovasz_ce` checkpoint and
are run from their own folders.

## Where to look first

- **Overview of the label set and mapping:** `03_models_19class/unet_dice_ce/README.md`
- **Why Lovász loss was tried:** `03_models_19class/unet_lovasz_ce/README.md`
- **UPerNet design notes:** `03_models_19class/upernet_dice_ce/TECHNICAL_NOTES.md`
- **Dataset imbalance:** `01_dataset_analysis/class_distribution_report/class_distribution_report.md`
- **Building error analysis and fixes:** `04_building_iou_improvement/01_error_audit/audit_summary.md`
