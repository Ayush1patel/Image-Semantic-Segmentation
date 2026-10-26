# UAVScenes Vanilla UNet — 19-class, 256x256, CE + Lovász-Softmax + Building weight override + Building-aware sampling

Sibling of [`../unet_lovasz_ce`](../unet_lovasz_ce)
(the current best checkpoint), identical in every respect except two
deliberate additions implementing your mentor's suggestion to "adjust or
increase the weight and the sampling of the required classes":

| | `..._ce_lovasz` | `..._ce_lovasz_building_boost` (this project) |
|---|---|---|
| Loss | 0.5 Lovász-Softmax + 0.5 class-weighted CE | unchanged |
| Building's CE weight | auto (median-frequency) ≈ 0.074 | **overridden to 2.0** |
| Training sampler | plain uniform shuffle | **WeightedRandomSampler**, Building-aware |
| Epoch length | 11,825 images (1 pass) | **1.3× — 15,372 samples/epoch** |
| Everything else | — | unchanged (architecture, split, optimizer, schedule, seed) |

## Why two changes, not one

These act on different parts of training and are complementary, not
alternatives — see `../../04_building_iou_improvement/` (error audit) for the
supporting analysis. Short version:

- **Loss weight** changes how much a Building pixel's error counts
  *once the model sees it*. The auto median-frequency formula weights by
  scarcity, not task importance — Building is common, so its auto weight
  (~0.074) is actively *below* neutral, even though it's the class we
  care about most. `class_weight_override: {Building: 2.0}` in
  `configs/config_256.yaml` directly overrides this (not multiplies it —
  multiplying an already-tiny number barely moves it).
- **Sampling** changes *how often* the model sees Building-rich images in
  the first place. Plain `shuffle=True` draws every image exactly once
  per epoch regardless of content. The sampler (`src/train.py`) assigns
  each training image a weight = `1 + building_alpha * (Building pixel
  fraction in that image)`, clipped at `clip_max`, so Building-rich images
  get drawn more than once per epoch on average.

## Deliberate risk mitigations (see `src/train.py` for the mechanism)

Both levers risk starving the already-fragile rare/site-exclusive classes
(Bridge, Pool, Solar_Board, Container, Traffic_Barrier — see
`../../01_dataset_analysis/`) of training exposure. Mitigated by:

- **Soft sampling curve, not raw-proportional** — every image keeps a
  weight-1 floor; Building-free images aren't starved outright, just not
  specially boosted.
- **Clipped at 3.0** — no single image can dominate the sampler.
- **Epoch length increased (1.3×), not just reweighted at the same
  length** — so other images aren't crowded out of their normal exposure
  to make room for Building-rich ones; this costs more wall-clock time
  per epoch, which is why it was only turned on after confirming the time
  budget allowed it.
- **Verification plan**: after training, check the *full* 19-class table
  (`results_256/test_eval/results.json`), not just Building — specifically
  watch the five classes above for regression, the same way every other
  comparison in this project has been checked.

Sampler weight distribution on the real training set, and the time-budget
math, are in `scripts/sanity_check_sampler.py`'s output — run it before
trusting the config blindly if the dataset ever changes.

## Class mapping, split

Identical to `../unet_lovasz_ce` — see
[`src/class_mapping.py`](src/class_mapping.py) and
[`src/splits.py`](src/splits.py). Same known spatial-leakage caveat
applies (see `../../01_dataset_analysis/`).

## Setup

Same shared venv as the sibling projects:
```powershell
& "E:\Multimodal\NamanAyush\\.venv\Scripts\python.exe" ...
```
`data_index/` copied verbatim (split unchanged) — training can start
directly, no need to re-run `scripts/build_index.py` unless the dataset
changes.

## Usage

```powershell
cd E:\Multimodal\NamanAyush\03_models_19class\unet_lovasz_ce_building_boost
& "E:\Multimodal\NamanAyush\\.venv\Scripts\python.exe" src\train.py --config configs\config_256.yaml
& "E:\Multimodal\NamanAyush\\.venv\Scripts\python.exe" src\evaluate.py --config configs\config_256.yaml
```
Results land in `results_256/` and `results_256/test_eval/`, same format
as every other project in this workspace.

## Notes

- `class_weight_override` and `sampler` are both config-driven (see
  `configs/config_256.yaml`), not hardcoded, specifically so the
  Building values can be tuned, or a `Road` entry added later for Phase 3,
  without touching code.
- Checkpoints from this project are compatible with the other 19-class
  UNet projects (identical architecture, `NUM_CLASSES=19`), not with the
  UPerNet or 9-class projects.
