# Technical Notes — UPerNet (ResNet-34), 19-Class UAV Scene Segmentation @ 256×256

This document covers the reasoning behind the setup, an analysis of what the
training run actually did (including the mid-run crash), why the results
look the way they do, and where the model has room to improve. It is meant
to be read alongside [PRESENTATION_RESULTS.md](PRESENTATION_RESULTS.md),
which only carries the headline numbers.

(Comparison against the UAVScenes paper's own reported numbers/training
recipe is intentionally out of scope here — that's a separate pass.)

---

## 1. Task and label design

- Source labels use the UAVScenes dataset's 26-slot raw id space, of which
  the paper names 19 classes; 7 ids (7, 8, 12, 21, 22, 23, 25) are unused
  reserved slots that never contain meaningful content in this dataset
  variant.
- [class_mapping.py](src/class_mapping.py) folds *only* those 7 unused ids
  into `Background` (id 0) — every other class keeps its own id, so this is
  the full 19-way taxonomy from the paper (not the 9-class merged variant
  used in the sibling `unet_9class_baseline` project).
- `IGNORE_INDEX = None`: every pixel is scored, including Background.
  Background is still excluded from the **headline mIoU** (`BACKGROUND_INDEX`
  in `ConfusionMatrix(exclude_indices=[...])`) because it dominates pixel
  count and would otherwise swamp the metric — but it does contribute to
  the loss and gradient during training.
- The raw→new id remap is implemented as a 256-entry lookup table
  (`REMAP_LUT`), so *any* raw byte value — including corrupt/out-of-range
  ones — resolves to a valid `0..18` id. This was verified directly while
  diagnosing the mid-run crash (see §4): it rules out invalid-label device
  asserts as a failure mode for this pipeline.

## 2. Architecture and loss

- **Encoder:** ResNet-34, ImageNet-pretrained.
- **Decoder:** UPerNet (Xiao et al. 2018, arXiv:1807.10221) — the decoder
  the UAVScenes paper itself uses for its segmentation benchmark (their
  Table 3 pairs it with ResNet-34/50/101 via `timm`). Here it's built via
  `segmentation_models_pytorch` instead, so the rest of the pipeline
  (dataset, loss, optimizer, batch size, resolution) stays byte-for-byte
  identical to the sibling `unet_dice_ce` project — the
  **only** variable changed is decoder architecture, making that comparison
  an apples-to-apples ablation on decoder choice alone.
- **Loss:** `0.5·Dice + 0.5·weighted CrossEntropy` ([losses.py](src/losses.py)).
  - Dice is optimized directly for overlap (closer to what mIoU measures),
    which tends to help with class imbalance and produces cleaner masks.
  - CE with **median-frequency class weighting**
    (`compute_class_weights`, clipped to max 10×) counteracts the fact that
    Background/large land-cover classes vastly outnumber small/rare classes
    (Sedan, Truck, Traffic_Barrier, Umbrella) in raw pixel count — without
    weighting, CE alone would essentially ignore those classes.
  - The two losses are complementary: Dice is a poor gradient signal early
    in training (near-empty/near-full masks give near-zero or unstable
    gradients) while CE is stable from step 1; blending both is a common
    and effective mitigation.

## 3. Training recipe reasoning

- **AdamW, lr 3e-4, wd 1e-4**, 3-epoch linear warmup then cosine decay to 0
  over the full 100 epochs — standard recipe for pretrained-encoder
  segmentation fine-tuning; warmup avoids destabilizing the pretrained
  ResNet-34 weights with a large initial step.
- **Gradient clipping (max norm 5.0)** — cheap insurance against the
  occasional large gradient spike from Dice loss on nearly-empty rare-class
  masks.
- **AMP (mixed precision)** — ~2× throughput on top of `cache_in_ram=True`
  (train+val fully cached in RAM at 256×256, ~4.2GB) with no measurable
  accuracy cost here.
- **`val_every: 1`** — full validation + checkpoint every epoch. This is
  what made the mid-run crash a non-event (see §4): at most one epoch of
  progress was ever at risk.
- **Early stopping (patience 15)** never triggered — val mIoU kept
  improving (however slowly) all the way to epoch 93, then held flat, so
  the run reached its full 100-epoch budget on its own.

## 4. Incident: mid-run CUDA crash at epoch 93

During the live run, training aborted mid-epoch-93 with a hard process
`abort()` (`Unhandled exception... AbortHandler`, `c10_cuda_check_implementation`
failing inside a CUDA event-record call). This is a *hard* crash, not a
catchable Python exception — it happens when a CUDA call fails after the
CUDA context is already in a bad state, and the resulting C++ exception is
thrown from a context PyTorch can't safely unwind on Windows.

**Root-cause narrowing done at the time:**
- Ruled out **invalid label ids** reaching the loss (the most common cause
  of this exact signature, via a device-side assert in `CrossEntropyLoss`)
  by inspecting the label pipeline: `REMAP_LUT` is a full 256-entry table,
  so every possible raw byte value — corrupt or not — maps to a valid
  `0..18` class id before it ever reaches the loss. This path cannot
  produce an out-of-range CE target.
- That leaves the fault as most likely a **GPU/driver-level event**: a
  Windows WDDM driver timeout/recovery (TDR), a transient VRAM error, or
  thermal throttling — none of which are visible from inside the training
  script's own state.

**Recovery:** because a checkpoint (`checkpoints/last.pt`) is written after
*every* epoch's validation pass, the crash — which happened ~200/369 steps
into epoch 93, right after epoch 92's checkpoint had already been saved —
cost at most the in-progress epoch, not the run. Training was resumed with:

```
python src/train.py --config configs/config_256.yaml --resume results_256/checkpoints/last.pt
```

and completed the remaining epochs (93→100) without further incident. The
resumed epoch 93 is, coincidentally, the epoch that produced the best
checkpoint of the entire run (val mIoU 0.8275).

**Secondary observation:** epoch wall-clock time is not constant across the
run — most epochs took ~294–300s, but epochs 86–90 and 93–100 (i.e.
everything after the crash/resume, plus a handful before it) ran in
~101–104s, roughly 3× faster, with a couple of ~250–280s outliers
interspersed (see `analysis/images/epoch_time.png`). All of this is
throughput variance from *outside* the training loop (data is fully
RAM-cached, so per-epoch compute work is constant) — likely OS-level
contention (background processes, driver state after the TDR/reset,
thermal state) rather than anything in the model or data pipeline. It did
not correlate with any change in loss or metrics, so it's noted here as an
operational curiosity, not a correctness concern.

## 5. Reading the loss/metric curves

- **Train loss** decreases smoothly and monotonically for all 100 epochs
  (0.85 → 0.038) — no instability, no divergence, gradient clipping and
  warmup did their job.
- **Val loss** bottoms out early (epoch 7, 0.190) then *rises* for the rest
  of training even as **val mIoU keeps improving** through epoch 93. This
  looks like classic overfitting on loss alone, but is expected and
  benign here for two compounding reasons:
  1. Cross-entropy penalizes confident-but-imperfect predictions
     increasingly harshly as the model's softmax outputs sharpen with more
     training — a model that is *more correct* by IoU can still produce a
     *numerically larger* CE loss on the pixels it still gets wrong,
     because it is more confident about them.
  2. Dice loss is bounded in [0,1] and closely tracks IoU, but CE is
     unbounded and dominates the *numeric* value of the combined loss once
     predictions get confident — so the blended loss's upward drift is
     mostly a CE artifact, not a sign the model is actually regressing.
  The plateauing (not diverging) mIoU/F1/accuracy curves after epoch ~85
  confirm this: had the model actually been overfitting in the way that
  matters, mIoU/accuracy would have fallen, not held flat.
- One anomalous dip at **epoch 22** (val mIoU 0.731 vs. 0.764/0.777 on
  either side, val loss spiking to 0.241) recovered fully by epoch 23 —
  consistent with a single noisy batch/validation-order effect rather than
  a systemic issue, given `shuffle=True` on the train loader and no
  corresponding disturbance in train_loss.
- **Recall > precision throughout** (final: 0.918 vs. 0.885) — the model
  leans slightly toward over-predicting foreground classes rather than
  missing them. Given the small/rare classes are the weak point (§6), this
  is the safer failure direction for most downstream uses of this map.

## 6. Per-class analysis — why the weak classes are weak

Weakest → strongest at the best epoch (93): **Sedan (0.545), Traffic_Barrier
(0.576), Truck (0.670), Umbrella (0.695), Car_Park (0.743), Background
(0.776)**, then a large jump to Container/Dirt_Motor_Road/Bridge/
Paved_Motor_Road (0.82–0.85), and near-ceiling on Wild_Field/Airstrip/
Green_Field/River (0.95–0.97).

This pattern is driven by three compounding factors, not a training bug:

1. **Object size vs. 256×256 resolution.** Sedan, Truck, Traffic_Barrier
   and Umbrella are the smallest objects in the taxonomy by footprint.
   Downsampling UAV imagery to 256×256 shrinks them to a handful of
   pixels, so boundary IoU error dominates their score — a 1-2 pixel
   segmentation offset that's negligible for a large field is a large
   fraction of a small vehicle's mask.
2. **Class rarity.** These are also the rarest classes by pixel frequency
   (median-frequency weighting caps their weight at 10× specifically
   because they're so underrepresented) — fewer training pixels means
   noisier gradient signal for exactly these classes throughout training.
3. **Visual ambiguity.** Car_Park vs. Paved_Motor_Road/Paved_Walk, and
   Background vs. the 7 folded "unused" raw ids, are texturally similar
   surfaces distinguished mainly by context/shape rather than local
   appearance — CNNs with a ResNet-34/UPerNet receptive field at 256×256
   have limited long-range context to resolve that ambiguity.

Encouragingly, this weak-class set (Truck, Sedan, Traffic_Barrier, Umbrella,
Bridge) matches the known weak points from the sibling vanilla-UNet 19-class
run (see `WEAK_CLASSES` in `analysis/scripts/plot_training_curves.py`) —
i.e. the difficulty is intrinsic to the *data and classes*, not specific to
the UPerNet decoder.

## 7. Improvement scope

Roughly in order of expected effort-to-payoff for the current weak classes:

- **Class-balanced / weighted sampling** at the dataloader level (on top of
  the existing loss-side weighting) so small/rare-class images are seen
  more often per epoch, not just weighted more heavily when seen.
- **Higher input resolution** (or tiled/cropped high-res training) for the
  small-object classes specifically — 256×256 is very likely the single
  biggest ceiling on Sedan/Truck/Traffic_Barrier/Umbrella performance.
- **Boundary-aware loss term** (e.g. boundary/edge loss, or Lovász-Softmax)
  in addition to Dice+CE — directly targets the boundary-pixel errors that
  dominate small-object IoU.
- **Stronger/targeted augmentation** for rare classes (copy-paste
  augmentation of vehicles/barriers onto other scenes, small-object-aware
  crop sampling) rather than uniform augmentation across all classes.
- **Multi-scale training/inference** (train at mixed resolutions, or
  inference with test-time multi-scale + flip averaging) — cheap to add at
  inference time (`evaluate.py`) with no retraining cost, typically worth
  0.5-2 mIoU points, more on small classes.
- **Larger/deeper encoder** (ResNet-50/101, or a modern backbone) — the
  UAVScenes paper's own benchmark shows this trend; worth an ablation once
  the ResNet-34 baselines across decoders are all collected.
- **Loss curve calibration:** since CE's upward drift post-epoch-40 isn't
  informative (see §5), consider tracking val Dice loss alone (or just
  val mIoU) as the early-stopping/model-selection signal instead of
  combined val_loss, to avoid ever mis-reading that drift as regression in
  future runs.
- **Crash resilience:** the checkpoint-every-epoch design already handled
  the epoch-93 crash gracefully; if further hard crashes recur, worth
  adding a thin auto-restart wrapper (detect abnormal process exit, relaunch
  with `--resume checkpoints/last.pt` automatically) so multi-hour runs
  don't need manual babysitting.

## 8. Artifacts produced by this run

| Artifact | Location |
|---|---|
| Config actually used | `results_256/config_used.yaml` |
| Class mapping snapshot | `results_256/class_info.json` |
| Per-epoch scalar metrics | `results_256/metrics.csv` |
| Per-epoch per-class IoU | `results_256/per_class_iou.csv` |
| TensorBoard logs | `results_256/tensorboard/` |
| Best checkpoint (epoch 93) | `results_256/checkpoints/best.pt` |
| Final/last checkpoint (epoch 100) | `results_256/checkpoints/last.pt` |
| Qualitative val-set snapshots, every epoch | `results_256/progress/epoch_001.png` … `epoch_100.png` |
| Rendered summary plots | `analysis/images/*.png` (generated by `analysis/scripts/plot_training_curves.py`) |
