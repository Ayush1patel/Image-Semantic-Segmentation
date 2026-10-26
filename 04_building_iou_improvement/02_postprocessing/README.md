# Phase 1 — Building mask post-processing: full technical explainer

This document explains, in full, what `postprocess_building_masks.py`
does, every technical term it uses, what every number in
`building_postprocess_comparison.csv` means, and how to read the images
in `images/`. Written so it's self-contained — no need to already know
OpenCV or segmentation terminology to follow it.

## 1. What this experiment actually is

We already had a trained segmentation model (the CE+Lovász UNet,
`../../../03_models_19class/unet_lovasz_ce/results_256/checkpoints/best.pt`).
This script does **not** touch that model or retrain anything. It takes
the model's existing predictions on the test set and runs a second,
separate cleanup step on just the Building class, purely as image
processing on the output mask. The question being asked: *"if we clean up
the model's raw Building predictions with standard image-processing
tricks, does the Building score get better?"* — cheap to test (minutes,
no GPU training) before committing hours to retraining anything.

## 2. Background: what a segmentation mask is, concretely

For each test image, the model outputs one predicted class (out of 19)
per pixel. For this experiment we only care about one question per pixel:
*"is this pixel Building, yes or no?"* — so every mask in this experiment
is a **binary mask**: a 256×256 grid where each cell is either `True`
(the model/ground-truth says "Building here") or `False` (anything else).

Comparing a predicted binary mask to the true binary mask (from the
human-labeled ground truth) gives four possible outcomes per pixel:

| | Predicted Building | Predicted not-Building |
|---|---|---|
| **Actually Building** | TP (true positive) — correct | FN (false negative) — missed it |
| **Actually not Building** | FP (false positive) — invented it | TN (true negative) — correctly ignored, not tracked here |

From these three tracked counts (TP, FP, FN — TN isn't used for this
metric), the whole experiment's scoring comes from three standard
formulas:

- **IoU** (Intersection-over-Union, a.k.a. the Jaccard index) = `TP / (TP + FP + FN)`.
  This is "of all the pixels either side called Building, what fraction did both sides agree on." The single headline number used throughout this whole project.
- **Precision** = `TP / (TP + FP)` — "of everything the model called Building, what fraction was actually right." Low precision = too many false alarms.
- **Recall** = `TP / (TP + FN)` — "of everything that's actually Building, what fraction did the model find." Low recall = too much missed.

## 3. The post-processing pipeline, step by step

For every test image, the model's raw predicted Building mask goes
through three steps, in this exact order, before being compared back to
the ground truth:

### Step 1 — Morphological opening, then closing

"Morphological" operations treat the binary mask as a shape and reshape
it using a small sliding window called a **kernel** (here, a 3×3 square —
literally "look at each pixel's 3×3 neighborhood"). Two base operations:

- **Erosion**: a pixel stays `True` only if *every* pixel in its 3×3
  neighborhood is also `True`. Effect: shrinks every white (Building)
  region inward by about 1 pixel on all sides, deleting anything thinner
  than the kernel entirely.
- **Dilation**: a pixel becomes `True` if *any* pixel in its 3×3
  neighborhood is `True`. Effect: grows every white region outward by
  about 1 pixel, also fills in small gaps.

Combining them gives two composite operations, both used here:

- **Opening** = erosion, then dilation. Net effect: small isolated
  specks (smaller than the kernel) get erased entirely (erosion deletes
  them, dilation has nothing left to grow back), while large regions lose
  and then regain their size — intended to remove noise without changing
  big shapes much.
- **Closing** = dilation, then erosion. Net effect: small holes *inside*
  a region get filled in (dilation closes the gap, erosion shrinks back
  down), while the overall outline is preserved.

This script applies opening first, then closing, to the raw predicted
mask — the intent was "strip small false-positive speckles, then fill
small false-negative holes." (See §5 for why this step backfired here.)

### Step 2 — Connected-component area filtering

A **connected component** is one contiguous blob — a group of `True`
pixels that all touch each other (directly or via a chain of touching
neighbors). `cv2.findContours(..., cv2.RETR_EXTERNAL, ...)` finds the
outer boundary of every separate blob in the mask (`RETR_EXTERNAL` means
it only traces each blob's outer edge, not any holes inside it).

For each blob found, `cv2.contourArea(...)` computes its pixel area. Any
blob with **area < 30 pixels** is discarded outright — the reasoning
(from the earlier error audit, not re-derived here) was that
ground-truth Building regions under ~30 pixels are almost always
annotation specks/edge artifacts, not real structure, so predicted blobs
that tiny were assumed to be noise too.

### Step 3 — Contour simplification (Douglas-Peucker) and rasterization

For every surviving blob, `cv2.approxPolyDP(contour, epsilon, True)`
runs the **Douglas-Peucker algorithm**: it replaces the blob's full,
pixel-by-pixel jagged outline with a much smaller set of straight line
segments (a polygon), such that no point on the simplified polygon is
farther than `epsilon` pixels from the original outline. `epsilon` here
is set to **1% of the blob's own perimeter** (`cv2.arcLength`) — so a
large building (long perimeter) gets a fine simplification, a small
building (short perimeter) gets a coarser one, in absolute pixel terms.

The simplified polygon is then **rasterized** back into a filled binary
mask (`cv2.fillPoly`) — turned back into ordinary `True`/`False` pixels —
so it can be compared pixel-for-pixel against the ground truth exactly
like the raw prediction was.

### Parameters used, summarized

| Parameter | Value | Meaning |
|---|---|---|
| `MORPH_KERNEL` | 3 (3×3 window) | Smallest unit of "noise" the opening/closing step can affect |
| `MIN_AREA` | 30 pixels | Blobs smaller than this (out of 65,536 pixels/image) are deleted entirely |
| `EPSILON_RATIO` | 0.01 (1% of perimeter) | How aggressively each blob's boundary is simplified into straight segments |

## 4. How to read the output files

**`building_postprocess_comparison.csv`** — one row per test image:
`sequence`, `stem` (the image's identifying timestamp filename),
`index` (position in the test set), then `raw_tp/raw_fp/raw_fn` (the
three counts from §2, for the *unmodified* model prediction) and
`post_tp/post_fp/post_fn` (the same three counts *after* the Steps 1–3
pipeline above was applied).

**`images/before_after_NN_<sequence>_<stem>.png`** — the same 20 worst
(highest raw-error) images identified in the original Building audit
(`../01_error_audit/audit_summary.md`), each rendered as 4 panels:

1. **Image** — the original aerial RGB photo.
2. **Ground Truth (all classes)** — the human-labeled answer, colorized
   across all 19 classes (not just Building) so you can see the scene
   context — maroon is specifically the Building color in this palette.
3. **Raw prediction (error=N)** — the *error overlay* for the unmodified
   model output, Building class only: **green** = TP (model and ground
   truth agree it's Building), **red** = FP (model said Building, it
   wasn't), **blue** = FN (it was Building, model missed it), light gray
   = everything else (correctly not-Building on both sides). `N` in the
   title is that image's total error pixel count (`fp + fn`).
4. **Post-processed (error=N)** — the identical error overlay, but using
   the mask *after* Steps 1–3 were applied, for direct visual comparison
   against panel 3.

## 5. What actually happened — the result, explained mechanistically

Aggregate result across all 3,285 test images:

| Metric | Raw | Post-processed | Δ |
|---|---|---|---|
| IoU | 0.8497 | 0.8351 | **−0.0146** |
| Precision | 0.8960 | 0.8958 | ≈0 |
| Recall | 0.9427 | 0.9250 | **−0.0176** |

Per-image: 630 images improved, **1,634 got worse**, 1,021 unchanged —
a broad, systematic shift, not a few bad outliers.

**Why this happened**, tying back to the mechanics in §3: Precision
staying flat while recall drops is the signature of the pipeline
*deleting previously-correct Building pixels* (turning TP into FN)
without removing many false positives. The most likely cause is **Step 1
specifically**: opening's erosion half can fully erase real Building
regions that are already thin or small relative to the 3×3 kernel —
common in this dataset's dense, tree-occluded building clusters — and
dilation only grows back from whatever survived erosion, so thin
structure that gets fully eroded away never returns. Step 2's 30-pixel
filter compounds this by also deleting any genuinely-correct small
prediction outright (e.g. a building clipped at the image edge), not
just the annotation noise it targeted.

**Conclusion:** at these specific parameter values, this post-processing
recipe is a net negative for pixel-level Building accuracy on this
dataset. It does not rule out post-processing in general — a version
that skips the erosion-based opening step (keeping only closing, which
only fills holes and never deletes real structure) and/or relaxes the
area filter might behave differently — but that would be a new,
separate experiment, not a continuation of this one.
