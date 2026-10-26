# Building class error audit — findings

Model audited: CE+Lovász UNet (`unet_lovasz_ce`, `best.pt`,
epoch 82), scored on its full test split (3,285 images), read-only — no
training. Full per-image numbers: `building_error_per_image.csv`. Worst-20
visualizations (Image | Ground Truth | Prediction | Building error overlay):
`images/worst_01...20_*.png`.

## Headline stats

- 2,286 of 3,285 test images contain any Building ground truth.
- Mean per-image error (FP+FN pixels) on those: 570. Median: 226.
- The 20 worst-by-raw-error images account for only **4.8%** of total
  error pixels across the whole test set. Error is broadly spread across
  most building-containing images, not concentrated in a handful of
  catastrophic outliers.
- Re-ranking by per-image Building IoU instead of raw error count mostly
  surfaces images with **tiny (1–30 px) Building ground-truth regions** —
  almost certainly annotation specks/edge artifacts, not meaningful
  failures (trivially 0% IoU on a 2-pixel region tells us nothing useful).
  Noted as a possible minor label-quality issue, not a priority.

## What the worst-by-pixel-error images actually show (3 recurring patterns)

**1. Dense building clusters under/near tree canopy → fragmented errors.**
Example `AMtown03 / 1658132106...`: large, clearly-visible buildings in
the lower half of the frame are segmented almost perfectly (solid green
in the error overlay). The upper half — smaller buildings partially
obscured by tree canopy — shows scattered red/blue speckling. Same
image, same model, very different accuracy depending on occlusion.

**2. Large, irregularly-shaped single buildings → shape/extent misalignment.**
Example `HKisland03 / 1669702867...`: one building with a non-rectangular,
stepped/zigzag footprint. The prediction is a smoother blob that's offset
from the true outline — large contiguous red (false positive) on one
side, large contiguous blue (false negative) on the other, only a small
green core actually overlapping. Example `HKisland03 / 1669702896...` is
the same pattern even more starkly: the model's predicted extent runs
systematically short along one entire edge of a large building (fn=2651
vs fp=899 — a one-sided undershoot, not random noise).

**3. Frames with many separate, clearly-visible buildings → error accumulates from many small per-building rims, not from failing any one building badly.**
Examples `AMtown03 / 1658132249...` and `.../1658132250...`: 10+ distinct
buildings in frame, each individually segmented quite well (solid green
blobs), but each has a thin ring of red/blue around its boundary. No
single building is badly wrong, but with that many instances in one
frame, the summed error is large enough to land in the worst-20 by raw
pixel count — this is a volume effect, not a severity effect.

## Reading this against the goal

The errors aren't really a *classification* problem — in every example
reviewed, the all-classes panels show Building is still being recognized
as Building, just with the wrong shape/extent. That rules out "model
confuses Building with X" as the main story (consistent with the
confusion matrix already showing Building is clean, ~95% on-diagonal).
The three patterns above are about **where** the building boundary is
localized, in two different regimes:

- **Occlusion/clutter-driven** (pattern 1) — the scene content is making
  it genuinely hard to tell where the building ends, independent of
  boundary-loss-shaping choices.
- **Shape/extent bias on complex footprints** (pattern 2) — the model
  smooths or shifts irregular outlines toward something simpler.
- **Accumulated per-instance boundary imprecision** (pattern 3) — the
  single biggest driver of "worst image" rank in dense scenes is many
  small, consistent per-building boundary rims summed together, not
  anything catastrophic.

Flagging pattern 3 plainly since it does point toward boundary
localization as a real, recurring factor — raising it as data rather than
re-opening the boundary-loss discussion, since that's a call to make
deliberately, not something to infer unilaterally from one audit.
