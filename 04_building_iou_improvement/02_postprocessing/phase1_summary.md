# Phase 1 — Post-processing experiment on Building masks: results

Tested on the CE+Lovász checkpoint's existing test-set predictions (no
retraining). Pipeline: 3×3 morphological open→close, drop connected
components under 30px, simplify remaining contours with
`cv2.approxPolyDP` (epsilon = 1% of perimeter), rasterize back to a mask.
Full per-image numbers: `building_postprocess_comparison.csv`. Before/after
visuals for the same 20 worst-case images as the original audit:
`images/before_after_01...20_*.png`.

## Headline result: it made Building IoU slightly *worse*, not better

| Metric | Raw prediction | Post-processed | Δ |
|---|---|---|---|
| Building IoU | 0.8497 | 0.8351 | **−0.0146** |
| Building Precision | 0.8960 | 0.8958 | −0.0002 |
| Building Recall | 0.9427 | 0.9250 | **−0.0176** |

Per-image breakdown (3,285 test images): **630 improved, 1,634 got worse,
1,021 unchanged.** This is a systematic regression, not a few outliers
dragging the average down — most of the net change came from recall
dropping while precision stayed essentially flat.

## Why — the recall-only drop is the tell

Precision barely moved; recall dropped almost 2 points. That pattern
means the pipeline is **erasing previously-correct Building pixels**
(turning TP into FN) without meaningfully reducing false positives. Two
likely culprits in the recipe, both bias toward shrinking:

1. **Morphological opening erodes before it dilates back.** On small or
   thin building regions — common in this dataset's dense town scenes —
   opening can shave off real structure that never gets fully restored by
   the closing step, especially where a building's predicted shape is
   already only a few pixels wide in places.
2. **The 30px area filter and 1%-of-perimeter contour simplification are
   both proportionally much more aggressive on small buildings than large
   ones.** A big building's perimeter is long, so 1% of it is still a
   fine simplification; a small building's perimeter is short, so the
   same 1% collapses it into far fewer vertices relative to its size,
   distorting its shape more than it cleans it. The 30px filter risks
   deleting genuinely-correct small/partial building predictions outright
   (e.g. a building clipped at the frame edge), not just the annotation
   noise it was aimed at.

Visually (see `images/before_after_02...` and `.../09...`), the
post-processed masks in dense clusters don't look dramatically different
from the raw ones at a glance — the regression is real but not a visibly
obvious disaster, which is exactly why the quantitative check mattered
more than eyeballing it would have.

## Reading

This specific recipe, at these specific parameters, is a net negative —
worth saying plainly rather than soft-pedaling it. It does **not** mean
post-processing is a dead end in general; it means this first attempt was
tuned wrong for this dataset's building-size distribution, specifically
too aggressive on small/thin buildings. A better-tuned version (skip the
opening step and keep only closing, which only fills holes rather than
eroding; drop or shrink the area filter; make epsilon scale-aware instead
of a flat 1%) might behave differently — but that's a second experiment,
not a conclusion to chase immediately given the time budget. Recommend
treating Phase 1 as closed for now (confirmed: cheap post-processing
isn't a free win at these settings) and moving to Phase 2, since that's
where the actual budgeted time is going regardless.
