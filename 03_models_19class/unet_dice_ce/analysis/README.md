# Analysis & Results

Consolidated analysis and result snapshots for the `unet_dice_ce`
run (vanilla UNet, resnet34 encoder, 19 UAVScenes classes, 256x256).

**Training is complete** (100/100 epochs, finished normally, best
checkpoint epoch 98) and the held-out test split has been evaluated —
see [final_results.md](final_results.md) for the converged result
(test mIoU 0.8055). The other files were written mid-training as
diagnostics (val-split snapshots at epochs 45/66, the val-class-presence
check, the paper comparison) and are kept for their trend/methodology
value; each states the epoch/checkpoint it was generated from. The
scripts in `scripts/` can be re-run at any time (e.g. against a future
retrain) to regenerate a fresh snapshot.

## Contents

- **[final_results.md](final_results.md)** — **Start here.** Training
  finished (100/100 epochs, no early stop); this is the converged
  test-set result (mIoU 0.8055), full per-class table, confusion-matrix
  read, and qualitative sample discussion. Supersedes the val-split
  snapshots in the other files below for "what's the final number."
- **[paper_comparison.md](paper_comparison.md)** — Full analysis of the
  UAVScenes paper (`../../UAVScenesPaper.pdf`) vs. this implementation:
  what architecture/hyperparameters/eval protocol the paper actually used
  for its Table 3 image segmentation benchmark, why our val mIoU (~80%+)
  comes out well above the paper's best reported number (68.3% mIoU,
  DeiT3-m), and a concrete list of every implementation difference that
  explains the gap (input crop size, loss function, decoder architecture,
  background handling, optimizer/schedule).
- **[training_progress_summary.md](training_progress_summary.md)** — Run
  config, key val_mIoU/val_loss milestones by epoch, and notes on the
  train/val loss-vs-mIoU divergence pattern seen from ~epoch 30 onward.
- **[classwise_val_iou_results.md](classwise_val_iou_results.md)** — Full
  19-class IoU/Precision/Recall/F1 breakdown on the val split, evaluated
  from a specific checkpoint (epoch noted in the file, includes a prior
  epoch-45 snapshot for trend comparison). Directly comparable to the
  paper's Table 3 per-class columns.
- **[val_class_distribution.md](val_class_distribution.md)** — Per-class
  pixel frequency and image-presence counts on the val split (4,207
  images) — confirms every one of the 18 non-background classes has real
  support in val (none silently dropped from the mIoU average).
- **images/** — Copies of the final test-set confusion matrix and three
  representative Image|GroundTruth|Prediction samples (forest/road,
  farmland/riverbank, courtyard — including one with a visible real
  error), referenced from `final_results.md`. Full set of 10 samples plus
  the original confusion matrix lives in `../results_256/test_eval/`.
- **scripts/** — Standalone, re-runnable versions of the diagnostic
  scripts used to generate the above:
  - `eval_val_classwise.py` — full per-class confusion-matrix breakdown on
    val from any checkpoint.
  - `check_val_class_distribution.py` — per-class pixel/image counts on
    val (no model needed, CPU-only, safe to run alongside active training).

## Regenerating a fresh snapshot

```powershell
cd E:\Multimodal\NamanAyush\03_models_19class\unet_dice_ce
python analysis\scripts\eval_val_classwise.py --checkpoint results_256\checkpoints\best.pt
python analysis\scripts\check_val_class_distribution.py
```

Both are read-only w.r.t. training state and safe to run while `train.py`
is still going (GPU inference batch size is kept small deliberately —
check `nvidia-smi` first if training is memory-tight).
