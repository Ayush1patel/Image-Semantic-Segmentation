# Paper vs. implementation: why our val mIoU exceeds the paper's numbers

Source: `UAVScenesPaper.pdf` (Wang et al., "UAVScenes: A Multi-Modal Dataset
for UAVs", arXiv:2507.22412), full 20-page text extracted and read in
detail, including the supplementary implementation details (Sec. S6/S7).

Trigger: our val mIoU reached ~80% by epoch 45-66, well above the paper's
best reported image-segmentation number (68.3% mIoU) and far above its
per-class numbers for rare classes (e.g. Umbrella: paper's best is 30.9%,
CNNs specifically score 1.1-8.5%; ours is ~67%). This doc is the result of
checking whether that's a bug or a real methodology difference.

**Conclusion up front: no bug found.** The gap is fully explained by real,
documented differences between the paper's segmentation benchmark protocol
and this project's setup — they are not directly comparable results, not a
case of "beating the paper."

## 1. The paper has no UNet baseline

Table 3's "Image Semantic Segmentation" block pairs classification
backbones (ResNet-34/50/101, ConvNext, ConvNextV2, ViT, DeiT3, MambaOut,
all via `timm`) with **UperNet** (Xiao et al. 2018) as the segmentation
decoder head. There is no UNet, UNet++, or any dense-skip-connection
encoder-decoder anywhere in the paper. The closest architectural relative
in their table is the ResNet-34/50/101 rows — same backbone family as
ours, but with a different decoder on top (see §4).

Paper's best result overall: **DeiT3-m, 68.3 mIoU** (Transformer, not CNN).
Best CNN result: **ResNet-50, 61.3 mIoU**.

## 2. Paper's exact segmentation training recipe (Sec. S7.1, page 15, verbatim)

> "In image semantic segmentation, we leverage the open-source timm
> package to conduct evaluations for various backbones, all of which are
> supported in timm. The input images are all resized and center-cropped
> to 224×224. The learning rate is set as 0.001, the batch size is set as
> 16, and we use Adam as the optimizer."

That is the entirety of the stated recipe. No mention of: loss function
beyond "Adam" (no Dice, no class weighting), data augmentation, epoch
count, or LR schedule.

Raw image size (Sec. S6.2, page 14): **2448×2048×3**.

## 3. Per-class Umbrella IoU, paper Table 3

| Model | Family | Umbrella IoU |
|---|---|---|
| ConvNext-t | CNN | 1.1% |
| MambaOut-t | CNN | 1.3% |
| ResNet-34 | CNN | 2.1% |
| MambaOut-s | CNN | 2.6% |
| ConvNextV2-t | CNN | 5.8% |
| ResNet-101 | CNN | 8.3% |
| ResNet-50 | CNN | 8.5% |
| ViT-s | Transformer | 11.2% |
| ViT-t | Transformer | 19.3% |
| DeiT3-s | Transformer | 24.1% |
| DeiT3-m | Transformer | 30.9% |
| **Ours (UNet+resnet34, epoch 45)** | — | **67.4%** |

See `classwise_val_iou_results.md` for our full 19-class table.

## 4. Concrete implementation differences

| | Paper (Table 3 protocol) | This project |
|---|---|---|
| Decoder | **UperNet** — pyramid pooling module on the backbone's last stage, fused top-down (FPN-style) with intermediate features, single fused map -> classifier | **Vanilla UNet** — symmetric decoder, encoder features **concatenated directly** at every resolution level (classic UNet skip connections) |
| Input crop | Resize + **center-crop to 224x224** from a native 2448x2048 image (~1% of image area, always the same central window) | **Full-frame resize to 256x256** — entire scene kept, no crop |
| Loss | Unspecified beyond Adam — no mention of class weighting or Dice | **0.5 Dice + 0.5 class-weighted CE** (median-frequency weighting, clipped at 10x) |
| Optimizer / schedule | Plain Adam, fixed LR 0.001 | AdamW + weight decay 1e-4, cosine schedule, 3-epoch warmup |
| Batch size | 16 | 32 |
| Background | Explicitly **ignored entirely** ("The background class is ignored," Fig. 5 caption; no Background column in Table 3) | Trained as a real class (`IGNORE_INDEX=None`), excluded only from the headline mean |
| Epochs | Not stated (single shared budget across a 9-backbone sweep) | 100 (cosine-scheduled), this run still in progress at time of writing |
| Backbone pretraining | ImageNet, via `timm` | ImageNet, via `segmentation_models_pytorch` (same ResNet-34 architecture/weights family) |

### Why the crop size is the dominant factor for small classes

Umbrella occupies a tiny footprint of a frame (our own val-set scan: 512 of
4207 images contain any Umbrella pixels at all, ~0.008% of all val pixels
— see `val_class_distribution.md`). A **fixed 224x224 window cropped from
the center** of a 2448x2048 image mostly does not contain a small,
off-center object like this:

- **At train time**: the model rarely sees an Umbrella to learn from —
  most training crops simply don't include one.
- **At eval time**: the metric is scored against a view that, for most
  images containing an umbrella, doesn't include it in the cropped window.

Our pipeline resizes the *whole* frame instead of cropping, so every rare
object is visible in every sample, just downsampled — a categorically
easier task for exactly the classes where the gap is largest (Umbrella,
Bridge, Solar_Board, Pool, Traffic_Barrier). Combined with explicit class
weighting + Dice loss (which the paper's recipe doesn't mention using),
this is sufficient to explain the ~2-8% (CNN) / ~31% (best Transformer)
vs. ~67% gap without needing a code defect.

### Why the decoder matters independently

UNet's defining feature — direct full-resolution skip connections at every
decoder stage — is specifically why it's the default for tasks with thin
or small structures (its origin is biomedical imaging: thin vessels, small
lesions). UperNet's pyramid-pooling fusion is built for general scene
parsing and doesn't preserve fine boundary detail the same way. This is a
second, independent reason to expect our small-class IoUs to beat a
UperNet+ResNet-34 result, on top of the crop-size and loss-weighting
differences above.

## 5. A hypothesis that was checked and ruled out

Initial suspicion: the train/val/test split reuses the same 4 physical
locations across different flight passes (`AMtown01`→train,
`AMtown02`→val, `AMtown03`→test, etc. — see `../src/splits.py`), which
looked like a leakage source inflating our val number relative to the
paper's.

**This does not differentiate our result from the paper's.** Sec. 3.2.1 +
Table S8 of the paper confirm that `01`/`02`/`03` within a split are
**the same continuous flight recorded on the same day**, time-sliced (e.g.
AMtown01/02/03 = one flight cut into 1354s/749s/619s time-ordered
segments). The paper's own official train/test split instructions are
built from this same sequence structure, so its own benchmark has the same
same-location characteristic our project inherits via `splits.py`. Ruled
out as a differentiator — the crop-size and loss-function differences in
§4 are the real explanation.

## 6. What was verified as correct (not a bug)

- `class_mapping.py`'s raw-id -> class-id LUT, nearest-neighbor mask
  resizing (never interpolated), and `ConfusionMatrix` accumulation were
  re-audited line by line.
- Every one of the 18 non-background classes has real pixel support in the
  val split (230-3890 images each) — none get silently dropped from the
  `nanmean`, which would otherwise inflate the average.
- `validate()` in `train.py` evaluates on `val_ds`, never train data.
- Per-class results are **not** suspiciously uniform — Traffic_Barrier
  (49%), Sedan (55%), and Truck (66%) are clearly the weak points, well
  below the mean and below the paper's headline numbers, which is what
  genuine (imperfect) learning looks like, not a metric artifact.
