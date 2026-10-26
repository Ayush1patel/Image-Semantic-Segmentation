r"""
Phase 2 (no retraining) -- inference-time gains for the Building class on the
EXISTING CE+Lovasz checkpoint.

Three levers, all applied to the model's softmax output rather than to a
hard mask (Phase 1's mask-level morphology/contour cleanup lost recall):
  1. Test-time augmentation: flips (4 views) and multiscale (0.75x/1x/1.25x),
     averaging the full 19-class softmax across views.
  2. Optional light Gaussian smoothing of the Building probability map.
  3. A Building decision threshold on P(Building), swept, instead of plain
     argmax (recall 0.94 > precision 0.90 at argmax, so a higher threshold
     should trade a little recall for precision).

Protocol: every parameter (TTA config, sigma, threshold) is chosen on the
VAL split only. The test split is scored once with the val-chosen settings.

Per image and per TTA config we keep only two fp16 maps:
  P_b       averaged Building probability
  P_other   max averaged probability over the 18 non-Building classes
so "argmax says Building" == (P_b > P_other), and the threshold rule is
(P_b > t). No other class is needed for Building IoU.

Usage (training venv):
    & "E:\Multimodal\NamanAyush\\.venv\Scripts\python.exe" tta_threshold_sweep.py
"""
import csv
import itertools
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT = os.path.join(os.path.dirname(__file__), "..", "..", "03_models_19class", "unet_lovasz_ce")
sys.path.insert(0, os.path.join(PROJECT, "src"))

from class_mapping import NEW_CLASS_NAMES  # noqa: E402
from dataset import UAVScenesDataset  # noqa: E402
from model import build_model  # noqa: E402
from transforms import get_eval_transform  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILDING_IDX = NEW_CLASS_NAMES.index("Building")
SCALES = (192, 256, 320)          # 0.75x, 1x, 1.25x of the 256 training size
FLIPS = ("id", "h", "v", "hv")
CONFIGS = {                        # name -> (scales used, flips used)
    "none":        ((256,),       ("id",)),
    "flip_h":      ((256,),       ("id", "h")),     # the only flip used in training aug
    "flip4":       ((256,),       FLIPS),
    "ms3":         (SCALES,       ("id",)),
    "ms3_flip_h":  (SCALES,       ("id", "h")),
    "ms3_flip4":   (SCALES,       FLIPS),
}
SIGMAS = (0.0, 0.5, 1.0, 1.5, 2.0)
THRESHOLDS = np.round(np.arange(0.30, 0.801, 0.025), 3)


def apply_flip(x, f):
    if f == "h":
        return x.flip(-1)
    if f == "v":
        return x.flip(-2)
    if f == "hv":
        return x.flip(-1).flip(-2)
    return x


@torch.no_grad()
def collect(split, model, device, cfg):
    """Returns dict config -> (P_b, P_other) fp16 tensors (N,H,W) on CPU, plus gt Building bool (N,H,W)."""
    h, w = cfg["data"]["height"], cfg["data"]["width"]
    ds = UAVScenesDataset(split, transform=get_eval_transform(h, w),
                          index_dir=os.path.join(PROJECT, "data_index"))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    n = len(ds)
    store = {c: (torch.empty(n, h, w, dtype=torch.float16), torch.empty(n, h, w, dtype=torch.float16))
             for c in CONFIGS}
    gt = torch.empty(n, h, w, dtype=torch.bool)

    i0 = 0
    for images, labels in tqdm(loader, desc=f"[{split}] TTA inference"):
        b = images.size(0)
        images = images.to(device, non_blocking=True)
        probs = {}                              # (scale, flip) -> (B,19,H,W) softmax at 256
        for s, f in itertools.product(SCALES, FLIPS):
            x = images if s == h else F.interpolate(images, size=(s, s), mode="bilinear", align_corners=False)
            logits = apply_flip(model(apply_flip(x, f)), f)
            p = torch.softmax(logits, dim=1)
            if s != h:
                p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)
            probs[(s, f)] = p
        for name, (scales, flips) in CONFIGS.items():
            views = [probs[(s, f)] for s in scales for f in flips]
            avg = sum(views) / len(views)
            p_b = avg[:, BUILDING_IDX]
            p_other = torch.cat([avg[:, :BUILDING_IDX], avg[:, BUILDING_IDX + 1:]], dim=1).amax(dim=1)
            store[name][0][i0:i0 + b] = p_b.half().cpu()
            store[name][1][i0:i0 + b] = p_other.half().cpu()
        gt[i0:i0 + b] = (labels == BUILDING_IDX)
        i0 += b
        del probs
    return ds, store, gt


def gaussian_kernel(sigma, device):
    r = max(1, int(round(3 * sigma)))
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return (k / k.sum()), r


def smooth(p, sigma):
    """p: (N,H,W) float tensor on device; separable Gaussian, reflect padding."""
    if sigma <= 0:
        return p
    k, r = gaussian_kernel(sigma, p.device)
    x = p.unsqueeze(1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return x.squeeze(1)


def counts(pred, gt):
    """Per-image (tp, fp, fn) as an (N,3) int64 tensor."""
    tp = (pred & gt).flatten(1).sum(1)
    fp = (pred & ~gt).flatten(1).sum(1)
    fn = (~pred & gt).flatten(1).sum(1)
    return torch.stack([tp, fp, fn], dim=1).long()


def metrics(c):
    tp, fp, fn = [int(v) for v in c.sum(0)]
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return iou, prec, rec


def per_image_counts(p_b, p_other, gt, device, mode, sigma=0.0, t=0.5, chunk=400):
    out = []
    for i in range(0, p_b.size(0), chunk):
        pb = p_b[i:i + chunk].to(device).float()
        g = gt[i:i + chunk].to(device)
        if mode == "argmax":
            pred = pb > p_other[i:i + chunk].to(device).float()
        else:
            pred = smooth(pb, sigma) > t
        out.append(counts(pred, g).cpu())
    return torch.cat(out)


def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(PROJECT, "configs", "config_256.yaml")) as f:
        cfg = yaml.safe_load(f)
    ckpt_path = os.path.join(PROJECT, "results_256", "checkpoints", "best.pt")
    model = build_model(encoder_name=cfg["model"]["encoder_name"], encoder_weights=None,
                        num_classes=len(NEW_CLASS_NAMES)).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {ckpt_path} (epoch {ckpt.get('epoch')}, best_miou={ckpt.get('best_miou')})")

    val_ds, val_store, val_gt = collect("val", model, device, cfg)
    t_val = time.time()
    test_ds, test_store, test_gt = collect("test", model, device, cfg)
    print(f"Inference done: val {t_val - t_start:.0f}s, test {time.time() - t_val:.0f}s")

    # ---------------- sweep on VAL ----------------
    sweep_rows = []
    best = None                       # (val_iou, config, sigma, t)
    val_curves = {}
    for name in CONFIGS:
        p_b, p_o = val_store[name]
        c = per_image_counts(p_b, p_o, val_gt, device, "argmax")
        iou, prec, rec = metrics(c)
        sweep_rows.append(dict(split="val", config=name, rule="argmax", sigma="", threshold="",
                               iou=iou, precision=prec, recall=rec))
        for sigma in SIGMAS:
            curve = []
            for t in THRESHOLDS:
                c = per_image_counts(p_b, p_o, val_gt, device, "thr", sigma, float(t))
                iou, prec, rec = metrics(c)
                curve.append(iou)
                sweep_rows.append(dict(split="val", config=name, rule="threshold", sigma=sigma,
                                       threshold=float(t), iou=iou, precision=prec, recall=rec))
                if best is None or iou > best[0]:
                    best = (iou, name, sigma, float(t))
            val_curves[(name, sigma)] = curve
        print(f"  val sweep done for config={name}")
    val_iou_best, b_cfg, b_sigma, b_t = best
    print(f"\nBest on VAL: config={b_cfg} sigma={b_sigma} threshold={b_t} -> Building IoU {val_iou_best:.4f}")

    # ---------------- score TEST once with val-chosen settings ----------------
    results = []   # (label, test_iou, prec, rec, val_iou)

    def test_row(label, name, rule, sigma=0.0, t=0.5):
        p_b, p_o = test_store[name]
        c = per_image_counts(p_b, p_o, test_gt, device, rule, sigma, t)
        iou, prec, rec = metrics(c)
        v_p_b, v_p_o = val_store[name]
        v_iou = metrics(per_image_counts(v_p_b, v_p_o, val_gt, device, rule, sigma, t))[0]
        results.append(dict(label=label, config=name, rule=rule, sigma=sigma, threshold=t if rule == "threshold" else "",
                            val_iou=v_iou, test_iou=iou, test_precision=prec, test_recall=rec))
        return c

    base_counts = test_row("Baseline (plain argmax, no TTA)", "none", "argmax")
    for name in CONFIGS:
        if name != "none":
            test_row(f"{name} + argmax", name, "argmax")
    # per-config best on val (threshold only/sigma chosen on val)
    cfg_best = {}
    for name in CONFIGS:
        bi = max(((max(val_curves[(name, s)]), s, float(THRESHOLDS[int(np.argmax(val_curves[(name, s)]))]))
                  for s in SIGMAS), key=lambda x: x[0])
        cfg_best[name] = bi
        test_row(f"{name} + smooth/threshold (val-tuned)", name, "threshold", bi[1], bi[2])
    final_counts = test_row("FINAL: best val config", b_cfg, "threshold", b_sigma, b_t)

    # threshold-only variant with the final TTA config (isolates the smoothing contribution)
    name = b_cfg
    t_only = max(((val_curves[(name, 0.0)][i], float(THRESHOLDS[i])) for i in range(len(THRESHOLDS))))
    test_row(f"{name} + threshold only, no smoothing (val-tuned)", name, "threshold", 0.0, t_only[1])

    # ---------------- outputs ----------------
    with open(os.path.join(OUT_DIR, "val_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader(); w.writerows(sweep_rows)
    with open(os.path.join(OUT_DIR, "test_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

    # per-image + per-sequence on test, baseline vs final
    seqs = [s["sequence"] for s in test_ds.samples]
    with open(os.path.join(OUT_DIR, "test_per_image.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["sequence", "stem", "base_tp", "base_fp", "base_fn", "final_tp", "final_fp", "final_fn"])
        for i, s in enumerate(test_ds.samples):
            wr.writerow([s["sequence"], s["stem"], *base_counts[i].tolist(), *final_counts[i].tolist()])
    per_seq = {}
    for seq in sorted(set(seqs)):
        idx = torch.tensor([i for i, s in enumerate(seqs) if s == seq])
        per_seq[seq] = (metrics(base_counts[idx])[0], metrics(final_counts[idx])[0], len(idx))
    improved = int((final_counts[:, 1] + final_counts[:, 2] < base_counts[:, 1] + base_counts[:, 2]).sum())
    worse = int((final_counts[:, 1] + final_counts[:, 2] > base_counts[:, 1] + base_counts[:, 2]).sum())
    same = len(seqs) - improved - worse

    # plot: IoU vs threshold on val (best-sigma per config) + test curve for final config
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for name in CONFIGS:
        s = cfg_best[name][1]
        ax[0].plot(THRESHOLDS, val_curves[(name, s)], label=f"{name} (sigma={s})")
    ax[0].set_title("VAL Building IoU vs threshold"); ax[0].set_xlabel("threshold on P(Building)")
    ax[0].set_ylabel("IoU"); ax[0].legend(); ax[0].grid(alpha=.3)
    p_b, p_o = test_store[b_cfg]
    test_curve = [metrics(per_image_counts(p_b, p_o, test_gt, device, "thr", b_sigma, float(t)))[0] for t in THRESHOLDS]
    ax[1].plot(THRESHOLDS, val_curves[(b_cfg, b_sigma)], label="val")
    ax[1].plot(THRESHOLDS, test_curve, label="test")
    ax[1].axvline(b_t, color="k", ls="--", lw=.8, label=f"chosen t={b_t}")
    ax[1].set_title(f"{b_cfg}, sigma={b_sigma}: val vs test"); ax[1].set_xlabel("threshold")
    ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "iou_vs_threshold.png"), dpi=110); plt.close(fig)

    meta = dict(best_val=dict(iou=val_iou_best, config=b_cfg, sigma=b_sigma, threshold=b_t),
                per_sequence_test={k: dict(base_iou=v[0], final_iou=v[1], n=v[2]) for k, v in per_seq.items()},
                per_image_test=dict(improved=improved, worse=worse, unchanged=same),
                elapsed_seconds=time.time() - t_start)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n=== TEST results (settings chosen on VAL) ===")
    for r in results:
        print(f"{r['label']:<60} val IoU {r['val_iou']:.4f} | test IoU {r['test_iou']:.4f}  "
              f"P {r['test_precision']:.4f}  R {r['test_recall']:.4f}")
    print(f"\nPer-image (test, by FP+FN): improved={improved} worse={worse} unchanged={same}")
    for k, v in per_seq.items():
        print(f"  {k}: base {v[0]:.4f} -> final {v[1]:.4f} (n={v[2]})")
    print(f"\nTotal elapsed: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
