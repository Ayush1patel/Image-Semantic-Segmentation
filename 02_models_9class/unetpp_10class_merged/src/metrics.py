import numpy as np
import torch


class ConfusionMatrix:
    """Accumulates a confusion matrix across batches and derives
    IoU / precision / recall / F1 / pixel accuracy."""

    def __init__(self, num_classes: int, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """preds, targets: LongTensors of the same shape (any dims), values
        in [0, num_classes)."""
        preds = preds.detach().cpu().numpy().ravel()
        targets = targets.detach().cpu().numpy().ravel()
        n = self.num_classes
        idx = n * targets.astype(np.int64) + preds.astype(np.int64)
        binc = np.bincount(idx, minlength=n * n)
        self.mat += binc.reshape(n, n)

    def reset(self):
        self.mat[:] = 0

    def _tp_fp_fn_support(self):
        mat = self.mat.astype(np.float64)
        tp = np.diag(mat)
        fp = mat.sum(axis=0) - tp
        fn = mat.sum(axis=1) - tp
        support = mat.sum(axis=1)  # ground-truth pixel count per class
        return tp, fp, fn, support

    def _safe_div(self, num, denom):
        out = np.full(self.num_classes, np.nan)
        valid = denom > 0
        out[valid] = num[valid] / denom[valid]
        return out

    def per_class_iou(self):
        tp, fp, fn, _ = self._tp_fp_fn_support()
        return self._safe_div(tp, tp + fp + fn)

    def per_class_precision(self):
        tp, fp, _, _ = self._tp_fp_fn_support()
        return self._safe_div(tp, tp + fp)

    def per_class_recall(self):
        tp, _, fn, _ = self._tp_fp_fn_support()
        return self._safe_div(tp, tp + fn)

    def per_class_f1(self):
        precision = self.per_class_precision()
        recall = self.per_class_recall()
        return self._safe_div(2 * precision * recall, precision + recall)

    def per_class_support(self):
        _, _, _, support = self._tp_fp_fn_support()
        return support

    def mean_iou(self):
        return float(np.nanmean(self.per_class_iou()))

    def mean_precision(self):
        return float(np.nanmean(self.per_class_precision()))

    def mean_recall(self):
        return float(np.nanmean(self.per_class_recall()))

    def mean_f1(self):
        return float(np.nanmean(self.per_class_f1()))

    def pixel_accuracy(self):
        mat = self.mat.astype(np.float64)
        return float(np.diag(mat).sum() / max(mat.sum(), 1.0))

    def as_dict(self):
        iou = self.per_class_iou()
        precision = self.per_class_precision()
        recall = self.per_class_recall()
        f1 = self.per_class_f1()
        support = self.per_class_support()

        def clean(x):
            return None if np.isnan(x) else float(x)

        return {
            "pixel_accuracy": self.pixel_accuracy(),
            "mean_iou": self.mean_iou(),
            "mean_precision": self.mean_precision(),
            "mean_recall": self.mean_recall(),
            "mean_f1": self.mean_f1(),
            "per_class": {
                name: {
                    "iou": clean(iou[i]),
                    "precision": clean(precision[i]),
                    "recall": clean(recall[i]),
                    "f1": clean(f1[i]),
                    "support_pixels": int(support[i]),
                }
                for i, name in enumerate(self.class_names)
            },
        }

    def summary(self):
        iou = self.per_class_iou()
        precision = self.per_class_precision()
        recall = self.per_class_recall()
        f1 = self.per_class_f1()
        support = self.per_class_support()

        lines = [
            f"Pixel Accuracy:  {self.pixel_accuracy():.4f}",
            f"Mean IoU:        {self.mean_iou():.4f}",
            f"Mean Precision:  {self.mean_precision():.4f}",
            f"Mean Recall:     {self.mean_recall():.4f}",
            f"Mean F1:         {self.mean_f1():.4f}",
            "",
            f"{'Class':<18}{'IoU':>8}{'Precision':>12}{'Recall':>10}{'F1':>8}{'Support(px)':>14}",
        ]

        def fmt(v):
            return "n/a" if np.isnan(v) else f"{v:.4f}"

        for i, name in enumerate(self.class_names):
            lines.append(
                f"{name:<18}{fmt(iou[i]):>8}{fmt(precision[i]):>12}"
                f"{fmt(recall[i]):>10}{fmt(f1[i]):>8}{int(support[i]):>14}"
            )
        return "\n".join(lines)
