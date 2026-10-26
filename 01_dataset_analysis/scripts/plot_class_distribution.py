"""
Visualize the 19-class pixel distribution computed by
summarize_class_distribution.py (01_dataset_analysis/outputs/class_distribution_19class.csv).

Produces 4 PNGs in "class_distribution_report/images/":

  01_global_class_frequency.png
      Global pixel frequency per class, log x-scale (dynamic range spans
      ~4000x, from Green_Field at 39% down to Bridge at 0.009%).

  02_sequence_class_heatmap_freq.png
      Sequence (folder) x class heatmap of pixel frequency WITHIN each
      sequence, log color scale. True zeros are masked distinctly (light
      gray) from "present but tiny", since for several classes that
      distinction is the whole point (see 03).

  03_sequence_class_heatmap_coverage.png
      Same grid, but color = fraction of that sequence's images containing
      the class at all (0-1, linear). Complements 02: a class can have low
      pixel share but appear in most frames (small object, e.g. Sedan), or
      higher pixel share but only in a few frames.

  04_rare_classes_per_sequence.png
      Small multiples: one bar subplot per rarest class (Bridge, Umbrella,
      Pool, Solar_Board, Truck), pixel freq within each sequence, bars
      colored by site group (AMtown / AMvalley / HKairport / HKisland).
      This is the chart that makes the site-exclusivity finding (several
      rare classes exist in only ONE site) visually obvious at a glance.

Uses the validated categorical/sequential palette from the dataviz skill
(references/palette.md): sequential = single blue hue ramp for magnitude
(heatmaps); categorical = fixed-order slots 1-4 (blue/orange/aqua/yellow)
for the 4 site groups, never re-cycled per chart.

Usage:
    python 01_dataset_analysis/scripts/plot_class_distribution.py
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
IN_CSV = os.path.join(OUT_DIR, "class_distribution_19class.csv")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "class_distribution_report", "images")

# ---------------------------------------------------------------------
# Palette (dataviz skill, references/palette.md) -- light chart surface.
# ---------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

CAT_BLUE = "#2a78d6"     # slot 1 -- AMtown
CAT_ORANGE = "#eb6834"   # slot 2 -- AMvalley
CAT_AQUA = "#1baf7a"     # slot 3 -- HKairport
CAT_YELLOW = "#eda100"   # slot 4 -- HKisland
ZERO_GRAY = "#dedcd4"

SEQ_BLUE_STEPS = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE_STEPS)

SITE_COLOR = {
    "AMtown": CAT_BLUE,
    "AMvalley": CAT_ORANGE,
    "HKairport": CAT_AQUA,
    "HKisland": CAT_YELLOW,
}
SITE_ORDER = ["AMtown", "AMvalley", "HKairport", "HKisland"]

UNVERIFIED_SEQUENCES = {"interval5_HKisland_GNSS_Evening"}


def site_group(seq):
    s = seq.replace("interval5_", "")
    for site in SITE_ORDER:
        if s.startswith(site):
            return site
    return "Other"


def seq_sort_key(seq):
    return (SITE_ORDER.index(site_group(seq)), seq)


def load():
    per_seq = defaultdict(dict)   # seq -> class -> (freq_within_seq, frac_images)
    class_names = []
    n_images_per_seq = {}
    global_freq = {}
    global_images_containing = {}
    n_images_total = 0

    with open(IN_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            seq = r["sequence"]
            cname = r["class"]
            if seq == "GLOBAL":
                global_freq[cname] = float(r["pixel_freq_within_sequence"])
                global_images_containing[cname] = int(r["n_images_containing"])
                n_images_total = int(r["n_images"])
                if cname not in class_names:
                    class_names.append(cname)
                continue
            n_images_per_seq[seq] = int(r["n_images"])
            per_seq[seq][cname] = (
                float(r["pixel_freq_within_sequence"]),
                float(r["frac_images_containing"]),
            )

    sequences = sorted(per_seq.keys(), key=seq_sort_key)
    return sequences, class_names, per_seq, n_images_per_seq, global_freq, global_images_containing, n_images_total


def plot_global_frequency(class_names, global_freq, n_images_total):
    order = sorted(class_names, key=lambda c: global_freq[c])
    freqs = [global_freq[c] for c in order]

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(order))
    ax.barh(y, freqs, color=CAT_BLUE, height=0.62, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9, color=INK_PRIMARY)
    ax.set_xscale("log")
    ax.set_xlim(min(freqs) * 0.5, 1.0)
    ax.set_xlabel("Pixel frequency (log scale, fraction of all pixels)", fontsize=9, color=INK_SECONDARY)
    ax.set_title(
        f"UAVScenes 19-class global pixel frequency  (n={n_images_total} images, 20 sequences)",
        fontsize=11, color=INK_PRIMARY, loc="left", pad=12,
    )
    ax.grid(axis="x", which="major", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.grid(axis="x", which="minor", color=GRIDLINE, linewidth=0.4, zorder=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)

    for yi, f in zip(y, freqs):
        ax.text(f * 1.15, yi, f"{f * 100:.4f}%", va="center", fontsize=7.5, color=INK_SECONDARY)

    fig.tight_layout()
    out = os.path.join(IMAGES_DIR, "01_global_class_frequency.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"wrote {out}")


def _heatmap_base(sequences, class_names, values, cmap, norm, cbar_label, title):
    fig, ax = plt.subplots(figsize=(11, 9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8, color=INK_PRIMARY)
    ax.set_yticks(np.arange(len(sequences)))
    ylabels = []
    for s in sequences:
        short = s.replace("interval5_", "")
        if s in UNVERIFIED_SEQUENCES:
            short += "  (unverified)"
        ylabels.append(short)
    ax.set_yticklabels(ylabels, fontsize=8, color=INK_PRIMARY)

    # Site-group separators
    prev_group = None
    for i, s in enumerate(sequences):
        g = site_group(s)
        if prev_group is not None and g != prev_group:
            ax.axhline(i - 0.5, color=INK_PRIMARY, linewidth=1.0, alpha=0.35)
        prev_group = g

    ax.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(sequences), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label, fontsize=9, color=INK_SECONDARY)
    cbar.ax.tick_params(labelsize=7.5, colors=INK_MUTED)
    cbar.outline.set_visible(False)

    ax.set_title(title, fontsize=11, color=INK_PRIMARY, loc="left", pad=12)
    fig.tight_layout()
    return fig, ax


def plot_freq_heatmap(sequences, class_names, per_seq):
    values = np.zeros((len(sequences), len(class_names)))
    for i, seq in enumerate(sequences):
        for j, c in enumerate(class_names):
            values[i, j] = per_seq[seq].get(c, (0.0, 0.0))[0]

    masked = np.ma.masked_where(values <= 0, values)
    cmap = SEQ_CMAP.copy()
    cmap.set_bad(ZERO_GRAY)
    norm = LogNorm(vmin=max(masked.min(), 1e-6), vmax=masked.max())

    fig, ax = _heatmap_base(
        sequences, class_names, masked, cmap, norm,
        cbar_label="Pixel freq within sequence (log)",
        title="Class pixel share by sequence  (gray = class absent entirely from that sequence)",
    )
    out = os.path.join(IMAGES_DIR, "02_sequence_class_heatmap_freq.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"wrote {out}")


def plot_coverage_heatmap(sequences, class_names, per_seq):
    values = np.zeros((len(sequences), len(class_names)))
    for i, seq in enumerate(sequences):
        for j, c in enumerate(class_names):
            values[i, j] = per_seq[seq].get(c, (0.0, 0.0))[1]

    masked = np.ma.masked_where(values <= 0, values)
    cmap = SEQ_CMAP.copy()
    cmap.set_bad(ZERO_GRAY)

    fig, ax = _heatmap_base(
        sequences, class_names, masked, cmap, None,
        cbar_label="Fraction of images in sequence containing class",
        title="Class presence (image coverage) by sequence  (gray = class never appears in that sequence)",
    )
    out = os.path.join(IMAGES_DIR, "03_sequence_class_heatmap_coverage.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"wrote {out}")


def plot_rare_classes(sequences, class_names, per_seq, global_freq):
    rarest = sorted(class_names, key=lambda c: global_freq[c])
    rarest = [c for c in rarest if c != "Background"][:5]

    fig, axes = plt.subplots(len(rarest), 1, figsize=(12, 3.0 * len(rarest)), facecolor=SURFACE)
    x = np.arange(len(sequences))
    labels = [s.replace("interval5_", "") for s in sequences]
    colors = [SITE_COLOR[site_group(s)] for s in sequences]

    for ax, cname in zip(axes, rarest):
        vals = [per_seq[s].get(cname, (0.0, 0.0))[0] for s in sequences]
        ax.set_facecolor(SURFACE)
        ax.bar(x, vals, color=colors, width=0.7, zorder=3)
        ax.set_title(f"{cname}  (global freq {global_freq[cname] * 100:.4f}%)",
                     fontsize=10, color=INK_PRIMARY, loc="left")
        ax.set_xticks(x)
        if ax is axes[-1]:
            ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7.5, color=INK_PRIMARY)
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis="y", labelsize=7.5, colors=INK_MUTED)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.7, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(BASELINE)
        ax.spines["bottom"].set_color(BASELINE)
        # site-group separators
        prev_group = None
        for i, s in enumerate(sequences):
            g = site_group(s)
            if prev_group is not None and g != prev_group:
                ax.axvline(i - 0.5, color=INK_PRIMARY, linewidth=0.8, alpha=0.25)
            prev_group = g

    handles = [plt.Rectangle((0, 0), 1, 1, color=SITE_COLOR[g]) for g in SITE_ORDER]
    fig.legend(handles, SITE_ORDER, loc="upper right", ncol=4, frameon=False,
               fontsize=9, labelcolor=INK_PRIMARY, bbox_to_anchor=(0.98, 1.0))
    fig.suptitle(
        "The 5 rarest 19-class classes, per sequence -- site-exclusivity is visible directly",
        fontsize=12, color=INK_PRIMARY, x=0.02, ha="left", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = os.path.join(IMAGES_DIR, "04_rare_classes_per_sequence.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    (sequences, class_names, per_seq, n_images_per_seq,
     global_freq, global_images_containing, n_images_total) = load()

    plot_global_frequency(class_names, global_freq, n_images_total)
    plot_freq_heatmap(sequences, class_names, per_seq)
    plot_coverage_heatmap(sequences, class_names, per_seq)
    plot_rare_classes(sequences, class_names, per_seq, global_freq)


if __name__ == "__main__":
    main()
