"""
Read 01_dataset_analysis/outputs/per_image_raw_class_counts.csv (produced by
scan_class_pixel_distribution.py) and produce, for the 19-class taxonomy
used by unet_dice_ce/src/class_mapping.py (every named
UAVScenes class kept separate; only the 7 unnamed/reserved raw ids fold
into Background):

  1. class_distribution_19class.csv
       per-sequence AND global pixel counts/freq/#images-containing, one
       row per (sequence, class) plus a "GLOBAL" sequence row per class.
  2. rare_class_top_images_19class.csv
       for each of the N rarest classes (by global freq), the top-K
       images ranked by that class's pixel count, with the pixel count
       and the fraction of that image it occupies.
  3. class_distribution_report.md
       human-readable summary: global table, per-sequence rare-class
       breakdown, and the top-offending images.

Only the 19-class taxonomy is used in this workspace from now on -- the
9-class scheme (and its class_distribution_9class.csv /
rare_class_top_images_9class.csv outputs) has been dropped entirely.

The raw-id -> new-id mapping table below is duplicated (not imported)
from unet_dice_ce/src/class_mapping.py's RAW_TO_NEW, on
purpose: this script lives outside that project tree and reading a
stable, well-documented 26-entry table here is less fragile than
sys.path-inserting that project's src/ dir. If that source file's mapping
changes, update the table below too.

Usage:
    python 01_dataset_analysis/scripts/summarize_class_distribution.py
"""
import csv
import os
from collections import defaultdict

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
IN_CSV = os.path.join(OUT_DIR, "per_image_raw_class_counts.csv")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "class_distribution_report")
REPORT_MD = os.path.join(REPORT_DIR, "class_distribution_report.md")

NUM_RAW_CLASSES = 26
TOP_K_IMAGES = 20
N_RAREST = 5

# Sequences the existing pipeline (data_utils.py UNVERIFIED_LABEL_SEQUENCES)
# refuses to include in any split automatically, pending manual
# confirmation that dense semantic labels actually exist. Flagged, not
# dropped, so this scan can show you what's actually in there.
UNVERIFIED_SEQUENCES = {"interval5_HKisland_GNSS_Evening"}

# ---------------------------------------------------------------------
# 19-class scheme, duplicated from
# unet_dice_ce/src/class_mapping.py RAW_TO_NEW
# ---------------------------------------------------------------------
CLASS_NAMES_19 = [
    "Background", "Building", "Dirt_Motor_Road", "Paved_Motor_Road",
    "River", "Pool", "Bridge", "Container", "Airstrip",
    "Traffic_Barrier", "Green_Field", "Wild_Field", "Solar_Board",
    "Umbrella", "Transparent_Roof", "Car_Park", "Paved_Walk", "Sedan",
    "Truck",
]
RAW_TO_19 = {i: 0 for i in range(NUM_RAW_CLASSES)}
RAW_TO_19.update({
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 9: 7, 10: 8, 11: 9,
    13: 10, 14: 11, 15: 12, 16: 13, 17: 14, 18: 15, 19: 16, 20: 17, 24: 18,
    # 7, 8, 12, 21, 22, 23, 25 -> Background (0), already default
})


def build_onehot(raw_to_new, num_new):
    m = np.zeros((NUM_RAW_CLASSES, num_new), dtype=np.int64)
    for raw_id, new_id in raw_to_new.items():
        m[raw_id, new_id] = 1
    return m


def load_rows():
    with open(IN_CSV, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        raw_start = header.index("raw_0")
        rows = []
        for r in reader:
            seq = r[0]
            stem = r[1]
            label_path = r[2]
            raw_counts = np.array(r[raw_start:raw_start + NUM_RAW_CLASSES], dtype=np.int64)
            rows.append((seq, stem, label_path, raw_counts))
    return rows


def summarize_scheme(rows, scheme_name, class_names, raw_to_new):
    onehot = build_onehot(raw_to_new, len(class_names))
    raw_matrix = np.stack([r[3] for r in rows])          # (n_images, 26)
    new_matrix = raw_matrix @ onehot                      # (n_images, n_classes)

    # ---- per-sequence + global totals -------------------------------
    seq_totals = defaultdict(lambda: np.zeros(len(class_names), dtype=np.int64))
    seq_images_containing = defaultdict(lambda: np.zeros(len(class_names), dtype=np.int64))
    global_totals = np.zeros(len(class_names), dtype=np.int64)
    global_images_containing = np.zeros(len(class_names), dtype=np.int64)

    for (seq, stem, label_path, _raw), new_counts in zip(rows, new_matrix):
        seq_totals[seq] += new_counts
        present = new_counts > 0
        seq_images_containing[seq] += present.astype(np.int64)
        global_totals += new_counts
        global_images_containing += present.astype(np.int64)

    n_images_per_seq = defaultdict(int)
    for r in rows:
        n_images_per_seq[r[0]] += 1
    n_images_total = len(rows)

    global_freq = global_totals / global_totals.sum()

    csv_path = os.path.join(OUT_DIR, f"class_distribution_{scheme_name}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "n_images", "class", "pixel_count", "pixel_freq_within_sequence",
                    "n_images_containing", "frac_images_containing"])
        for seq in sorted(seq_totals):
            tot = seq_totals[seq]
            freq = tot / max(tot.sum(), 1)
            for c_idx, cname in enumerate(class_names):
                w.writerow([seq, n_images_per_seq[seq], cname, int(tot[c_idx]),
                            f"{freq[c_idx]:.8f}", int(seq_images_containing[seq][c_idx]),
                            f"{seq_images_containing[seq][c_idx] / n_images_per_seq[seq]:.6f}"])
        for c_idx, cname in enumerate(class_names):
            w.writerow(["GLOBAL", n_images_total, cname, int(global_totals[c_idx]),
                        f"{global_freq[c_idx]:.8f}", int(global_images_containing[c_idx]),
                        f"{global_images_containing[c_idx] / n_images_total:.6f}"])
    print(f"[{scheme_name}] wrote {csv_path}")

    # ---- rarest classes + top offending images -----------------------
    rarest_order = np.argsort(global_freq)
    rarest_idx = [c for c in rarest_order if c != 0][:N_RAREST]  # skip Background

    top_images_path = os.path.join(OUT_DIR, f"rare_class_top_images_{scheme_name}.csv")
    top_rows_for_report = {}
    with open(top_images_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "rank", "sequence", "stem", "pixel_count",
                     "frac_of_image", "label_path"])
        for c_idx in rarest_idx:
            cname = class_names[c_idx]
            per_image_counts = new_matrix[:, c_idx]
            order = np.argsort(per_image_counts)[::-1][:TOP_K_IMAGES]
            entries = []
            for rank, i in enumerate(order, start=1):
                seq, stem, label_path, raw = rows[i]
                px = int(per_image_counts[i])
                total_px = int(raw.sum())
                frac = px / total_px if total_px else 0.0
                w.writerow([cname, rank, seq, stem, px, f"{frac:.6f}", label_path])
                entries.append((rank, seq, stem, px, frac))
            top_rows_for_report[cname] = entries
    print(f"[{scheme_name}] wrote {top_images_path}")

    return {
        "class_names": class_names,
        "global_totals": global_totals,
        "global_freq": global_freq,
        "global_images_containing": global_images_containing,
        "n_images_total": n_images_total,
        "seq_totals": seq_totals,
        "seq_images_containing": seq_images_containing,
        "n_images_per_seq": n_images_per_seq,
        "rarest_idx": rarest_idx,
        "top_rows_for_report": top_rows_for_report,
    }


def write_report(summary_19, sequences):
    lines = []
    lines.append("# UAVScenes full-dataset class pixel distribution (19-class)\n")
    lines.append(
        "Generated by `scan_class_pixel_distribution.py` + "
        "`summarize_class_distribution.py` against every sequence found "
        f"under `D:\\Saqib\\Multimodal_segmentation\\UAVScenes\\interval5_CAM_label` "
        f"({len(sequences)} sequences, {summary_19['n_images_total']} labeled images total). "
        "CPU-only, read-only, raw-id-level scan; aggregated to the 19-class "
        "taxonomy (unet_dice_ce/src/class_mapping.py) -- the "
        "only taxonomy used in this workspace going forward.\n"
    )
    if UNVERIFIED_SEQUENCES:
        lines.append(
            f"**Unverified sequence(s) included in this scan for visibility, "
            f"but excluded from `data_utils.py`'s automatic splits pending "
            f"manual confirmation of dense semantic labels:** "
            f"{', '.join(sorted(UNVERIFIED_SEQUENCES))}\n"
        )

    summary = summary_19
    lines.append("\n## Global distribution\n")
    lines.append(f"| Class | pixel freq | # images containing it (of {summary['n_images_total']}) |")
    lines.append("|---|---|---|")
    order = np.argsort(summary["global_freq"])[::-1]
    for c in order:
        lines.append(
            f"| {summary['class_names'][c]} | {summary['global_freq'][c]:.6f} "
            f"| {summary['global_images_containing'][c]} |"
        )

    lines.append("\n### Rarest classes, per-sequence pixel share\n")
    for c in summary["rarest_idx"]:
        cname = summary["class_names"][c]
        lines.append(f"\n**{cname}** (global freq {summary['global_freq'][c]:.6f})\n")
        lines.append("| Sequence | pixel_count | freq within sequence | # images containing it |")
        lines.append("|---|---|---|---|")
        rows_for_class = []
        for seq in sorted(summary["seq_totals"]):
            tot = summary["seq_totals"][seq]
            cnt = int(tot[c])
            freq = cnt / max(tot.sum(), 1)
            n_containing = int(summary["seq_images_containing"][seq][c])
            rows_for_class.append((seq, cnt, freq, n_containing))
        rows_for_class.sort(key=lambda r: r[1], reverse=True)
        for seq, cnt, freq, n_containing in rows_for_class:
            flag = " *(unverified)*" if seq in UNVERIFIED_SEQUENCES else ""
            lines.append(f"| {seq}{flag} | {cnt} | {freq:.6f} | {n_containing} |")

        lines.append(f"\nTop images for **{cname}**:\n")
        lines.append("| rank | sequence | stem | pixel_count | frac of image |")
        lines.append("|---|---|---|---|---|")
        for rank, seq, stem, px, frac in summary["top_rows_for_report"][cname][:10]:
            lines.append(f"| {rank} | {seq} | {stem} | {px} | {frac:.4f} |")

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {REPORT_MD}")


def main():
    rows = load_rows()
    sequences = sorted(set(r[0] for r in rows))
    print(f"Loaded {len(rows)} images across {len(sequences)} sequences")

    summary_19 = summarize_scheme(rows, "19class", CLASS_NAMES_19, RAW_TO_19)
    write_report(summary_19, sequences)


if __name__ == "__main__":
    main()
