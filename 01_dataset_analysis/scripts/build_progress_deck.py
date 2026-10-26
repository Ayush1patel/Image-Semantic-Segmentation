"""
Build the two-week progress presentation (UPerNet training + dataset
spatial-split analysis) in the same visual style as
../../saqibbMultimodalReport.pptx (Inter font, slate/navy/sky color
system, card-based layout, kicker+title header pattern) -- reverse
engineered from that file's shape formatting rather than using a .potx
template, since python-pptx has no format-preserving "load and extend"
path for arbitrary Google-Slides-exported decks.

Usage:
    python 01_dataset_analysis/scripts/build_progress_deck.py
"""
import os

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
UPERNET = os.path.join(ROOT, "upernet_dice_ce")
DATASET_ANALYSIS = os.path.join(ROOT, "01_dataset_analysis")
CDA = os.path.join(DATASET_ANALYSIS, "class_distribution_report")
OUT_PATH = os.path.join(ROOT, "UAVScenes_TwoWeek_Progress_Report.pptx")

# ---------------------------------------------------------------------
# Palette + type, matched to saqibbMultimodalReport.pptx
# ---------------------------------------------------------------------
NAVY = RGBColor(0x0F, 0x17, 0x2A)
SKY = RGBColor(0x38, 0xBD, 0xF8)
BODY = RGBColor(0x64, 0x74, 0x8B)
MUTED = RGBColor(0x94, 0xA3, 0xB8)
CARD_FILL = RGBColor(0xF8, 0xFA, 0xFC)
CARD_BORDER = RGBColor(0xE2, 0xE8, 0xF0)
STAT_FILL = RGBColor(0xCB, 0xD5, 0xE1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GOOD = RGBColor(0x16, 0xA3, 0x4A)
WARN = RGBColor(0xD9, 0x77, 0x06)

FONT = "Inter"
FONT_XB = "Inter ExtraBold"
FONT_MED = "Inter Medium"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN = Inches(1.042)
CONTENT_W = Inches(11.25)


def new_presentation():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def blank_slide(prs):
    layout = prs.slide_layouts[6]  # blank
    slide = prs.slides.add_slide(layout)
    # white background
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = WHITE
    return slide


def _set_run(run, text, font=FONT, size=11, bold=False, color=NAVY, italic=False):
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color


def add_textbox(slide, left, top, width, height, word_wrap=True):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = word_wrap
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    return box, tf


def add_kicker_title(slide, kicker, title, top=Inches(0.625)):
    _, tf = add_textbox(slide, MARGIN, top, CONTENT_W, Inches(0.24))
    _set_run(tf.paragraphs[0].add_run(), kicker.upper(), font=FONT_XB, size=11, color=SKY)

    _, tf2 = add_textbox(slide, MARGIN, top + Inches(0.31), CONTENT_W, Inches(0.55))
    _set_run(tf2.paragraphs[0].add_run(), title, font=FONT_XB, size=24, color=NAVY)


def add_card(slide, left, top, width, height, fill=CARD_FILL, border=CARD_BORDER, radius=0.06):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    try:
        shape.adjustments[0] = radius
    except Exception:
        pass
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = border
    shape.line.width = Pt(1.0)
    shape.shadow.inherit = False
    return shape


def add_bullet_card(slide, left, top, width, height, header, bullets, header_color=BODY):
    add_card(slide, left, top, width, height)
    inset = Inches(0.208)
    _, tf = add_textbox(slide, left + inset, top + inset, width - 2 * inset, height - 2 * inset)

    p0 = tf.paragraphs[0]
    _set_run(p0.add_run(), header, font=FONT, size=13, bold=True, color=header_color)
    p0.space_after = Pt(10)

    for label, rest in bullets:
        p = tf.add_paragraph()
        p.space_after = Pt(8)
        r1 = p.add_run()
        _set_run(r1, f"• {label}", font=FONT, size=11, bold=True, color=NAVY)
        if rest:
            r2 = p.add_run()
            _set_run(r2, f" {rest}", font=FONT, size=11, bold=False, color=BODY)


def add_stat_card(slide, left, top, width, height, label, value, value_color=NAVY, value_size=24):
    add_card(slide, left, top, width, height)
    inset = Inches(0.2)
    _, tf = add_textbox(slide, left + inset, top + inset, width - 2 * inset, height - 2 * inset)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p0 = tf.paragraphs[0]
    p0.alignment = PP_ALIGN.LEFT
    _set_run(p0.add_run(), label.upper(), font=FONT, size=9, bold=True, color=MUTED)
    for i, line in enumerate(value.split("\n")):
        p = tf.add_paragraph()
        if i == 0:
            p.space_before = Pt(4)
        _set_run(p.add_run(), line, font=FONT_XB, size=value_size, bold=True, color=value_color)


def set_cell(cell, text, font=FONT, size=10.5, bold=False, color=NAVY, fill=None, align=PP_ALIGN.LEFT):
    cell.margin_left = Inches(0.1)
    cell.margin_right = Inches(0.1)
    cell.margin_top = Inches(0.05)
    cell.margin_bottom = Inches(0.05)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    if fill is not None:
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill
    else:
        cell.fill.solid()
        cell.fill.fore_color.rgb = WHITE
    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    _set_run(p.add_run(), text, font=font, size=size, bold=bold, color=color)


def add_table(slide, left, top, width, height, headers, rows, col_widths=None):
    n_rows = len(rows) + 1
    n_cols = len(headers)
    gtable = slide.shapes.add_table(n_rows, n_cols, left, top, width, height)
    table = gtable.table
    # kill default banding style visuals by using explicit fills
    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = w
    for j, h in enumerate(headers):
        set_cell(table.cell(0, j), h, bold=True, color=WHITE, fill=NAVY, size=10.5)
    for i, row in enumerate(rows, start=1):
        fill = CARD_FILL if i % 2 == 0 else WHITE
        for j, val in enumerate(row):
            set_cell(table.cell(i, j), str(val), fill=fill, size=10.5)
    return table


def add_picture_fit(slide, path, left, top, max_w, max_h, align="center"):
    from PIL import Image
    with Image.open(path) as im:
        iw, ih = im.size
    ar = iw / ih
    box_ar = max_w / max_h
    if ar > box_ar:
        w = max_w
        h = int(max_w / ar)
    else:
        h = max_h
        w = int(max_h * ar)
    x = left + (max_w - w) // 2 if align == "center" else left
    y = top + (max_h - h) // 2
    slide.shapes.add_picture(path, x, y, width=w, height=h)


def add_footer(slide, text, n):
    _, tf = add_textbox(slide, MARGIN, Inches(7.12), Inches(9), Inches(0.3))
    _set_run(tf.paragraphs[0].add_run(), text, font=FONT_MED, size=8, color=MUTED)
    _, tf2 = add_textbox(slide, Inches(12.3), Inches(7.12), Inches(0.6), Inches(0.3))
    p = tf2.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    _set_run(p.add_run(), str(n), font=FONT_MED, size=8, color=MUTED)


# =======================================================================
def build():
    prs = new_presentation()

    # ---------------- Slide 1: Title -----------------------------------
    s = blank_slide(prs)
    _, tf = add_textbox(s, MARGIN, Inches(0.9), CONTENT_W, Inches(1.8))
    p0 = tf.paragraphs[0]
    _set_run(p0.add_run(), "UAVScenes Semantic Segmentation", font=FONT_XB, size=38, color=NAVY)
    p1 = tf.add_paragraph()
    _set_run(p1.add_run(), "Two-Week Progress Report", font=FONT_XB, size=38, color=NAVY)

    card_w = Inches(3.542)
    card_h = Inches(1.2)
    add_stat_card(s, Inches(2.047), Inches(3.0), card_w, card_h,
                  "Model Trained", "UPerNet + ResNet-34\n19-class @ 256×256", value_size=16)
    add_stat_card(s, Inches(7.744), Inches(3.0), card_w, card_h,
                  "Key Finding", "Train/Val/Test split\nleaks ground truth", value_size=16, value_color=WARN)

    line = s.shapes.add_connector(1, MARGIN, Inches(5.5), MARGIN + CONTENT_W, Inches(5.5))
    line.line.color.rgb = CARD_BORDER
    line.line.width = Pt(1)

    _, tf = add_textbox(s, MARGIN, Inches(5.75), Inches(5.2), Inches(0.75))
    pA = tf.paragraphs[0]
    _set_run(pA.add_run(), "UNDER SUPERVISION OF", font=FONT_MED, size=10, color=MUTED)
    pB = tf.add_paragraph()
    _set_run(pB.add_run(), "Prof. Jaya Sreevalsan Nair", font=FONT, size=14, color=NAVY)

    _, tf = add_textbox(s, Inches(6.625), Inches(5.75), Inches(5.625), Inches(0.75))
    pA = tf.paragraphs[0]
    _set_run(pA.add_run(), "PRESENTED BY", font=FONT_MED, size=10, color=MUTED)
    pB = tf.add_paragraph()
    _set_run(pB.add_run(), "Aryan Malik  ·  September 2026", font=FONT, size=14, color=NAVY)

    # ---------------- Slide 2: UPerNet setup ----------------------------
    s = blank_slide(prs)
    add_kicker_title(s, "Part 1 — Model Training", "UPerNet (ResNet-34) — Training Setup")
    headers = ["Parameter", "Value"]
    rows = [
        ["Input resolution", "256 × 256"],
        ["Classes", "19 (Background + 18 named UAVScenes classes)"],
        ["Encoder", "ResNet-34 (ImageNet pretrained)"],
        ["Decoder", "UPerNet (Xiao et al., 2018)"],
        ["Loss", "0.5 × Dice + 0.5 × class-weighted Cross-Entropy"],
        ["Optimizer", "AdamW, lr 3e-4, weight decay 1e-4"],
        ["LR schedule", "3-epoch linear warmup → cosine decay to 0"],
        ["Batch size", "32"],
        ["Epochs", "100 (no early stop triggered)"],
        ["Mixed precision / Grad clip", "AMP enabled / max norm 5.0"],
    ]
    add_table(s, MARGIN, Inches(1.7), CONTENT_W, Inches(4.9), headers, rows,
              col_widths=[Inches(4.5), Inches(6.75)])
    add_footer(s, "UAVScenes Two-Week Progress Report", 2)

    # ---------------- Slide 3: Headline results --------------------------
    s = blank_slide(prs)
    add_kicker_title(s, "Part 1 — Model Training", "Headline Results — Best Checkpoint (Epoch 93)")
    stats = [
        ("Val mIoU (18 classes)", "0.8275"),
        ("Pixel Accuracy", "0.9597"),
        ("Mean F1", "0.8999"),
        ("Mean Precision", "0.8847"),
        ("Mean Recall", "0.9179"),
    ]
    card_w = Inches(2.14)
    card_h = Inches(1.35)
    gap = Inches(0.104)
    x = MARGIN
    for label, val in stats:
        add_stat_card(s, x, Inches(1.9), card_w, card_h, label, val, value_color=SKY, value_size=22)
        x += card_w + gap

    add_bullet_card(s, MARGIN, Inches(3.55), CONTENT_W, Inches(3.1), "Notes", [
        ("Training completed all 100 epochs;", "no early stop triggered — performance plateaued from ~epoch 85 onward."),
        ("14 of 19 classes exceed 0.77 IoU;", "5 small/rare/dynamic classes (Sedan, Traffic_Barrier, Truck, Umbrella, Car_Park) remain the improvement target."),
        ("Operational note:", "training was interrupted once by a hard GPU/driver crash mid-epoch-93; recovered automatically from the last checkpoint (epoch 92) with zero loss of progress."),
    ])
    add_footer(s, "UAVScenes Two-Week Progress Report", 3)

    # ---------------- Slide 4: Training curves ---------------------------
    s = blank_slide(prs)
    add_kicker_title(s, "Part 1 — Model Training", "Training Curves")
    add_picture_fit(s, os.path.join(UPERNET, "analysis", "images", "training_dashboard.png"),
                     MARGIN, Inches(1.65), CONTENT_W, Inches(4.6))
    _, tf = add_textbox(s, MARGIN, Inches(6.35), CONTENT_W, Inches(0.6))
    _set_run(tf.paragraphs[0].add_run(),
             "Loss falls smoothly for all 100 epochs; val loss bottoms out early (~epoch 7) then drifts up while mIoU keeps improving. Recall leads precision throughout.",
             font=FONT, size=10.5, color=BODY)
    add_footer(s, "UAVScenes Two-Week Progress Report", 4)

    # ---------------- Slide 5: Per-class IoU ------------------------------
    s = blank_slide(prs)
    add_kicker_title(s, "Part 1 — Model Training", "Per-Class IoU (Best Epoch)")
    add_picture_fit(s, os.path.join(UPERNET, "analysis", "images", "per_class_iou_all.png"),
                     MARGIN, Inches(1.65), Inches(7.0), Inches(4.9))
    headers = ["Weakest classes", "IoU"]
    rows = [
        ["Sedan", "0.545"],
        ["Traffic_Barrier", "0.576"],
        ["Truck", "0.670"],
        ["Umbrella", "0.695"],
        ["Car_Park", "0.743"],
    ]
    add_table(s, Inches(8.25), Inches(1.9), Inches(4.04), Inches(2.2), headers, rows,
              col_widths=[Inches(2.7), Inches(1.34)])
    _, tf = add_textbox(s, Inches(8.25), Inches(4.35), Inches(4.04), Inches(2.2))
    _set_run(tf.paragraphs[0].add_run(),
             "Large, texturally-distinct land-cover classes (fields, water, airstrip) are near-solved. The weakest classes are consistently the small/thin/rare objects.",
             font=FONT, size=10.5, color=BODY)
    add_footer(s, "UAVScenes Two-Week Progress Report", 5)

    # ---------------- Slide 6: Qualitative progression --------------------
    s = blank_slide(prs)
    add_kicker_title(s, "Part 1 — Model Training", "Qualitative Progression")
    imgs = [
        ("epoch_001.png", "Epoch 1 (near-random)"),
        ("epoch_050.png", "Epoch 50"),
        ("epoch_093.png", "Epoch 93 (best)"),
    ]
    col_w = Inches(3.542)
    gap = Inches(0.312)
    x = MARGIN
    for fname, caption in imgs:
        add_card(s, x, Inches(1.7), col_w, Inches(4.6))
        add_picture_fit(s, os.path.join(UPERNET, "results_256", "progress", fname),
                         x + Inches(0.08), Inches(1.78), col_w - Inches(0.16), Inches(4.1))
        _, tf = add_textbox(s, x, Inches(5.95), col_w, Inches(0.3))
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        _set_run(p.add_run(), caption, font=FONT, size=10.5, bold=True, color=NAVY)
        x += col_w + gap
    add_footer(s, "UAVScenes Two-Week Progress Report", 6)

    # ---------------- Slide 7: Data analysis motivation --------------------
    s = blank_slide(prs)
    add_kicker_title(s, "Part 2 — Dataset Analysis", "Why We Re-Examined the Train/Val/Test Split")
    add_bullet_card(s, MARGIN, Inches(1.7), CONTENT_W, Inches(2.35), "The current split", [
        ("Whole flight runs are assigned to a set:", "e.g. AMtown01 → train, AMtown02 → val, AMtown03 → test, one run per site."),
        ("This is a spatial-CV assumption that was never checked:", "does a repeat flight over the same site actually cover new ground, or the same ground again?"),
    ])
    add_bullet_card(s, MARGIN, Inches(4.2), CONTENT_W, Inches(2.35), "What we did", [
        ("Scanned every labeled image in the dataset:", "24,126 images across all 20 sequences (not just the 16 currently used in a split), at raw 26-class pixel granularity."),
        ("Built per-image, per-sequence, and per-class statistics,", "then cross-checked flight geography using each sequence's RTK GPS log."),
    ])
    add_footer(s, "UAVScenes Two-Week Progress Report", 7)

    # ---------------- Slide 8: Global class distribution --------------------
    s = blank_slide(prs)
    add_kicker_title(s, "Part 2 — Dataset Analysis", "Global Class Distribution Is Heavily Imbalanced")
    add_picture_fit(s, os.path.join(CDA, "images", "01_global_class_frequency.png"),
                     MARGIN, Inches(1.6), Inches(8.6), Inches(4.9))
    add_bullet_card(s, Inches(9.9), Inches(1.6), Inches(2.38), Inches(4.9), "Range", [
        ("~4,000× spread:", "Green_Field at 39.0% down to Bridge at 0.009% of all pixels."),
        ("18 non-background classes,", "most under 3% pixel share individually."),
    ])
    add_footer(s, "UAVScenes Two-Week Progress Report", 8)

    # ---------------- Slide 9: Per-sequence heatmaps --------------------
    s = blank_slide(prs)
    add_kicker_title(s, "Part 2 — Dataset Analysis", "Class Presence by Sequence (Folder)")
    half_w = Inches(5.48)
    add_picture_fit(s, os.path.join(CDA, "images", "02_sequence_class_heatmap_freq.png"),
                     MARGIN, Inches(1.6), half_w, Inches(5.0))
    add_picture_fit(s, os.path.join(CDA, "images", "03_sequence_class_heatmap_coverage.png"),
                     MARGIN + half_w + Inches(0.29), Inches(1.6), half_w, Inches(5.0))
    add_footer(s, "Left: pixel share per sequence. Right: fraction of images containing the class. Gray = class entirely absent from that sequence.", 9)

    # ---------------- Slide 10: rare classes are site-exclusive -----------
    s = blank_slide(prs)
    add_kicker_title(s, "Part 2 — Dataset Analysis", "Rare Classes Are Site-Exclusive, Not Just Rare")
    add_picture_fit(s, os.path.join(CDA, "images", "04_rare_classes_per_sequence.png"),
                     MARGIN, Inches(1.6), Inches(7.6), Inches(4.9))
    headers = ["Class", "Present in", "Absent from"]
    rows = [
        ["Bridge, Pool, Solar_Board", "AMtown / AMvalley only", "All HKairport, HKisland"],
        ["Container, Traffic_Barrier", "HKairport / HKisland only", "All AMtown, AMvalley"],
        ["All 5 rarest classes", "—", "HKisland: zero of all five"],
    ]
    add_table(s, Inches(8.85), Inches(1.9), Inches(3.44), Inches(2.6), headers, rows,
              col_widths=[Inches(1.35), Inches(1.1), Inches(0.99)])
    _, tf = add_textbox(s, Inches(8.85), Inches(4.75), Inches(3.44), Inches(1.9))
    _set_run(tf.paragraphs[0].add_run(),
             "Consequence: holding out a whole site for spatial-CV would permanently delete some classes from either train or test.",
             font=FONT, size=10.5, bold=True, color=WARN)
    add_footer(s, "UAVScenes Two-Week Progress Report", 10)

    # ---------------- Slide 11: spatial leakage confirmed -----------------
    s = blank_slide(prs)
    add_kicker_title(s, "Part 2 — Dataset Analysis", "Spatial Leakage — Confirmed With GPS, Not Guessed")
    _, tf = add_textbox(s, MARGIN, Inches(1.55), CONTENT_W, Inches(0.5))
    _set_run(tf.paragraphs[0].add_run(),
             "Using each sequence's RTK GPS log: distance from every val/test image to its nearest train image, per site.",
             font=FONT, size=11, color=BODY)
    headers = ["Site", "Median dist. val→train", "Median dist. test→train", "Train's own frame spacing", "% within 5m of a train frame"]
    rows = [
        ["AMtown", "0.81 m", "0.74 m", "1.62 m", "100%"],
        ["AMvalley", "0.42 m", "0.61 m", "1.62 m", "~100%"],
        ["HKairport", "0.43 m", "0.51 m", "1.22 m", "100%"],
        ["HKisland", "0.39 m", "0.22 m", "1.20 m", "~99.5%"],
    ]
    add_table(s, MARGIN, Inches(2.2), CONTENT_W, Inches(1.9), headers, rows,
              col_widths=[Inches(1.6), Inches(2.5), Inches(2.5), Inches(2.5), Inches(2.15)])
    add_bullet_card(s, MARGIN, Inches(4.4), CONTENT_W, Inches(2.15), "Reading", [
        ("Val/test images sit closer to a train image", "than consecutive frames within the train flight itself sit to each other."),
        ("This means 01 / 02 / 03 are not independent surveys of a site —", "they are the same pre-programmed GPS flight path, re-flown."),
    ])
    add_footer(s, "UAVScenes Two-Week Progress Report", 11)

    # ---------------- Slide 12: what this means ---------------------------
    s = blank_slide(prs)
    add_kicker_title(s, "Part 2 — Dataset Analysis", "What This Means for the Reported Numbers")
    add_bullet_card(s, MARGIN, Inches(1.7), CONTENT_W, Inches(2.35), "The metric, reinterpreted", [
        ("Same train/val/test convention underlies both projects,", "including the UPerNet run in Part 1 (val mIoU 0.8275) — this is not an isolated issue."),
        ("A small val→test gap does not prove generalization here:", "val and test are both near-duplicate locations of train, so a small gap is expected whether or not the model generalizes at all."),
    ])
    add_bullet_card(s, MARGIN, Inches(4.2), CONTENT_W, Inches(2.35), "What's still trustworthy vs. not", [
        ("Small/rare object classes (Sedan, Truck, Traffic_Barrier)", "are probably close to their real difficulty — memorization doesn't help much with high-variance dynamic objects."),
        ("Site-exclusive classes (Container, Bridge, Solar_Board, Airstrip)", "are the least trustworthy — the model may be recognizing a handful of specific revisited instances, not generalizing."),
    ])
    add_footer(s, "UAVScenes Two-Week Progress Report", 12)

    # ---------------- Slide 13: next steps ---------------------------------
    s = blank_slide(prs)
    add_kicker_title(s, "Next Steps", "Building a Genuine Spatial Train/Val/Test Split")
    steps = [
        ("Geo-index every frame", "using the per-sequence RTK GPS log (already validated as reliable, absolute UTM)."),
        ("Pool repeat missions per site (01+02+03+...)", "into one shared spatial index — they retrace the same path, so they can't be split by run number."),
        ("Tile each site's flight area into a grid,", "sized from where the rare/site-exclusive classes actually cluster."),
        ("Assign tiles to train/val/test", "with a spatial buffer between folds AND a check that every fold keeps coverage of every class present at that site."),
        ("Verify with the same nearest-neighbor distance check used in this analysis,", "before trusting any new split."),
        ("Re-train and re-evaluate,", "to get the first honest unseen-location mIoU for this dataset."),
    ]
    add_bullet_card(s, MARGIN, Inches(1.7), CONTENT_W, Inches(4.9), "Roadmap", steps)
    add_footer(s, "UAVScenes Two-Week Progress Report", 13)

    # ---------------- Slide 14: summary ------------------------------------
    s = blank_slide(prs)
    add_kicker_title(s, "Summary", "Where Things Stand")
    add_bullet_card(s, MARGIN, Inches(1.7), CONTENT_W, Inches(2.3), "Done", [
        ("UPerNet + ResNet-34 baseline trained,", "full 100 epochs, val mIoU 0.8275, 14/19 classes above 0.77 IoU."),
        ("Full-dataset class distribution analysis completed,", "24,126 images / 20 sequences, 19-class taxonomy, with visualizations."),
        ("Spatial leakage in the current split confirmed quantitatively,", "via each sequence's own GPS log — not assumed."),
    ])
    add_bullet_card(s, MARGIN, Inches(4.2), CONTENT_W, Inches(1.85), "Next", [
        ("Build and verify a tile-based, GPS-buffered spatial split,", "then re-train and re-evaluate for an honest generalization estimate."),
    ])
    add_footer(s, "UAVScenes Two-Week Progress Report", 14)

    prs.save(OUT_PATH)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    build()
