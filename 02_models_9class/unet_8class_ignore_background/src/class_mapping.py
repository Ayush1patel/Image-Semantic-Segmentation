"""
Class remapping for the vanilla-UNet UAVScenes run.

Unlike the UNet++ project (10 merged classes, background kept as a real
class), this uses the taxonomy from class distribution.txt verbatim:
8 named classes, plus a genuine IGNORED bucket (background, solar_board,
umbrella, and unnamed/reserved raw ids) that is excluded from both the
loss and the metrics via ignore_index - never predicted, never scored.

  PMNet ID  Class             Raw UAVScenes ids merged into it
  0         Building          1 (roof), 17 (transparent_roof)
  1         Road              2 (dirt_motor_road), 3 (paved_motor_road),
                               10 (airstrip), 18 (car_park), 19 (paved_walk)
  2         Water             4 (river), 5 (pool)
  3         Vegetation        13 (green_field), 14 (wild_field)
  4         Bridge            6 (bridge)
  5         Container         9 (container)
  6         Traffic_Barrier   11 (traffic_barrier)
  7         Vehicle           20 (sedan), 24 (truck)
  IGNORE    -                 0 (background), 15 (solar_board),
                               16 (umbrella), 7, 8, 12, 21, 22, 23, 25 (unnamed)

Note: Airstrip is folded into Road here - a deliberate difference from the
UNet++ project, where it was kept as its own class.
"""
import numpy as np

# Raw id -> name, straight from the dataset's cmap.py (26 slots, 0-25).
RAW_CLASSES = {
    0: "background", 1: "roof", 2: "dirt_motor_road", 3: "paved_motor_road",
    4: "river", 5: "pool", 6: "bridge", 7: "unused_7", 8: "unused_8",
    9: "container", 10: "airstrip", 11: "traffic_barrier", 12: "unused_12",
    13: "green_field", 14: "wild_field", 15: "solar_board", 16: "umbrella",
    17: "transparent_roof", 18: "car_park", 19: "paved_walk", 20: "sedan",
    21: "unused_21", 22: "unused_22", 23: "unused_23", 24: "truck", 25: "unused_25",
}

NUM_CLASSES = 8
IGNORE_INDEX = 255  # excluded from loss (CrossEntropy/Dice) and from metrics

NEW_CLASS_NAMES = [
    "Building",         # 0
    "Road",              # 1
    "Water",             # 2
    "Vegetation",        # 3
    "Bridge",            # 4
    "Container",         # 5
    "Traffic_Barrier",  # 6
    "Vehicle",           # 7
]

# Raw id -> new (trainable) id, or IGNORE_INDEX.
RAW_TO_NEW = {
    0: IGNORE_INDEX,   # background -> IGNORED
    1: 0,               # roof -> Building
    2: 1,               # dirt_motor_road -> Road
    3: 1,               # paved_motor_road -> Road
    4: 2,               # river -> Water
    5: 2,               # pool -> Water
    6: 4,               # bridge -> Bridge
    7: IGNORE_INDEX,    # unused -> IGNORED
    8: IGNORE_INDEX,    # unused -> IGNORED
    9: 5,               # container -> Container
    10: 1,              # airstrip -> Road
    11: 6,              # traffic_barrier -> Traffic_Barrier
    12: IGNORE_INDEX,   # unused -> IGNORED
    13: 3,              # green_field -> Vegetation
    14: 3,              # wild_field -> Vegetation
    15: IGNORE_INDEX,   # solar_board -> IGNORED
    16: IGNORE_INDEX,   # umbrella -> IGNORED
    17: 0,              # transparent_roof -> Building
    18: 1,              # car_park -> Road
    19: 1,              # paved_walk -> Road
    20: 7,              # sedan -> Vehicle
    21: IGNORE_INDEX,   # unused -> IGNORED
    22: IGNORE_INDEX,   # unused -> IGNORED
    23: IGNORE_INDEX,   # unused -> IGNORED
    24: 7,              # truck -> Vehicle
    25: IGNORE_INDEX,   # unused -> IGNORED
}

# LUT for O(1) vectorized remap: new = REMAP_LUT[raw]. Any raw id not
# explicitly listed above (out-of-range bytes) also falls back to IGNORE_INDEX.
REMAP_LUT = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for _raw_id, _new_id in RAW_TO_NEW.items():
    REMAP_LUT[_raw_id] = _new_id

# Visualization palette (RGB) for the 8 trainable classes.
NEW_PALETTE = np.array([
    [119, 11, 32],    # Building
    [128, 64, 128],   # Road
    [173, 216, 230],  # Water
    [107, 142, 35],   # Vegetation
    [150, 100, 100],  # Bridge
    [250, 170, 30],   # Container
    [102, 102, 156],  # Traffic_Barrier
    [0, 0, 142],       # Vehicle
], dtype=np.uint8)

IGNORE_COLOR = np.array([0, 0, 0], dtype=np.uint8)  # shown black in visualizations


def remap_label(raw_label: np.ndarray) -> np.ndarray:
    """Vectorized raw-id -> {0..7, IGNORE_INDEX} remap for a uint8 label array."""
    return REMAP_LUT[raw_label]


def colorize(label: np.ndarray) -> np.ndarray:
    """Label array (values 0..7 or IGNORE_INDEX) -> RGB visualization image."""
    out = np.empty(label.shape + (3,), dtype=np.uint8)
    ignore_mask = label == IGNORE_INDEX
    safe = np.where(ignore_mask, 0, label)
    out[:] = NEW_PALETTE[safe]
    out[ignore_mask] = IGNORE_COLOR
    return out
