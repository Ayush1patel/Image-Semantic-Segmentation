"""
Class remapping for the vanilla-UNet UAVScenes run.

The 8 named classes from class distribution.txt keep their exact ids and
order; Background is appended as a 9th class (id 8). Background absorbs
raw id 0 plus every raw id that has no named class in this dataset variant
(solar_board, umbrella, and the unnamed/reserved slots), so *every* pixel
now belongs to a trainable, scored class - nothing is ignored any more.

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
  8         Background        0 (background), 15 (solar_board),
                               16 (umbrella), 7, 8, 12, 21, 22, 23, 25 (unnamed)

Background is ~15% of all labelled pixels, so promoting it from an ignored
bucket to a real class changes both the loss and every reported metric -
checkpoints trained against the previous 8-class head are not loadable here.

Note: Airstrip is folded into Road - a deliberate difference from the
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

NEW_CLASS_NAMES = [
    "Building",         # 0
    "Road",              # 1
    "Water",             # 2
    "Vegetation",        # 3
    "Bridge",            # 4
    "Container",         # 5
    "Traffic_Barrier",  # 6
    "Vehicle",           # 7
    "Background",        # 8
]
NUM_CLASSES = len(NEW_CLASS_NAMES)  # 9

BACKGROUND_INDEX = 8

# Every raw id now maps to a real class, so there is no ignore bucket.
# Kept as a module-level name (rather than deleted) because the loss,
# metrics and eval code all take an ignore_index argument; None means
# "score every pixel".
IGNORE_INDEX = None

# Raw id -> new (trainable) id.
RAW_TO_NEW = {
    0: 8,   # background -> Background
    1: 0,   # roof -> Building
    2: 1,   # dirt_motor_road -> Road
    3: 1,   # paved_motor_road -> Road
    4: 2,   # river -> Water
    5: 2,   # pool -> Water
    6: 4,   # bridge -> Bridge
    7: 8,   # unused -> Background
    8: 8,   # unused -> Background
    9: 5,   # container -> Container
    10: 1,  # airstrip -> Road
    11: 6,  # traffic_barrier -> Traffic_Barrier
    12: 8,  # unused -> Background
    13: 3,  # green_field -> Vegetation
    14: 3,  # wild_field -> Vegetation
    15: 8,  # solar_board -> Background
    16: 8,  # umbrella -> Background
    17: 0,  # transparent_roof -> Building
    18: 1,  # car_park -> Road
    19: 1,  # paved_walk -> Road
    20: 7,  # sedan -> Vehicle
    21: 8,  # unused -> Background
    22: 8,  # unused -> Background
    23: 8,  # unused -> Background
    24: 7,  # truck -> Vehicle
    25: 8,  # unused -> Background
}

# LUT for O(1) vectorized remap: new = REMAP_LUT[raw]. Any raw id not
# explicitly listed above (out-of-range bytes) also falls back to Background.
REMAP_LUT = np.full(256, BACKGROUND_INDEX, dtype=np.uint8)
for _raw_id, _new_id in RAW_TO_NEW.items():
    REMAP_LUT[_raw_id] = _new_id

# Visualization palette (RGB) for the 9 trainable classes.
NEW_PALETTE = np.array([
    [119, 11, 32],    # Building
    [128, 64, 128],   # Road
    [173, 216, 230],  # Water
    [107, 142, 35],   # Vegetation
    [150, 100, 100],  # Bridge
    [250, 170, 30],   # Container
    [102, 102, 156],  # Traffic_Barrier
    [0, 0, 142],       # Vehicle
    [0, 0, 0],         # Background (black, as in the official cmap)
], dtype=np.uint8)

# Colour for pixels that carry no label at all in a given view - used only
# by the LiDAR-masked visualizations, where most pixels have no LiDAR return.
NO_DATA_COLOR = np.array([64, 64, 64], dtype=np.uint8)
NO_DATA_INDEX = 255


def remap_label(raw_label: np.ndarray) -> np.ndarray:
    """Vectorized raw-id -> {0..8} remap for a uint8 label array."""
    return REMAP_LUT[raw_label]


def colorize(label: np.ndarray) -> np.ndarray:
    """Label array (values 0..8, or NO_DATA_INDEX) -> RGB visualization image."""
    out = np.empty(label.shape + (3,), dtype=np.uint8)
    no_data = label == NO_DATA_INDEX
    safe = np.where(no_data, 0, label)
    out[:] = NEW_PALETTE[safe]
    out[no_data] = NO_DATA_COLOR
    return out
