"""
Class remapping for UAVScenes semantic segmentation.

Raw labels use the official 26-slot UAVScenes id space (see cmap.py in the
dataset root), of which 19 ids are actually named in the paper:
  - 1 background class
  - 16 static classes
  - 2 dynamic classes (sedan, truck)
The remaining 7 ids (7, 8, 12, 21, 22, 23, 25) are unused/reserved slots
that reuse leftover Cityscapes palette colors and never appear as
meaningful content in this dataset variant.

We merge those 19 classes down to 10 classes for training:
  0 Background       <- background, all unused/reserved ids, solar_board
                         (removed), umbrella (removed)
  1 Building          <- roof, transparent_roof
  2 Road              <- dirt_motor_road, paved_motor_road, paved_walk,
                         car_park
  3 Airstrip          <- airstrip
  4 Water             <- river, pool
  5 Bridge            <- bridge
  6 Container         <- container
  7 Traffic_Barrier    <- traffic_barrier
  8 Vegetation        <- green_field, wild_field
  9 Vehicle           <- sedan, truck (merged dynamic class)
"""
import numpy as np

# Raw id -> name, straight from the dataset's cmap.py (26 slots, 0-25).
RAW_CLASSES = {
    0: "background",
    1: "roof",
    2: "dirt_motor_road",
    3: "paved_motor_road",
    4: "river",
    5: "pool",
    6: "bridge",
    7: "unused_7",
    8: "unused_8",
    9: "container",
    10: "airstrip",
    11: "traffic_barrier",
    12: "unused_12",
    13: "green_field",
    14: "wild_field",
    15: "solar_board",
    16: "umbrella",
    17: "transparent_roof",
    18: "car_park",
    19: "paved_walk",
    20: "sedan",
    21: "unused_21",
    22: "unused_22",
    23: "unused_23",
    24: "truck",
    25: "unused_25",
}
NUM_RAW_CLASSES = 26

# New merged class ids and names.
NEW_CLASS_NAMES = [
    "Background",       # 0
    "Building",          # 1
    "Road",               # 2
    "Airstrip",           # 3
    "Water",              # 4
    "Bridge",             # 5
    "Container",          # 6
    "Traffic_Barrier",   # 7
    "Vegetation",         # 8
    "Vehicle",            # 9
]
NUM_CLASSES = len(NEW_CLASS_NAMES)  # 10

# Raw id -> new (merged) id.
RAW_TO_NEW = {
    0: 0,   # background
    1: 1,   # roof -> Building
    2: 2,   # dirt_motor_road -> Road
    3: 2,   # paved_motor_road -> Road
    4: 4,   # river -> Water
    5: 4,   # pool -> Water
    6: 5,   # bridge -> Bridge
    7: 0,   # unused -> Background
    8: 0,   # unused -> Background
    9: 6,   # container -> Container
    10: 3,  # airstrip -> Airstrip
    11: 7,  # traffic_barrier -> Traffic_Barrier
    12: 0,  # unused -> Background
    13: 8,  # green_field -> Vegetation
    14: 8,  # wild_field -> Vegetation
    15: 0,  # solar_board -> removed -> Background
    16: 0,  # umbrella -> removed -> Background
    17: 1,  # transparent_roof -> Building
    18: 2,  # car_park -> Road
    19: 2,  # paved_walk -> Road
    20: 9,  # sedan -> Vehicle
    21: 0,  # unused -> Background
    22: 0,  # unused -> Background
    23: 0,  # unused -> Background
    24: 9,  # truck -> Vehicle
    25: 0,  # unused -> Background
}

# LUT for O(1) vectorized remap: new = REMAP_LUT[raw]
REMAP_LUT = np.zeros(256, dtype=np.uint8)  # anything outside 0-25 also -> 0
for _raw_id, _new_id in RAW_TO_NEW.items():
    REMAP_LUT[_raw_id] = _new_id

# Visualization palette (RGB) for the merged classes.
NEW_PALETTE = np.array([
    [0, 0, 0],        # Background
    [119, 11, 32],    # Building
    [128, 64, 128],   # Road
    [81, 0, 81],      # Airstrip
    [173, 216, 230],  # Water
    [150, 100, 100],  # Bridge
    [250, 170, 30],   # Container
    [102, 102, 156],  # Traffic_Barrier
    [107, 142, 35],   # Vegetation
    [0, 0, 142],       # Vehicle
], dtype=np.uint8)


def remap_label(raw_label: np.ndarray) -> np.ndarray:
    """Vectorized raw-id -> merged-id remap for a uint8 label array."""
    return REMAP_LUT[raw_label]


def colorize(label: np.ndarray) -> np.ndarray:
    """Merged-id label array -> RGB visualization image."""
    return NEW_PALETTE[label]
