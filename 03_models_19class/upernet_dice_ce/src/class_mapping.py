"""
Class remapping for the UPerNet+ResNet34 UAVScenes run, 19-class / 256x256
variant. Identical taxonomy to the sibling `unet_dice_ce`
project (vanilla UNet decoder) - only the decoder architecture differs
between the two projects (see src/model.py); this file is unchanged.

Unlike the sibling `unet_9class_baseline` project (which
merges the 19 named UAVScenes classes down into 9 semantic super-classes:
Building, Road, Water, Vegetation, Bridge, Container, Traffic_Barrier,
Vehicle, Background), this project keeps every one of the 19 classes named
in the UAVScenes paper as its own separate trainable class. The ONLY
merging that happens is folding the 7 unnamed/reserved raw ids into
Background - nothing else is grouped.

Raw labels use the official 26-slot UAVScenes id space (see cmap.py in the
dataset root), of which the paper actually names 19:
  - 1 background class
  - 16 static classes
  - 2 dynamic classes (sedan, truck)
The remaining 7 ids (7, 8, 12, 21, 22, 23, 25) are unused/reserved slots
that reuse leftover Cityscapes palette colors and never appear as
meaningful content in this dataset variant - all 7 are merged into
Background (raw id 0), same as every other project in this workspace.

  New ID  Class              Raw UAVScenes id(s) merged into it
  0       Background         0 (background), 7, 8, 12, 21, 22, 23, 25 (unnamed)
  1       Building           1 (roof)
  2       Dirt_Motor_Road    2
  3       Paved_Motor_Road   3
  4       River              4
  5       Pool               5
  6       Bridge             6
  7       Container          9
  8       Airstrip           10
  9       Traffic_Barrier    11
  10      Green_Field        13
  11      Wild_Field         14
  12      Solar_Board        15
  13      Umbrella           16
  14      Transparent_Roof   17
  15      Car_Park           18
  16      Paved_Walk         19
  17      Sedan              20
  18      Truck              24

Background is trained as a real class (not an ignore bucket) - same
convention as `unet_9class_baseline` and `unet_9class_background_lidar_iou`:
IGNORE_INDEX=None so every pixel is scored, but Background is excluded from
the headline mIoU (see BACKGROUND_INDEX + exclude_indices in train.py /
evaluate.py) since it dominates pixel count and would otherwise swamp the
metric. Checkpoints from this project are NOT compatible with the 9-class
`updated_mapping` project or the 10-class `unetpp` project - different
NUM_CLASSES, different class ids.
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
    "Background",        # 0
    "Building",           # 1
    "Dirt_Motor_Road",    # 2
    "Paved_Motor_Road",   # 3
    "River",              # 4
    "Pool",               # 5
    "Bridge",             # 6
    "Container",          # 7
    "Airstrip",           # 8
    "Traffic_Barrier",    # 9
    "Green_Field",        # 10
    "Wild_Field",         # 11
    "Solar_Board",        # 12
    "Umbrella",           # 13
    "Transparent_Roof",   # 14
    "Car_Park",           # 15
    "Paved_Walk",         # 16
    "Sedan",              # 17
    "Truck",              # 18
]
NUM_CLASSES = len(NEW_CLASS_NAMES)  # 19

BACKGROUND_INDEX = 0

# Every raw id now maps to a real class, so there is no ignore bucket.
# Kept as a module-level name (rather than deleted) because the loss,
# metrics and eval code all take an ignore_index argument; None means
# "score every pixel".
IGNORE_INDEX = None

# Raw id -> new (trainable) id. Only the 7 unnamed/reserved ids collapse
# into Background (0); every other named class keeps its own id.
RAW_TO_NEW = {
    0: 0,   # background -> Background
    1: 1,   # roof -> Building
    2: 2,   # dirt_motor_road -> Dirt_Motor_Road
    3: 3,   # paved_motor_road -> Paved_Motor_Road
    4: 4,   # river -> River
    5: 5,   # pool -> Pool
    6: 6,   # bridge -> Bridge
    7: 0,   # unused -> Background
    8: 0,   # unused -> Background
    9: 7,   # container -> Container
    10: 8,  # airstrip -> Airstrip
    11: 9,  # traffic_barrier -> Traffic_Barrier
    12: 0,  # unused -> Background
    13: 10,  # green_field -> Green_Field
    14: 11,  # wild_field -> Wild_Field
    15: 12,  # solar_board -> Solar_Board
    16: 13,  # umbrella -> Umbrella
    17: 14,  # transparent_roof -> Transparent_Roof
    18: 15,  # car_park -> Car_Park
    19: 16,  # paved_walk -> Paved_Walk
    20: 17,  # sedan -> Sedan
    21: 0,  # unused -> Background
    22: 0,  # unused -> Background
    23: 0,  # unused -> Background
    24: 18,  # truck -> Truck
    25: 0,  # unused -> Background
}

# LUT for O(1) vectorized remap: new = REMAP_LUT[raw]. Any raw id not
# explicitly listed above (out-of-range bytes) also falls back to Background.
REMAP_LUT = np.full(256, BACKGROUND_INDEX, dtype=np.uint8)
for _raw_id, _new_id in RAW_TO_NEW.items():
    REMAP_LUT[_raw_id] = _new_id

# Visualization palette (RGB) for the 19 trainable classes.
NEW_PALETTE = np.array([
    [0, 0, 0],          # Background (black)
    [119, 11, 32],       # Building
    [128, 64, 128],      # Dirt_Motor_Road
    [244, 35, 232],      # Paved_Motor_Road
    [70, 130, 180],      # River
    [173, 216, 230],     # Pool
    [150, 100, 100],     # Bridge
    [250, 170, 30],      # Container
    [81, 0, 81],         # Airstrip
    [102, 102, 156],     # Traffic_Barrier
    [107, 142, 35],      # Green_Field
    [154, 205, 50],      # Wild_Field
    [255, 215, 0],       # Solar_Board
    [220, 20, 60],       # Umbrella
    [188, 143, 143],     # Transparent_Roof
    [190, 153, 153],     # Car_Park
    [152, 251, 152],     # Paved_Walk
    [0, 0, 142],         # Sedan
    [0, 0, 70],          # Truck
], dtype=np.uint8)


def remap_label(raw_label: np.ndarray) -> np.ndarray:
    """Vectorized raw-id -> {0..18} remap for a uint8 label array."""
    return REMAP_LUT[raw_label]


def colorize(label: np.ndarray) -> np.ndarray:
    """Label array (values 0..18) -> RGB visualization image."""
    safe = np.clip(label, 0, NUM_CLASSES - 1)
    return NEW_PALETTE[safe]
