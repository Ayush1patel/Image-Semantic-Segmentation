"""Dataset roots and the fixed train/val/test sequence split."""

LABEL_ROOT = r"D:\Saqib\Multimodal_segmentation\UAVScenes\interval5_CAM_label"
IMAGE_ROOT = r"D:\Saqib\Multimodal_segmentation\UAVScenes\interval5_CAM_LIDAR"

# Sub-path (relative to LABEL_ROOT/<seq>) holding the raw class-id PNGs.
LABEL_ID_SUBDIR = "interval5_CAM_label_id"
# Sub-path (relative to IMAGE_ROOT/<seq>) holding the RGB camera JPEGs.
IMAGE_SUBDIR = "interval5_CAM"

SPLITS = {
    "train": [
        "interval5_AMtown01", "interval5_AMvalley01",
        "interval5_HKairport01", "interval5_HKisland01",
        "interval5_HKairport_GNSS01", "interval5_HKisland_GNSS01",
        "interval5_HKairport_GNSS02", "interval5_HKisland_GNSS02",
    ],
    "val": [
        "interval5_AMtown02", "interval5_AMvalley02",
        "interval5_HKairport02", "interval5_HKisland02",
    ],
    "test": [
        "interval5_AMtown03", "interval5_AMvalley03",
        "interval5_HKairport03", "interval5_HKisland03",
    ],
}
