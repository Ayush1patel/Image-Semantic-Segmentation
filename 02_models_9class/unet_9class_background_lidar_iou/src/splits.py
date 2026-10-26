"""Dataset roots and the fixed train/val/test sequence split."""

LABEL_ROOT = r"D:\Saqib\Multimodal_segmentation\UAVScenes\interval5_CAM_label"
IMAGE_ROOT = r"D:\Saqib\Multimodal_segmentation\UAVScenes\interval5_CAM_LIDAR"
LIDAR_LABEL_ROOT = r"D:\Saqib\Multimodal_segmentation\UAVScenes\interval5_LIDAR_label"

# Sub-path (relative to LABEL_ROOT/<seq>) holding the raw class-id PNGs.
LABEL_ID_SUBDIR = "interval5_CAM_label_id"
# Sub-path (relative to IMAGE_ROOT/<seq>) holding the RGB camera JPEGs.
IMAGE_SUBDIR = "interval5_CAM"
# Sub-path (relative to IMAGE_ROOT/<seq>) holding the per-frame (N, 3) XYZ
# point clouds in the LiDAR frame.
LIDAR_SUBDIR = "interval5_LIDAR"
# Sub-path (relative to LIDAR_LABEL_ROOT/<seq>) holding the per-point raw
# class ids, one (N,) array per frame, aligned with LIDAR_SUBDIR.
LIDAR_LABEL_ID_SUBDIR = "interval5_LIDAR_label_id"

# LiDAR files carry both timestamps: image<cam_ts>_lidar<lidar_ts>.npy,
# so they are matched back to a camera frame by the <cam_ts> stem.
LIDAR_FILE_PREFIX = "image"
LIDAR_FILE_SEP = "_lidar"

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
