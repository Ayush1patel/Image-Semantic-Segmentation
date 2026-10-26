"""
Project a frame's LiDAR point cloud into the camera image plane.

Used to answer "how well does the model do on the pixels the LiDAR actually
sees?" - we rasterize the projected points into a boolean mask at the eval
resolution and restrict the IoU computation to those pixels.

Geometry
--------
Each frame in interval5_CAM_LIDAR/<seq>/interval5_LIDAR/*.npy is an (N, 3)
float32 array of XYZ points in the LiDAR frame. The camera-LiDAR extrinsics
below (from the dataset root's calibration_results.py, which sources them
from MARS-LVIG) take a point into the camera frame as

    P_cam = R @ P_lidar + t

then the pinhole + Brown-Conrady distortion model gives the pixel. This
direction was verified empirically against the data: 99.1% of a frame's
points land inside the image, and the per-point LiDAR labels agree with the
CAM label PNG at the projected pixel 99.0% of the time on non-background
points (93.3% overall). The inverse transform puts <4% of points in frame
and destroys that agreement, so the convention above is the correct one.

Resolution
----------
Projected coordinates are scaled straight from the source image size to the
eval size, rather than rasterizing at full resolution and downsampling. At
2448x2048 a frame's ~24.5k points cover only 0.48% of pixels, so a nearest-
neighbour downsample to 512x512 would throw away ~95% of them; scaling the
coordinates keeps every point. Resulting coverage is ~8.3% of pixels at
512x512 (~49% at 128x128).
"""
import numpy as np

# --- Camera-LiDAR calibrations -------------------------------------------
# Verbatim copy of the tables in
#   D:\Saqib\Multimodal_segmentation\UAVScenes\calibration_results.py
# kept in-tree so the project is self-contained. camera_intrinsic is a
# row-major 3x3 K; dist coeffs are [k1, k2, p1, p2, k3]; camera_ext_R is a
# row-major 3x3 rotation; camera_ext_t is a translation in metres.

AMtown_calibration = {
    'camera_intrinsic': [1453.72, 0.0, 1172.18,
                         0.0, 1453.28, 1041.78,
                         0.0, 0.0, 1.0],
    'camera_dist_coeffs': [-0.121, 0.1113, 0.0016, 0.00013, -0.062353],
    'camera_ext_R': [0.00298088, -0.999728, -0.0231416,
                     -0.00504636, 0.0231263, -0.99972,
                     0.999983, 0.00309683, -0.00497605],
    'camera_ext_t': [0.0025563, 0.0567484, -0.0512149],
}

AMvalley_calibration = {
    'camera_intrinsic': [1453.88, 0.0, 1182.53,
                         0.0, 1452.85, 1045.82,
                         0.0, 0.0, 1.0],
    'camera_dist_coeffs': [-0.052, 0.1168, 0.0015, 0.00013, -0.068564],
    'camera_ext_R': [0.00298068, -0.999735, -0.0231428,
                     -0.00504595, 0.023132, -0.99974,
                     0.999985, 0.00309701, -0.00497598],
    'camera_ext_t': [-0.0025563, 0.0567484, -0.0512149],
}

HKairport_calibration = {
    'camera_intrinsic': [1451.28, 0.0, 1177.5,
                         0.0, 1451.29, 1043.5,
                         0.0, 0.0, 1.0],
    'camera_dist_coeffs': [-0.0572, 0.1209, 0.00124, -0.00018, -0.06327],
    'camera_ext_R': [0.00322743, -0.999736, -0.022768,
                     -0.0560389, 0.0227496, -0.999725,
                     0.999979, 0.00335414, -0.00552898],
    'camera_ext_t': [0.00242461, 0.0765454, -0.0313375],
}

HKisland_calibration = {
    'camera_intrinsic': [1444.43, 0.0, 1177.8,
                         0.0, 1444.34, 1043.6,
                         0.0, 0.0, 1.0],
    'camera_dist_coeffs': [-0.053, 0.121, 0.00127, 0.00043, -0.06495],
    'camera_ext_R': [0.00352762, -0.999765, -0.0213775,
                     -0.0111803, 0.0213369, -0.99971,
                     0.999931, 0.00376561, -0.0111025],
    'camera_ext_t': [-0.0025563, 0.0470454, -0.0513375],
}

HK_GNSS_calibration = {
    'camera_intrinsic': [1444.43, 0.0, 1179.50,
                         0.0, 1444.34, 1044.90,
                         0.0, 0.0, 1.0],
    'camera_dist_coeffs': [-0.0560, 0.1180, 0.00122, 0.00064, -0.0627],
    'camera_ext_R': [0.00363212, -0.999819, -0.0213618,
                     -0.0111679, 0.0214512, -0.999591,
                     0.999879, 0.00375613, -0.0111134],
    'camera_ext_t': [-0.0021928, 0.0470312, -0.0513126],
}

Featureless_GNSS_calibration = {
    'camera_intrinsic': [1450.09, 0, 1177.5,
                         0.0, 1450.09, 1044.5,
                         0.0, 0.0, 1.0],
    'camera_dist_coeffs': [-0.0558, 0.1247, 0.00126, 0.0007, -0.06762],
    'camera_ext_R': [0.00941312, -0.999586, -0.0271775,
                     -0.00737429, 0.0271086, -0.999605,
                     0.999929, 0.00960988, -0.00711613],
    'camera_ext_t': [-0.0025563, 0.0567484, -0.0512149],
}

# Note HKairport01/02/03 deliberately use HK_GNSS_calibration, not
# HKairport_calibration - that is what the dataset's own mapping says.
SCENE_TO_CALIBRATION = {
    'AMtown01': AMtown_calibration,
    'AMtown02': AMtown_calibration,
    'AMtown03': AMtown_calibration,
    'AMvalley01': AMvalley_calibration,
    'AMvalley02': AMvalley_calibration,
    'AMvalley03': AMvalley_calibration,
    'HKairport01': HK_GNSS_calibration,
    'HKairport02': HK_GNSS_calibration,
    'HKairport03': HK_GNSS_calibration,
    'HKisland01': HKisland_calibration,
    'HKisland02': HKisland_calibration,
    'HKisland03': HKisland_calibration,
    'HKairport_GNSS_Evening': HK_GNSS_calibration,
    'HKisland_GNSS_Evening': Featureless_GNSS_calibration,
    'HKairport_GNSS01': HK_GNSS_calibration,
    'HKairport_GNSS02': HK_GNSS_calibration,
    'HKairport_GNSS03': HK_GNSS_calibration,
    'HKisland_GNSS01': HK_GNSS_calibration,
    'HKisland_GNSS02': HK_GNSS_calibration,
    'HKisland_GNSS03': HK_GNSS_calibration,
    'Featureless_GNSS01': Featureless_GNSS_calibration,
    'Featureless_GNSS02': Featureless_GNSS_calibration,
    'Featureless_GNSS03': Featureless_GNSS_calibration,
}

SEQ_PREFIX = "interval5_"
MIN_DEPTH_M = 0.1  # points closer than this (or behind the camera) are dropped


def calibration_for_sequence(sequence: str) -> dict:
    """'interval5_AMtown03' -> that scene's calibration dict."""
    scene = sequence[len(SEQ_PREFIX):] if sequence.startswith(SEQ_PREFIX) else sequence
    try:
        return SCENE_TO_CALIBRATION[scene]
    except KeyError:
        raise KeyError(
            f"No camera-LiDAR calibration for sequence {sequence!r} (scene {scene!r}). "
            f"Known scenes: {sorted(SCENE_TO_CALIBRATION)}"
        ) from None


def _unpack(calib):
    K = np.asarray(calib["camera_intrinsic"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(calib["camera_dist_coeffs"], dtype=np.float64)
    R = np.asarray(calib["camera_ext_R"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(calib["camera_ext_t"], dtype=np.float64)
    return K, D, R, t


def project_to_pixels(points, calib, src_size, out_size=None):
    """LiDAR-frame points -> pixel coordinates in the (optionally resized) image.

    points   : (N, 3) float array, XYZ in the LiDAR frame.
    calib    : dict from calibration_for_sequence().
    src_size : (height, width) of the original camera image.
    out_size : (height, width) to scale pixel coordinates into. Defaults to
               src_size.

    Returns (cols, rows, depth, keep) where the first three are int/float
    arrays over the *kept* points only, and `keep` is the boolean mask into
    the original N points (in front of the camera and inside the frame), so
    per-point attributes can be selected with points_attr[keep].
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected an (N, 3) point array, got {points.shape}")
    points = points[:, :3]

    K, D, R, t = _unpack(calib)
    src_h, src_w = src_size
    out_h, out_w = out_size if out_size is not None else src_size

    cam = points @ R.T + t            # P_cam = R @ P_lidar + t
    depth = cam[:, 2]
    in_front = depth > MIN_DEPTH_M

    x = np.zeros(len(points))
    y = np.zeros(len(points))
    np.divide(cam[:, 0], depth, out=x, where=in_front)
    np.divide(cam[:, 1], depth, out=y, where=in_front)

    k1, k2, p1, p2, k3 = D
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    u = K[0, 0] * x_d + K[0, 1] * y_d + K[0, 2]
    v = K[1, 1] * y_d + K[1, 2]

    # Scale from the source image grid onto the eval grid.
    u *= out_w / src_w
    v *= out_h / src_h

    cols = np.floor(u)
    rows = np.floor(v)
    keep = (in_front
            & (cols >= 0) & (cols < out_w)
            & (rows >= 0) & (rows < out_h))

    return (cols[keep].astype(np.int64), rows[keep].astype(np.int64),
            depth[keep], keep)


def rasterize(points, point_labels, calib, src_size, out_size,
              no_data_index: int = 255):
    """Rasterize a frame's points into an (H, W) mask + label image.

    Returns (mask, lidar_gt):
      mask     bool (out_h, out_w) - True where at least one point projects.
      lidar_gt uint8 (out_h, out_w) - the projecting point's own label at
               those pixels, `no_data_index` elsewhere. When several points
               land on one pixel the nearest (smallest depth) wins, which is
               the physically correct occlusion order.

    point_labels may be None, in which case lidar_gt is returned as all
    `no_data_index` and only the mask is meaningful.
    """
    out_h, out_w = out_size
    cols, rows, depth, keep = project_to_pixels(points, calib, src_size, out_size)

    mask = np.zeros((out_h, out_w), dtype=bool)
    lidar_gt = np.full((out_h, out_w), no_data_index, dtype=np.uint8)
    if len(cols) == 0:
        return mask, lidar_gt

    mask[rows, cols] = True
    if point_labels is not None:
        labels = np.asarray(point_labels).reshape(-1)[keep]
        # Farthest first, so the nearest point is written last and wins.
        order = np.argsort(-depth, kind="stable")
        lidar_gt[rows[order], cols[order]] = labels[order].astype(np.uint8)
    return mask, lidar_gt
