"""Общие функции: загрузка трекинга, модель камеры, интерполяция поз."""
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

VIDEO_W, VIDEO_H = 2160, 810  # side-by-side стерео
EYE_W = VIDEO_W // 2

# Названия суставов (PICO / OpenXR hand joints, 26 шт.)
HAND_JOINTS = [
    "Palm", "Wrist",
    "Thumb_metacarpal", "Thumb_proximal", "Thumb_distal", "Thumb_tip",
    "Index_metacarpal", "Index_proximal", "Index_intermediate", "Index_distal", "Index_tip",
    "Middle_metacarpal", "Middle_proximal", "Middle_intermediate", "Middle_distal", "Middle_tip",
    "Ring_metacarpal", "Ring_proximal", "Ring_intermediate", "Ring_distal", "Ring_tip",
    "Little_metacarpal", "Little_proximal", "Little_intermediate", "Little_distal", "Little_tip",
]
HAND_BONES = [(1, 0)] + [
    (a, b)
    for chain in ([1, 2, 3, 4, 5], [1, 6, 7, 8, 9, 10], [1, 11, 12, 13, 14, 15],
                  [1, 16, 17, 18, 19, 20], [1, 21, 22, 23, 24, 25])
    for a, b in zip(chain[:-1], chain[1:])
] + [(7, 12), (12, 17), (17, 22)]  # поперечные связи ладони

BODY_JOINTS = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle",
    "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder",
    "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]
BODY_BONES = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10),
              (8, 11), (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18),
              (17, 19), (18, 20), (19, 21), (20, 22), (21, 23)]

# OpenXR (x right, y up, z back)  ->  OpenCV camera (x right, y down, z forward)
XR_TO_CV = np.diag([1.0, -1.0, -1.0])


def load(path="data/tracking.npz"):
    return dict(np.load(path))


def eye_intrinsics(intr):
    """intrinsics в файле = [cx, cy, fx, fy] для кадра 2160x810 целиком
    (считаются по FOV 76.35x61.05 deg). Фактически каждый глаз — 1080x810,
    поэтому по горизонтали делим на 2: получаем fx ~= fy (квадратные пиксели)."""
    cx, cy, fx, fy = intr
    return np.array([[fx / 2, 0, (cx + 0.5) / 2 - 0.5], [0, fy, cy], [0, 0, 1]])


def head_interpolator(ts_s, head):
    """Возвращает f(t) -> (pos[n,3], Rotation) с линейной/сферической интерполяцией."""
    rot = R.from_quat(head[:, 3:])
    slerp = Slerp(ts_s, rot)

    def f(t):
        t = np.clip(t, ts_s[0], ts_s[-1])
        pos = np.stack([np.interp(t, ts_s, head[:, k]) for k in range(3)], -1)
        return pos, slerp(t)

    return f


def nearest_idx(ts_s, t):
    i = np.searchsorted(ts_s, t)
    i = np.clip(i, 1, len(ts_s) - 1)
    return np.where(np.abs(ts_s[i - 1] - t) < np.abs(ts_s[i] - t), i - 1, i)
