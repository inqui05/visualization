"""Оценка фокусного расстояния (и поправки ориентации камеры) по вращению головы.

Для двух кадров при чистом повороте камеры точки связаны гомографией
H = K * R_rel * K^-1, где R_rel — относительный поворот камеры, известный
из трекинга головы (с учётом найденного временного сдвига). Сопоставляем
ORB-признаки между кадрами (разнесёнными на STEP кадров), берём пары с заметным
поворотом и подбираем fx, fy, минимизируя медианную ошибку переноса точек.
Параллакс от поступательного движения головы добавляет шум, поэтому
используется робастная функция потерь (soft L1) и медианная ошибка.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from common import XR_TO_CV, eye_intrinsics, head_interpolator, load

STEP = 4
d = load()
sync = np.load("data/sync.npz")
LAG = float(sync["lag"])
t = (d["ts_ns"] - d["header_ts_ns"]) / 1e9
head_at = head_interpolator(t, d["head"])
K0 = eye_intrinsics(d["intrinsics"])

cap = cv2.VideoCapture("sources/CameraRecord_20260505_165740.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# кадры, где за STEP кадров голова повернулась на 3..12 градусов
tv = np.arange(n) / fps + LAG
_, rots = head_at(tv)
ang = np.degrees((rots[:-STEP].inv() * rots[STEP:]).magnitude())
cand = np.where((ang > 3) & (ang < 12))[0]
rng = np.random.default_rng(0)
sel = np.sort(rng.choice(cand, size=min(150, len(cand)), replace=False))
print("candidate pairs:", len(cand), "used:", len(sel))

orb = cv2.ORB_create(1500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def grab(i):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, f = cap.read()
    return cv2.cvtColor(f[:, :1080], cv2.COLOR_BGR2GRAY)


P1, P2, RR = [], [], []
for i in sel:
    a, b = grab(i), grab(i + STEP)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None:
        continue
    m = bf.match(da, db)
    if len(m) < 30:
        continue
    pa = np.float32([ka[x.queryIdx].pt for x in m])
    pb = np.float32([kb[x.trainIdx].pt for x in m])
    # отсекаем выбросы фундаментальной матрицей (модельно-независимо)
    F, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 1.5)
    if mask is None:
        continue
    mask = mask.ravel() == 1
    # R камеры: R_cam_b<-a = C * R_head_b^-1 * R_head_a * C^T
    Rrel = XR_TO_CV @ (rots[i + STEP].inv() * rots[i]).as_matrix() @ XR_TO_CV.T
    P1.append(pa[mask]); P2.append(pb[mask]); RR.append(Rrel)
print("pairs with matches:", len(P1), "points:", sum(len(p) for p in P1))


def residuals(x):
    fx, fy, cx, cy, rx, ry, rz = x
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    Ki = np.linalg.inv(K)
    Rc = R.from_rotvec([rx, ry, rz]).as_matrix()  # поправка ориентации камеры относительно головы
    res = []
    for pa, pb, Rr in zip(P1, P2, RR):
        H = K @ Rc @ Rr @ Rc.T @ Ki
        q = cv2.perspectiveTransform(pa[None].astype(np.float64), H)[0]
        res.append(np.linalg.norm(q - pb, axis=1))
    return np.concatenate(res)


x0 = [K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2], 0, 0, 0]
print("median err @ SDK intrinsics:", np.median(residuals(x0)))
for s in (1.2, 1.4, 1.5, 1.6, 1.8):
    print(f"  scale {s}: median err", np.median(residuals([K0[0, 0] * s, K0[1, 1] * s, K0[0, 2], K0[1, 2], 0, 0, 0])))
sol = least_squares(residuals, [K0[0, 0] * 1.5, K0[1, 1] * 1.5, K0[0, 2], K0[1, 2], 0, 0, 0],
                    loss="soft_l1", f_scale=3.0, x_scale=[100, 100, 50, 50, .05, .05, .05])
print("fit:", np.round(sol.x, 4), "median err:", np.median(residuals(sol.x)))
fx, fy = sol.x[:2]
print(f"FOV per eye: {np.degrees(2 * np.arctan(540 / fx)):.1f} x {np.degrees(2 * np.arctan(405 / fy)):.1f} deg")
np.savez("data/calib.npz", x=sol.x, x_sdk=np.array(x0))
