"""Абсолютная калибровка «трекинг -> пиксели» по маске кожи.

Параметры: f (fx=fy, квадратные пиксели), поворот камеры относительно СК головы
(rx, ry, rz), смещение левой камеры относительно точки головы (tx, ty, tz),
сдвиг времени lag. База стерео (64 мм по X головы) берётся из extrinsics файла.
Целевая функция: среднее (усечённое) расстояние от проекций суставов рук до
ближайшего пикселя кожи (distance transform маски YCrCb) в обоих глазах.
Результат сверяется с независимой оценкой фокуса из estimate_focal.py.
"""
import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from common import EYE_W, XR_TO_CV, eye_intrinsics, head_interpolator, load

N_FRAMES = 300
DS = 2  # даунскейл для distance transform
CLIP = 60.0  # px (в полном разрешении)

d = load()
t = (d["ts_ns"] - d["header_ts_ns"]) / 1e9
head_at = head_interpolator(t, d["head"])
K0 = eye_intrinsics(d["intrinsics"])
BASE = np.linalg.norm(d["extr_left"][:3, 3] - d["extr_right"][:3, 3])
LAG0 = float(np.load("data/sync.npz")["lag"])

cap = cv2.VideoCapture("sources/CameraRecord_20260505_165740.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
rng = np.random.default_rng(1)
frames = np.sort(rng.choice(np.arange(30, n - 30), N_FRAMES, replace=False))


def skin_dt(img):
    y = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(y, (40, 138, 80), (240, 175, 125))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.resize(m, None, fx=1 / DS, fy=1 / DS, interpolation=cv2.INTER_NEAREST)
    dt = cv2.distanceTransform(255 - m, cv2.DIST_L2, 3) * DS
    return np.minimum(dt, CLIP).astype(np.float32)


DT = []
for fi in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
    ok, f = cap.read()
    DT.append((skin_dt(f[:, :EYE_W]), skin_dt(f[:, EYE_W:])))
H_, W_ = DT[0][0].shape

# суставы рук (без ладони-центра и запястья: запястье часто под браслетом)
JOINTS = np.arange(2, 26)
tv0 = frames / fps


def gather(lag):
    """Мировые точки рук и поза головы на моменты кадров."""
    tt = tv0 + lag
    hp, hr = head_at(tt)
    idx = np.clip(np.searchsorted(t, tt), 0, len(t) - 1)
    pts, act = d["hand"][idx][:, :, JOINTS, :3], d["hand_active"][idx]
    # точки в СК головы: [F, 2 руки, J, 3]
    rel = np.einsum("fij,fhkj->fhki", hr.inv().as_matrix(), pts - hp[:, None, None, :])
    return rel, act


def cost(x, verbose=False):
    f, rx, ry, rz, tx, ty, tz, lag = x
    rel, act = gather(lag)
    Rc = R.from_rotvec([rx, ry, rz]).as_matrix() @ XR_TO_CV
    total, cnt = 0.0, 0
    for e, sx in ((0, 0.0), (1, 1.0)):
        off = np.array([tx + sx * BASE, ty, tz])
        pc = (rel - off) @ Rc.T  # [F,2,J,3]
        z = pc[..., 2]
        u = f * pc[..., 0] / z + K0[0, 2]
        v = f * pc[..., 1] / z + K0[1, 2]
        ok = (z > 0.05) & (u > 0) & (u < EYE_W - 1) & (v > 0) & (v < 809) & (act[:, :, None] == 1)
        fi, hi, ji = np.nonzero(ok)
        uu = (u[ok] / DS).astype(int).clip(0, W_ - 1)
        vv = (v[ok] / DS).astype(int).clip(0, H_ - 1)
        dts = np.array([DT[a][e][b, c] for a, b, c in zip(fi, vv, uu)])
        total += dts.sum()
        cnt += len(dts)
    # штраф за точки, вышедшие за кадр (иначе выгодно «вытолкнуть» руки из кадра)
    n_all = 2 * act.sum() * len(JOINTS)
    total += (n_all - cnt) * CLIP * 0.5
    return total / n_all


if __name__ == "__main__":
    starts = [
        [1.5 * K0[0, 0], 0, 0, 0, -BASE / 2, 0, 0, LAG0],
        [K0[0, 0], 0, 0, 0, -BASE / 2, 0, 0, LAG0],
        [1.3 * K0[0, 0], -0.2, 0, 0, -BASE / 2, -0.02, -0.05, LAG0],
    ]
    best = None
    for s in starts:
        print("start", np.round(s, 3), "cost", round(cost(s), 2))
        r = minimize(cost, s, method="Powell",
                     bounds=[(500, 1400), (-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6),
                             (-0.15, 0.1), (-0.15, 0.15), (-0.15, 0.15), (-0.4, 0.2)],
                     options=dict(maxiter=4000, xtol=1e-3, ftol=1e-4))
        print("  ->", np.round(r.x, 4), "cost", round(r.fun, 2))
        if best is None or r.fun < best.fun:
            best = r
    x = best.x
    print("BEST f=%.1f rot(deg)=%s off(m)=%s lag=%.3f cost=%.2f" % (
        x[0], np.degrees(x[1:4]).round(2), x[4:7].round(3), x[7], best.fun))
    print("FOV per eye: %.1f x %.1f deg" % (np.degrees(2 * np.arctan(540 / x[0])), np.degrees(2 * np.arctan(405 / x[0]))))
    np.savez("data/calib_skin.npz", x=x, base=BASE, cost=best.fun)
