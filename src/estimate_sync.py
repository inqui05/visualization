"""Оценка временного сдвига видео относительно трекинга.

Идея: при вращении головы всё изображение смещается. Для поворота камеры с
угловой скоростью w (в СК камеры) точка в центре кадра движется со скоростью
  du/dt = -fx * w_y,  dv/dt = fy * w_x.
Сдвиг между соседними кадрами меряем фазовой корреляцией (cv2.phaseCorrelate),
угловую скорость берём из кватернионов головы. Максимум взаимной корреляции
по лагу даёт сдвиг времени, а знак/амплитуда подтверждают ориентацию осей.
"""
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from common import XR_TO_CV, eye_intrinsics, load

SCALE = 4


def video_flow(path):
    cap = cv2.VideoCapture(path)
    prev, flows = None, []
    win = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        eye = frame[:, : frame.shape[1] // 2]
        g = cv2.cvtColor(cv2.resize(eye, None, fx=1 / SCALE, fy=1 / SCALE, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
        if win is None:
            win = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
        if prev is None:
            flows.append((0.0, 0.0, 0.0))
        else:
            (dx, dy), resp = cv2.phaseCorrelate(prev, g, win)
            flows.append((dx * SCALE, dy * SCALE, resp))
        prev = g
    fps = cap.get(cv2.CAP_PROP_FPS)
    return np.array(flows), fps


def main(video, npz):
    d = load(npz)
    K = eye_intrinsics(d["intrinsics"])
    t = (d["ts_ns"] - d["header_ts_ns"]) / 1e9
    rot = R.from_quat(d["head"][:, 3:])
    # угловая скорость в СК головы
    dr = (rot[:-1].inv() * rot[1:]).as_rotvec() / np.diff(t)[:, None]
    tm = (t[:-1] + t[1:]) / 2
    w_cam = dr @ XR_TO_CV.T
    pred_u = -K[0, 0] * w_cam[:, 1]
    pred_v = K[1, 1] * w_cam[:, 0]

    flows, fps = video_flow(video)
    np.save("data/video_flow.npy", flows)
    n = len(flows)
    tv = np.arange(n) / fps  # время кадра от начала записи
    meas_u = flows[:, 0] * fps  # px/s
    meas_v = flows[:, 1] * fps
    ok = flows[:, 2] > 0.1

    best = None
    lags = np.arange(-0.5, 0.5, 0.002)
    scores = []
    for lag in lags:
        pu = np.interp(tv + lag, tm, pred_u)
        pv = np.interp(tv + lag, tm, pred_v)
        c = np.corrcoef(np.r_[pu[ok], pv[ok]], np.r_[meas_u[ok], meas_v[ok]])[0, 1]
        scores.append(c)
    scores = np.array(scores)
    i = scores.argmax()
    lag = lags[i]
    pu = np.interp(tv + lag, tm, pred_u)
    pv = np.interp(tv + lag, tm, pred_v)
    gain_u = np.dot(pu[ok], meas_u[ok]) / np.dot(pu[ok], pu[ok])
    gain_v = np.dot(pv[ok], meas_v[ok]) / np.dot(pv[ok], pv[ok])
    print(f"video fps={fps:.3f}, frames={n}")
    print(f"best lag = {lag * 1000:.0f} ms (frame t -> tracking t + lag), corr={scores[i]:.3f}")
    print(f"corr at lag 0: {scores[np.argmin(np.abs(lags))]:.3f}")
    print(f"gain u={gain_u:.3f}, v={gain_v:.3f} (1.0 = intrinsics & axes consistent)")
    np.savez("data/sync.npz", lag=lag, lags=lags, scores=scores, gain_u=gain_u, gain_v=gain_v,
             tv=tv, meas_u=meas_u, meas_v=meas_v, pred_u=pu, pred_v=pv)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
