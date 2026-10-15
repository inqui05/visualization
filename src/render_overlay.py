"""Рендер видео с наложенным трекингом.

Компоновка кадра 2160 x (810 + 360):
  верх  — исходное стерео-видео; в каждом глазу проекция скелетов рук (26 суставов),
          тела (24 сустава, тонко), следы кончиков указательных пальцев за 0.7 с;
  низ   — карта сверху (траектория головы и кистей), графики скоростей кистей и
          угловой скорости головы в окне ±6 с, текстовая телеметрия.

Запуск: python src/render_overlay.py [out.mp4] [start_s] [dur_s] [scale]
"""
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np
from scipy.spatial.transform import Rotation as R

from common import BODY_BONES, EYE_W, HAND_BONES, VIDEO_H, XR_TO_CV, head_interpolator, load, eye_intrinsics

SRC = "sources/CameraRecord_20260505_165740.mp4"
PANEL_H = 360
COL = {0: (255, 190, 40), 1: (40, 150, 255)}  # BGR: левая — голубая, правая — оранжевая
BODY_COL = (235, 235, 235)
TRAIL_S = 0.7
WIN_S = 6.0
# скелет тела — оценка PICO без внешних трекеров, с видео согласуется плохо; по умолчанию выключен
DRAW_BODY = False

d = load()
cal = np.load("data/calib_skin.npz")
F, RX, RY, RZ, TX, TY, TZ, LAG = cal["x"]
BASE = float(cal["base"])
K = eye_intrinsics(d["intrinsics"])
CX, CY = K[0, 2], K[1, 2]
RC = R.from_rotvec([RX, RY, RZ]).as_matrix() @ XR_TO_CV
CAM_OFF = [np.array([TX, TY, TZ]), np.array([TX + BASE, TY, TZ])]

t = (d["ts_ns"] - d["header_ts_ns"]) / 1e9
head_at = head_interpolator(t, d["head"])
hand = d["hand"][..., :3]
hand_ok = (d["hand_active"][:, :, None] == 1) & (d["hand_status"] > 0)
body = d["body"][..., :3]

# --- ряды для графиков (на частоте трекинга) ---
dt = np.gradient(t)
wrist_speed = np.linalg.norm(np.gradient(hand[:, :, 1], axis=0), axis=-1) / dt[:, None]
wrist_speed[~hand_ok[:, :, 1]] = np.nan
wrist_speed = np.stack([np.convolve(np.nan_to_num(w), np.ones(9) / 9, "same") for w in wrist_speed.T], 1)
rot = R.from_quat(d["head"][:, 3:])
ang_speed = np.r_[0, np.degrees((rot[:-1].inv() * rot[1:]).magnitude()) / np.diff(t)]
ang_speed = np.convolve(ang_speed, np.ones(9) / 9, "same")
eul = rot.as_euler("YXZ", degrees=True)  # yaw, pitch, roll (Y вверх)

# пинч: расстояние большой–указательный
pinch = np.linalg.norm(hand[:, :, 5] - hand[:, :, 10], axis=-1)


def interp_frame(arr, tt):
    """Линейная интерполяция массива трекинга на момент tt."""
    i = int(np.clip(np.searchsorted(t, tt), 1, len(t) - 1))
    w = np.clip((tt - t[i - 1]) / (t[i] - t[i - 1]), 0, 1)
    return arr[i - 1] * (1 - w) + arr[i] * w, i if w > 0.5 else i - 1


def project(pw, hp, hr, eye):
    """мир -> пиксели глаза (u, v, z)."""
    ph = hr.inv().apply(pw.reshape(-1, 3) - hp) - CAM_OFF[eye]
    pc = ph @ RC.T
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = F * pc[:, 0] / z + CX
        v = F * pc[:, 1] / z + CY
    return np.stack([u, v, z], -1).reshape(pw.shape[:-1] + (3,))


def draw_skeleton(img, uvz, bones, ok, col, thick, dot):
    vis = ok & (uvz[:, 2] > 0.05)
    for a, b in bones:
        if vis[a] and vis[b]:
            cv2.line(img, (int(uvz[a, 0] * 4), int(uvz[a, 1] * 4)), (int(uvz[b, 0] * 4), int(uvz[b, 1] * 4)),
                     col, thick, cv2.LINE_AA, shift=2)
    if dot:
        for j in np.nonzero(vis)[0]:
            cv2.circle(img, (int(uvz[j, 0] * 4), int(uvz[j, 1] * 4)), dot * 4, col, -1, cv2.LINE_AA, shift=2)


# --- статичная подложка карты сверху ---
MAP = 340
hm = np.nanpercentile(np.r_[d["head"][:, [0, 2]], hand[:, :, 1, :][hand_ok[:, :, 1]][:, [0, 2]]], [1, 99], axis=0)
mc = hm.mean(0)
mspan = max((hm[1] - hm[0]).max() * 1.25, 0.6)


def to_map(xz):
    p = (xz - mc) / mspan * (MAP - 30) + MAP / 2
    return p


def draw_panel(panel, tt, i):
    panel[:] = (24, 22, 20)
    # ---- карта сверху ----
    m = panel[10:10 + MAP, 10:10 + MAP]
    m[:] = (34, 32, 30)
    for g in np.arange(-2, 2.01, 0.1):  # сетка 10 см
        for k in (0, 1):
            a = np.array([[g, -2.0], [g, 2.0]]) if k == 0 else np.array([[-2.0, g], [2.0, g]])
            p = to_map(a + np.round(mc, 1))
            cv2.line(m, tuple(p[0].astype(int)), tuple(p[1].astype(int)), (48, 46, 44), 1)
    path = to_map(d["head"][:, [0, 2]]).astype(np.int32)
    cv2.polylines(m, [path], False, (70, 70, 70), 1, cv2.LINE_AA)
    j0 = np.searchsorted(t, tt - 5)
    cv2.polylines(m, [path[j0:i + 1]], False, (200, 200, 200), 2, cv2.LINE_AA)
    for h in (0, 1):
        sel = np.arange(j0, i + 1)
        sel = sel[hand_ok[sel, h, 1]]
        if len(sel) > 1:
            cv2.polylines(m, [to_map(hand[sel, h, 1][:, [0, 2]]).astype(np.int32)], False, COL[h], 1, cv2.LINE_AA)
        if hand_ok[i, h, 1]:
            cv2.circle(m, tuple(to_map(hand[i, h, 1, [0, 2]]).astype(int)), 5, COL[h], -1, cv2.LINE_AA)
    hp = d["head"][i, :3]
    fw = rot[i].apply([0, 0, -1])
    p0 = to_map(hp[[0, 2]])
    p1 = to_map(hp[[0, 2]] + fw[[0, 2]] / (np.linalg.norm(fw[[0, 2]]) + 1e-9) * 0.12)
    cv2.circle(m, tuple(p0.astype(int)), 6, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.arrowedLine(m, tuple(p0.astype(int)), tuple(p1.astype(int)), (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)
    cv2.putText(m, "top view (X-Z), grid 10 cm", (8, MAP - 10), 0, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

    # ---- графики ----
    x0, w, gh = 380, 1180, 150
    charts = [("wrist speed, m/s", [(wrist_speed[:, 0], COL[0]), (wrist_speed[:, 1], COL[1])], 1.5),
              ("head angular speed, deg/s", [(ang_speed, (200, 200, 200))], 120)]
    a, b = np.searchsorted(t, tt - WIN_S), np.searchsorted(t, tt + WIN_S)
    xs = x0 + (t[a:b] - (tt - WIN_S)) / (2 * WIN_S) * w
    for k, (title, series, vmax) in enumerate(charts):
        y0 = 15 + k * (gh + 25)
        cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + gh), (34, 32, 30), -1)
        for q in (0.5, 1.0):
            yy = int(y0 + gh - q * gh * 0.9)
            cv2.line(panel, (x0, yy), (x0 + w, yy), (52, 50, 48), 1)
            cv2.putText(panel, f"{q * vmax:g}", (x0 + 4, yy - 3), 0, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
        for s, c in series:
            ys = y0 + gh - np.clip(s[a:b] / vmax, 0, 1.05) * gh * 0.9
            pts = np.stack([xs, ys], 1).astype(np.int32)
            cv2.polylines(panel, [pts], False, c, 2, cv2.LINE_AA)
        cv2.putText(panel, title, (x0 + 60, y0 + 18), 0, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
    cx_ = x0 + w // 2
    cv2.line(panel, (cx_, 10), (cx_, 15 + 2 * gh + 30), (255, 255, 255), 1)
    cv2.putText(panel, f"-{WIN_S:g}s", (x0, 15 + 2 * gh + 45), 0, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(panel, f"+{WIN_S:g}s", (x0 + w - 30, 15 + 2 * gh + 45), 0, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

    # ---- телеметрия ----
    tx = x0 + w + 25
    lines = [
        (f"video t = {tt - LAG:7.2f} s", (230, 230, 230)),
        (f"tracking t = {tt:7.2f} s  (lag {LAG * 1000:+.0f} ms)", (160, 160, 160)),
        (f"head pos  x{hp[0]:+.2f} y{hp[1]:+.2f} z{hp[2]:+.2f} m", (200, 200, 200)),
        (f"yaw {eul[i, 0]:+6.1f}  pitch {eul[i, 1]:+6.1f}  roll {eul[i, 2]:+5.1f}", (200, 200, 200)),
    ]
    for h, name in ((0, "LEFT "), (1, "RIGHT")):
        if hand_ok[i, h, 1]:
            pin = pinch[i, h] * 100
            state = "pinch" if pin < 2.5 else ""
            lines.append((f"{name} hand: {wrist_speed[i, h]:4.2f} m/s  thumb-index {pin:4.1f} cm {state}", COL[h]))
        else:
            lines.append((f"{name} hand: not tracked", (90, 90, 90)))
    for k, (s, c) in enumerate(lines):
        cv2.putText(panel, s, (tx, 35 + k * 30), 0, 0.55, c, 1, cv2.LINE_AA)
    cv2.putText(panel, "trail = index fingertip, last 0.7 s (world-fixed)", (tx, 35 + len(lines) * 30 + 10), 0, 0.45,
                (150, 150, 150), 1, cv2.LINE_AA)


def render_frame(fr, fi, fps, panel):
    tt = fi / fps + LAG
    hp, hr = head_at(np.array([tt]))
    hp, hr = hp[0], hr[0]
    hnd, i = interp_frame(hand, tt)
    bod, _ = interp_frame(body, tt)
    j0 = np.searchsorted(t, tt - TRAIL_S)
    for e in (0, 1):
        eye = np.ascontiguousarray(fr[:, e * EYE_W:(e + 1) * EYE_W])
        over = eye.copy()
        if DRAW_BODY:
            bz = project(bod, hp, hr, e)
            draw_skeleton(over, bz, BODY_BONES, np.isfinite(bz[:, 0]), BODY_COL, 2, 0)
            cv2.addWeighted(over, 0.45, eye, 0.55, 0, eye)
        for h in (0, 1):
            if not hand_ok[i, h, 1]:
                continue
            # след кончика указательного пальца (в мире) — проверка согласованности по времени/позе
            sel = np.arange(j0, i + 1)
            sel = sel[hand_ok[sel, h, 10]]
            if len(sel) > 2:
                tr = project(hand[sel, h, 10], hp, hr, e)
                tr = tr[tr[:, 2] > 0.05]
                cv2.polylines(eye, [(tr[:, :2] * 4).astype(np.int32)], False, COL[h], 2, cv2.LINE_AA, shift=2)
            uvz = project(hnd[h], hp, hr, e)
            draw_skeleton(eye, uvz, HAND_BONES, hand_ok[i, h], (20, 20, 20), 5, 0)  # тёмная обводка
            draw_skeleton(eye, uvz, HAND_BONES, hand_ok[i, h], COL[h], 2, 3)
        fr[:, e * EYE_W:(e + 1) * EYE_W] = eye
    cv2.putText(fr, "L cam", (14, 30), 0, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(fr, "R cam", (EYE_W + 14, 30), 0, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    draw_panel(panel, tt, i)


def main(out="output/overlay.mp4", start=0.0, dur=None, scale=1.0):
    cap = cv2.VideoCapture(SRC)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = int(start * fps)
    f1 = n if dur is None else min(n, f0 + int(dur * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    W, H = 2160, VIDEO_H + PANEL_H
    ow, oh = int(W * scale) // 2 * 2, int(H * scale) // 2 * 2
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{ow}x{oh}", "-r", f"{fps:.5f}", "-i", "-", "-c:v", "libx264", "-preset", "medium",
           "-crf", "23" if scale >= 1 else "28", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    canvas = np.zeros((H, W, 3), np.uint8)
    for fi in range(f0, f1):
        ok, fr = cap.read()
        if not ok:
            break
        render_frame(fr, fi, fps, canvas[VIDEO_H:])
        canvas[:VIDEO_H] = fr
        img = canvas if scale == 1 else cv2.resize(canvas, (ow, oh), interpolation=cv2.INTER_AREA)
        proc.stdin.write(img.tobytes())
        if (fi - f0) % 500 == 0:
            print(f"frame {fi}/{f1}", flush=True)
    proc.stdin.close()
    proc.wait()


def still(fi, path):
    cap = cv2.VideoCapture(SRC)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, fr = cap.read()
    canvas = np.zeros((VIDEO_H + PANEL_H, 2160, 3), np.uint8)
    render_frame(fr, fi, fps, canvas[VIDEO_H:])
    canvas[:VIDEO_H] = fr
    cv2.imwrite(path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    if sys.argv[1:2] == ["still"]:
        still(int(sys.argv[2]), sys.argv[3])
    else:
        a = sys.argv[1:]
        main(a[0] if a else "output/overlay.mp4", float(a[1]) if len(a) > 1 else 0.0,
             float(a[2]) if len(a) > 2 else None, float(a[3]) if len(a) > 3 else 1.0)
