"""Парсинг trackingData_*.txt (XRoboToolkit Unity Client) в numpy-архив.

Особенность файла: числа записаны с десятичной ЗАПЯТОЙ (локаль устройства),
а разделитель значений — тоже запятая. Поэтому "0,1,-0,2" = [0.1, -0.2].
Значения без дробной части (например "0") встречаются, поэтому разбор
делается перебором с известным числом значений (7 для позы, 6 для скоростей).
"""
import json
import re
import sys
from functools import lru_cache

import numpy as np

FRAC_RE = re.compile(r"^\d+(E[+-]?\d+)?$")  # дробная часть: только цифры (+ экспонента)
INT_RE = re.compile(r"^-?\d+$")


def parse_floats(s: str, n: int) -> list[float]:
    tok = s.split(",")
    if len(tok) == 2 * n:
        return [float(f"{tok[2 * i]}.{tok[2 * i + 1]}") for i in range(n)]

    @lru_cache(None)
    def go(i: int, k: int):
        if i == len(tok):
            return () if k == n else None
        if k == n:
            return None
        # сначала пробуем "целая,дробная"
        if INT_RE.match(tok[i]) and i + 1 < len(tok) and FRAC_RE.match(tok[i + 1]):
            r = go(i + 2, k + 1)
            if r is not None:
                return (float(f"{tok[i]}.{tok[i + 1]}"),) + r
        try:
            v = float(tok[i])
        except ValueError:
            return None
        r = go(i + 1, k + 1)
        return None if r is None else (v,) + r

    res = go(0, 0)
    if res is None:
        raise ValueError(f"cannot parse {n} floats from {s!r}")
    return list(res)


def main(src: str, dst: str):
    with open(src, encoding="utf-8") as f:
        header = json.loads(f.readline())
        rows = [json.loads(l) for l in f if l.strip()]

    N = len(rows)
    out = {
        "ts_ns": np.array([r["timeStampNs"] for r in rows], dtype=np.int64),
        "predict_us": np.array([r.get("predictTime", np.nan) for r in rows], dtype=np.float64),
        "focus": np.array([r.get("appState", {}).get("focus", False) for r in rows]),
        "input": np.array([r.get("Input", -1) for r in rows], dtype=np.int8),
        "head": np.full((N, 7), np.nan),
        "head_status": np.zeros(N, np.int8),
        "hand": np.full((N, 2, 26, 7), np.nan),       # [кадр, лев/прав, сустав, xyz+qxyzw]
        "hand_active": np.zeros((N, 2), np.int8),
        "hand_scale": np.full((N, 2), np.nan),
        "hand_status": np.zeros((N, 2, 26), np.int16),
        "hand_radius": np.full((N, 2, 26), np.nan),
        "body": np.full((N, 24, 7), np.nan),
        "ctrl": np.full((N, 2, 7), np.nan),
        "ctrl_trigger": np.full((N, 2), np.nan),
        "ctrl_grip": np.full((N, 2), np.nan),
    }
    keys_seen = {}
    for i, r in enumerate(rows):
        for k in r:
            keys_seen[k] = keys_seen.get(k, 0) + 1
        if "Head" in r:
            out["head"][i] = parse_floats(r["Head"]["pose"], 7)
            out["head_status"][i] = r["Head"].get("status", 0)
        if "Hand" in r:
            for h, name in enumerate(("leftHand", "rightHand")):
                hd = r["Hand"].get(name)
                if not hd:
                    continue
                out["hand_active"][i, h] = hd["isActive"]
                out["hand_scale"][i, h] = hd["scale"]
                for j, jl in enumerate(hd["HandJointLocations"][:26]):
                    out["hand"][i, h, j] = parse_floats(jl["p"], 7)
                    out["hand_status"][i, h, j] = int(jl["s"])
                    out["hand_radius"][i, h, j] = jl["r"]
        if "Body" in r and "joints" in r["Body"]:
            for j, jt in enumerate(r["Body"]["joints"][:24]):
                out["body"][i, j] = parse_floats(jt["p"], 7)
        if "Controller" in r:
            for h, name in enumerate(("left", "right")):
                c = r["Controller"].get(name)
                if c:
                    out["ctrl"][i, h] = parse_floats(c["pose"], 7)
                    out["ctrl_trigger"][i, h] = c.get("trigger", np.nan)
                    out["ctrl_grip"][i, h] = c.get("grip", np.nan)

    # заголовок пишется Java-кодом — там обычные точки, это JSON-массивы в строке
    ext = [np.array(json.loads(e)).reshape(4, 4) for e in header["cameraExtrinsics"].split("|")]
    intr = json.loads(header["cameraIntrinsics"])
    out["header_ts_ns"] = np.int64(header["timeStampNs"])
    out["extr_left"], out["extr_right"] = ext
    out["intrinsics"] = np.array(intr)
    np.savez_compressed(dst, **out)

    print("rows:", N, "keys:", keys_seen)
    print("intrinsics:", intr)
    print("extrinsics L:\n", ext[0], "\nR:\n", ext[1])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
