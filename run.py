"""Полный конвейер: от исходных файлов до видео с наложением.

    python run.py            # всё видео -> output/overlay.mp4 (~13 мин)
    python run.py --quick    # 24-секундный фрагмент в половинном размере -> output/overlay_quick.mp4 (~4 мин)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

VIDEO = "sources/CameraRecord_20260505_165740.mp4"
TRACKING = "sources/trackingData_20260505_165740.txt"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="отрисовать только фрагмент 1:58–2:22 в половинном размере")
    args = ap.parse_args()

    missing = [p for p in (VIDEO, TRACKING) if not Path(p).is_file()]
    if missing:
        sys.exit("Не найдены исходные файлы: " + ", ".join(missing) + "\nПоложите их в папку sources/.")
    Path("data").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)

    render = ["output/overlay_quick.mp4", "118", "24", "0.5"] if args.quick else ["output/overlay.mp4"]
    steps = [
        ("Разбор трекинга", ["src/parse_tracking.py", TRACKING, "data/tracking.npz"]),
        ("Синхронизация видео и трекинга", ["src/estimate_sync.py", VIDEO, "data/tracking.npz"]),
        ("Калибровка камеры", ["src/calibrate_skin.py"]),
        ("Рендер видео", ["src/render_overlay.py", *render]),
    ]
    t0 = time.time()
    for i, (name, cmd) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {name}", flush=True)
        t = time.time()
        subprocess.run([sys.executable, *cmd], check=True)
        print(f"    готово за {time.time() - t:.0f} с", flush=True)
    print(f"\nВсё готово за {(time.time() - t0) / 60:.1f} мин: {render[0]}")


if __name__ == "__main__":
    main()
