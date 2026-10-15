
"""
SAFE-X | Standalone live PPE detection test bench (YOLO11n + OpenCV)

Purpose
-------
Compare RAW model output against an optionally STABILIZED display, side by side,
so you can see what is a model problem and what is a code/threshold problem.

This file does not touch any other SAFE-X file. No RFID, workers, or servers.

Keys (click the video window first)
-----------------------------------
  Q  quit
  M  cycle display mode: both -> raw -> stable
  T  toggle per-class thresholds <-> one global threshold
  [  ]  lower / raise the global threshold by 0.05 (only used when T is off)
  R  reset statistics and tracks
  P  print statistics table now

What stabilization does and does NOT do
---------------------------------------
It only smooths the DISPLAY. It cannot add detections the model never made,
cannot fix wrong classes, and cannot remove a false positive that is stable.
A class that is not drawn is NOT proof that the PPE is absent.
"""

import argparse
import threading
import time
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# SETTINGS - adjust these first
# ============================================================

MODEL_PATH = Path(
    r"C:\Users\Nirmalya Lenka\runs\detect\runs\detect"
    r"\safe_x_ppe_compliance_clean\weights\best.pt"
)

# ---- Camera ------------------------------------------------
CAMERA_INDEX = 0
CAMERA_BACKEND = "dshow"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_FOURCC = "MJPG"
CAMERA_WARMUP_FRAMES = 10
CAMERA_READ_TIMEOUT_S = 3.0
CAMERA_MAX_CONSECUTIVE_FAILS = 30

# ---- Inference ---------------------------------------------
IMGSZ = 640
DEVICE = None
USE_HALF = False
NMS_IOU = 0.70
MAX_DET = 100
AGNOSTIC_NMS = False

INFERENCE_CONF_FLOOR = 0.10

# ---- Confidence thresholds ---------------------------------
USE_PER_CLASS_THRESHOLDS = True
GLOBAL_CONF = 0.35
DEFAULT_CLASS_CONF = 0.35

CLASS_CONF = {
    "Person":     0.40,
    "Helmet":     0.35,
    "Vest":       0.30,
    "Mask":       0.25,
    "Goggles":    0.25,
    "Gloves":     0.25,
    "Boots":      0.25,
    "NO-Helmet":  0.40,
    "NO-Vest":    0.40,
    "NO-Mask":    0.40,
    "NO-Goggles": 0.40,
    "NO-Gloves":  0.40,
    "NO-Boots":   0.40,
}

# ---- Hysteresis --------------------------------------------
HYSTERESIS_RATIO = 0.70

# ---- Temporal stabilization --------------------------------
DISPLAY_MODE = "both"
CONFIRM_HITS = 2
MATCH_IOU = 0.30
MAX_COAST_SECONDS = 0.35
MAX_COAST_FRAMES = 6
BOX_SMOOTH_ALPHA = 0.60
CONF_SMOOTH_ALPHA = 0.50

CONFLICT_IOU = 0.50
RESOLVE_CONFLICTS = False

# ---- Display -----------------------------------------------
DISPLAY_SCALE = 1.0
FONT_SCALE = 0.50
BOX_THICKNESS = 2
WINDOW_NAME = "SAFE-X | PPE detection test bench"

EXPECTED_CLASSES = [
    "Boots", "Gloves", "Goggles", "Helmet", "Mask",
    "NO-Boots", "NO-Gloves", "NO-Goggles", "NO-Helmet",
    "NO-Mask", "NO-Vest", "Person", "Vest",
]

# ============================================================
# Data types and helpers
# ============================================================

Det = namedtuple("Det", "name conf box")
Item = namedtuple("Item", "name conf box coasting")

COLOR_PPE = (0, 170, 0)
COLOR_NO_PPE = (0, 0, 220)
COLOR_PERSON = (255, 128, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

CONFLICT_PAIRS = {}
for _base in ("Boots", "Gloves", "Goggles", "Helmet", "Mask", "Vest"):
    CONFLICT_PAIRS[_base] = f"NO-{_base}"
    CONFLICT_PAIRS[f"NO-{_base}"] = _base


def color_for(name):
    if name.upper().startswith("NO-"):
        return COLOR_NO_PPE
    if name == "Person":
        return COLOR_PERSON
    return COLOR_PPE


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (
        (a[2] - a[0]) * (a[3] - a[1])
        + (b[2] - b[0]) * (b[3] - b[1])
        - inter
    )
    return inter / union if union > 0 else 0.0


def find_conflicts(items):
    pairs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if CONFLICT_PAIRS.get(items[i].name) == items[j].name:
                if box_iou(items[i].box, items[j].box) >= CONFLICT_IOU:
                    pairs.append((i, j))
    return pairs


def resolve_conflicts(items):
    drop = set()
    for i, j in find_conflicts(items):
        drop.add(i if items[i].conf < items[j].conf else j)
    return [it for k, it in enumerate(items) if k not in drop]


class Settings:
    """Runtime-adjustable thresholds and display mode."""

    def __init__(self, mode, per_class, global_conf):
        self.mode = mode
        self.per_class = per_class
        self.global_conf = global_conf

    def start(self, name):
        if self.per_class:
            return CLASS_CONF.get(name, DEFAULT_CLASS_CONF)
        return self.global_conf

    def keep(self, name):
        return max(
            INFERENCE_CONF_FLOOR,
            self.start(name) * HYSTERESIS_RATIO
        )


# ============================================================
# Temporal stabilizer (greedy IoU matching, per class)
# ============================================================

class Track:
    __slots__ = (
        "name", "box", "conf", "hits", "misses",
        "last_seen", "confirmed", "matched_now"
    )

    def __init__(self, det, now):
        self.name = det.name
        self.box = np.asarray(det.box, dtype=float).copy()
        self.conf = float(det.conf)
        self.hits = 1
        self.misses = 0
        self.last_seen = now
        self.confirmed = CONFIRM_HITS <= 1
        self.matched_now = True


class Stabilizer:
    """Associates detections to tracks by box overlap, same class only."""

    def __init__(self):
        self.tracks = []

    def reset(self):
        self.tracks = []

    def update(self, dets, now, settings):
        for t in self.tracks:
            t.matched_now = False

        pairs = []
        for ti, t in enumerate(self.tracks):
            for di, d in enumerate(dets):
                if d.name != t.name:
                    continue
                iou = box_iou(t.box, d.box)
                if iou >= MATCH_IOU:
                    pairs.append((iou, ti, di))
        pairs.sort(reverse=True)

        used_tracks, used_dets = set(), set()

        for iou, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue

            used_tracks.add(ti)
            used_dets.add(di)

            t, d = self.tracks[ti], dets[di]
            t.box = (
                BOX_SMOOTH_ALPHA * np.asarray(d.box, dtype=float)
                + (1.0 - BOX_SMOOTH_ALPHA) * t.box
            )
            t.conf = (
                CONF_SMOOTH_ALPHA * d.conf
                + (1.0 - CONF_SMOOTH_ALPHA) * t.conf
            )
            t.hits += 1
            t.misses = 0
            t.last_seen = now
            t.matched_now = True

            if t.hits >= CONFIRM_HITS:
                t.confirmed = True

        for di, d in enumerate(dets):
            if di not in used_dets and d.conf >= settings.start(d.name):
                self.tracks.append(Track(d, now))

        survivors = []
        for t in self.tracks:
            if t.matched_now:
                survivors.append(t)
                continue

            t.misses += 1
            if (
                t.confirmed
                and t.misses <= MAX_COAST_FRAMES
                and (now - t.last_seen) <= MAX_COAST_SECONDS
            ):
                survivors.append(t)

        self.tracks = survivors

        return [
            Item(t.name, t.conf, t.box, not t.matched_now)
            for t in self.tracks if t.confirmed
        ]


# ============================================================
# Statistics (consistency, NOT accuracy)
# ============================================================

class Stats:
    def __init__(self, names):
        self.names = list(names)
        self.reset()

    def reset(self):
        n = self.names
        self.t0 = time.perf_counter()
        self.frames = 0
        self.floor_frames = {k: 0 for k in n}
        self.best_conf_sum = {k: 0.0 for k in n}
        self.raw_on = {k: 0 for k in n}
        self.raw_flips = {k: 0 for k in n}
        self.stable_on = {k: 0 for k in n}
        self.stable_flips = {k: 0 for k in n}
        self.prev_raw = {k: False for k in n}
        self.prev_stable = {k: False for k in n}

    def update(self, dets, raw_items, stable_items):
        self.frames += 1
        best = {}

        for d in dets:
            best[d.name] = max(best.get(d.name, 0.0), d.conf)

        for name, conf in best.items():
            if name in self.floor_frames:
                self.floor_frames[name] += 1
                self.best_conf_sum[name] += conf

        raw_names = {i.name for i in raw_items}
        stable_names = {i.name for i in stable_items}

        for name in self.names:
            r, s = name in raw_names, name in stable_names
            self.raw_on[name] += r
            self.stable_on[name] += s

            if self.frames > 1:
                self.raw_flips[name] += r != self.prev_raw[name]
                self.stable_flips[name] += s != self.prev_stable[name]

            self.prev_raw[name], self.prev_stable[name] = r, s

    def report(self):
        secs = time.perf_counter() - self.t0
        print("\n" + "=" * 92)
        print(f"STATS over {self.frames} frames / {secs:.1f} s")
        print("'flips' = times the class went from shown to not shown (or back).")
        print("Fewer flips = steadier display. It says NOTHING about correctness.")
        print("-" * 92)
        print(
            f"{'Class':<12}{'seen>=floor':>12}{'mean best conf':>16}"
            f"{'raw shown':>11}{'raw flips':>11}"
            f"{'stab shown':>12}{'stab flips':>12}"
        )

        for name in self.names:
            seen = self.floor_frames[name]
            mean = self.best_conf_sum[name] / seen if seen else 0.0
            print(
                f"{name:<12}{seen:>12}{mean:>16.2f}"
                f"{self.raw_on[name]:>11}{self.raw_flips[name]:>11}"
                f"{self.stable_on[name]:>12}{self.stable_flips[name]:>12}"
            )

        print("=" * 92 + "\n")


# ============================================================
# Camera
# ============================================================

class LatestFrameReader:
    """Reads the camera in a background thread and keeps only the newest frame."""

    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.ts = 0.0
        self.frame_id = 0
        self.failed = False
        self.fail_reason = ""
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        fails = 0
        while not self.stop_flag.is_set():
            ok, frame = self.cap.read()

            if ok and frame is not None:
                fails = 0
                with self.lock:
                    self.frame = frame
                    self.ts = time.perf_counter()
                    self.frame_id += 1
            else:
                fails += 1
                if fails >= CAMERA_MAX_CONSECUTIVE_FAILS:
                    self.failed = True
                    self.fail_reason = (
                        f"{fails} consecutive failed reads "
                        "(camera unplugged, in use by another app, or driver error)"
                    )
                    return
                time.sleep(0.01)

    def get(self, last_id, timeout):
        deadline = time.perf_counter() + timeout

        while True:
            with self.lock:
                if self.frame is not None and self.frame_id != last_id:
                    return self.frame_id, self.frame, self.ts

            if self.failed or time.perf_counter() > deadline:
                return None

            time.sleep(0.002)

    def stop(self):
        self.stop_flag.set()
        self.thread.join(timeout=2.0)


def open_camera(index):
    api = {
        "dshow": getattr(cv2, "CAP_DSHOW", None),
        "msmf": getattr(cv2, "CAP_MSMF", None),
        "any": cv2.CAP_ANY,
    }

    order = [CAMERA_BACKEND] + [
        b for b in ("dshow", "msmf", "any") if b != CAMERA_BACKEND
    ]

    for backend in order:
        if api.get(backend) is None:
            continue

        cap = cv2.VideoCapture(index, api[backend])
        if not cap.isOpened():
            cap.release()
            continue

        if CAMERA_FOURCC:
            cap.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*CAMERA_FOURCC)
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        got_frame = False
        for _ in range(CAMERA_WARMUP_FRAMES):
            ok, _frame = cap.read()
            got_frame = got_frame or ok

        if not got_frame:
            cap.release()
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(
            f"Camera {index} opened via {backend}: "
            f"{w}x{h} @ {fps:.0f} fps (reported)"
        )
        return cap

    raise RuntimeError(
        f"Could not open camera index {index}. Close other apps using the webcam "
        "(Teams, Zoom, browser tabs) or try --camera 1."
    )


# ============================================================
# Drawing
# ============================================================

def draw_dashed_rect(img, x1, y1, x2, y2, color, thickness=1, dash=8):
    for x in range(x1, x2, dash * 2):
        xe = min(x + dash, x2)
        cv2.line(img, (x, y1), (xe, y1), color, thickness)
        cv2.line(img, (x, y2), (xe, y2), color, thickness)

    for y in range(y1, y2, dash * 2):
        ye = min(y + dash, y2)
        cv2.line(img, (x1, y), (x1, ye), color, thickness)
        cv2.line(img, (x2, y), (x2, ye), color, thickness)


def _overlaps(a, b):
    return not (
        a[2] <= b[0] or b[2] <= a[0]
        or a[3] <= b[1] or b[3] <= a[1]
    )


def draw_items(img, items):
    """Boxes first, then labels with simple collision avoidance."""
    h, w = img.shape[:2]
    ordered = sorted(items, key=lambda it: it.conf)

    for it in ordered:
        x1, y1, x2, y2 = (int(round(v)) for v in it.box)
        x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
        y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
        color = color_for(it.name)

        if it.coasting:
            draw_dashed_rect(img, x1, y1, x2, y2, color, 1)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    placed = []

    for it in sorted(items, key=lambda it: -it.conf):
        x1, y1 = int(round(it.box[0])), int(round(it.box[1]))
        text = f"{it.name} {it.conf:.2f}" + (" ~" if it.coasting else "")
        (tw, th), base = cv2.getTextSize(text, FONT, FONT_SCALE, 1)
        rw, rh = tw + 8, th + base + 6

        rx = max(0, min(w - rw, x1))
        ry = y1 - rh if y1 - rh >= 0 else y1

        for _ in range(20):
            rect = (rx, ry, rx + rw, ry + rh)
            if not any(_overlaps(rect, p) for p in placed):
                break
            ry += rh
            if ry + rh > h:
                ry = h - rh
                break

        rect = (rx, ry, rx + rw, ry + rh)
        placed.append(rect)
        color = color_for(it.name)

        cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color, -1)
        cv2.putText(
            img, text, (rx + 4, ry + th + 2),
            FONT, FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA
        )


def titled(img, title):
    strip = np.full((24, img.shape[1], 3), 35, dtype=np.uint8)
    cv2.putText(strip, title, (8, 17), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([strip, img])


def with_hud(view, lines):
    bar_h = 20 * len(lines) + 8
    bar = np.full((bar_h, view.shape[1], 3), 25, dtype=np.uint8)
    y = 20

    for line in lines:
        cv2.putText(
            bar, line, (8, y), FONT, 0.45,
            (255, 255, 255), 1, cv2.LINE_AA
        )
        y += 20

    return np.vstack([view, bar])


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="SAFE-X PPE detection test bench")
    p.add_argument("--model", type=Path, default=MODEL_PATH)
    p.add_argument("--camera", type=int, default=CAMERA_INDEX)
    p.add_argument("--imgsz", type=int, default=IMGSZ)
    p.add_argument(
        "--conf", type=float, default=GLOBAL_CONF,
        help="global threshold, used only when per-class thresholds are off"
    )
    p.add_argument(
        "--mode", choices=("raw", "stable", "both"), default=DISPLAY_MODE
    )
    p.add_argument(
        "--per-class", choices=("on", "off"),
        default="on" if USE_PER_CLASS_THRESHOLDS else "off"
    )
    return p.parse_args()


def extract_detections(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    names = result.names

    return [
        Det(str(names[c]), float(s), b.astype(float))
        for b, s, c in zip(xyxy, conf, cls)
    ]


def main():
    args = parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"YOLO model not found: {args.model}")

    model = YOLO(str(args.model))
    names = [model.names[i] for i in sorted(model.names)]

    print(f"Model: {args.model}")
    print(f"Classes ({len(names)}): {', '.join(names)}")

    if set(names) != set(EXPECTED_CLASSES):
        print("WARNING: model classes differ from EXPECTED_CLASSES:")
        print(f"  missing from model: {sorted(set(EXPECTED_CLASSES) - set(names))}")
        print(f"  extra in model:     {sorted(set(names) - set(EXPECTED_CLASSES))}")

    unknown = sorted(set(CLASS_CONF) - set(names))
    if unknown:
        print(f"WARNING: CLASS_CONF has names the model does not have: {unknown}")

    model.predict(
        source=np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), np.uint8),
        imgsz=args.imgsz,
        conf=INFERENCE_CONF_FLOOR,
        device=DEVICE,
        half=USE_HALF,
        verbose=False
    )

    settings = Settings(args.mode, args.per_class == "on", args.conf)
    stabilizer = Stabilizer()
    stats = Stats(names)

    cap = open_camera(args.camera)
    reader = LatestFrameReader(cap).start()

    print("Running. Click the video window, press Q to quit.")

    modes = ["both", "raw", "stable"]
    last_id = -1
    dropped_total = 0
    fps_ema = None
    infer_ema = None
    t_prev = time.perf_counter()

    try:
        while True:
            got = reader.get(last_id, CAMERA_READ_TIMEOUT_S)

            if got is None:
                reason = reader.fail_reason or (
                    f"no new frame for {CAMERA_READ_TIMEOUT_S:.1f} s"
                )
                print(f"CAMERA FAILURE: {reason}")
                break

            frame_id, frame, ts = got

            if last_id >= 0:
                dropped_total += max(0, frame_id - last_id - 1)

            last_id = frame_id
            t0 = time.perf_counter()

            result = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=INFERENCE_CONF_FLOOR,
                iou=NMS_IOU,
                max_det=MAX_DET,
                agnostic_nms=AGNOSTIC_NMS,
                device=DEVICE,
                half=USE_HALF,
                verbose=False
            )[0]

            infer_ms = (time.perf_counter() - t0) * 1000.0
            infer_ema = (
                infer_ms if infer_ema is None
                else 0.9 * infer_ema + 0.1 * infer_ms
            )

            dets = extract_detections(result)

            raw_items = [
                Item(d.name, d.conf, d.box, False)
                for d in dets if d.conf >= settings.start(d.name)
            ]

            keep_dets = [
                d for d in dets if d.conf >= settings.keep(d.name)
            ]
            stable_items = stabilizer.update(keep_dets, ts, settings)

            if RESOLVE_CONFLICTS:
                stable_items = resolve_conflicts(stable_items)

            stats.update(dets, raw_items, stable_items)

            mode = settings.mode
            panels = []

            if mode in ("raw", "both"):
                img = frame.copy()
                draw_items(img, raw_items)
                panels.append(titled(img, "RAW model output (no smoothing)"))

            if mode in ("stable", "both"):
                img = frame.copy()
                draw_items(img, stable_items)
                panels.append(
                    titled(img, "STABILIZED  (dashed ~ = held, not seen this frame)")
                )

            view = panels[0] if len(panels) == 1 else np.hstack(panels)

            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now

            if dt > 0:
                fps_ema = (
                    1.0 / dt if fps_ema is None
                    else 0.9 * fps_ema + 0.1 / dt
                )

            thr_text = (
                "per-class" if settings.per_class
                else f"global {settings.global_conf:.2f}"
            )

            hud = [
                f"{mode.upper()} | FPS {fps_ema or 0:4.1f} | "
                f"infer {infer_ema or 0:4.1f} ms | imgsz {args.imgsz} | "
                f"skipped frames {dropped_total}",
                f"Thr: {thr_text} (floor {INFERENCE_CONF_FLOOR:.2f}, "
                f"hyst x{HYSTERESIS_RATIO:.2f}) | raw {len(raw_items)} "
                f"stable {len(stable_items)} | "
                f"conflicts R{len(find_conflicts(raw_items))} "
                f"S{len(find_conflicts(stable_items))}",
                "Q quit | M mode | T thresholds | [ ] global conf | R reset | "
                "P stats   (undetected != absent)"
            ]

            view = with_hud(view, hud)

            if DISPLAY_SCALE != 1.0:
                view = cv2.resize(
                    view, None,
                    fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                    interpolation=cv2.INTER_AREA
                )

            cv2.imshow(WINDOW_NAME, view)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break
            elif key in (ord("m"), ord("M")):
                settings.mode = modes[
                    (modes.index(settings.mode) + 1) % len(modes)
                ]
            elif key in (ord("t"), ord("T")):
                settings.per_class = not settings.per_class
                stabilizer.reset()
            elif key == ord("["):
                settings.global_conf = max(
                    INFERENCE_CONF_FLOOR,
                    settings.global_conf - 0.05
                )
            elif key == ord("]"):
                settings.global_conf = min(
                    0.95, settings.global_conf + 0.05
                )
            elif key in (ord("r"), ord("R")):
                stats.reset()
                stabilizer.reset()
            elif key in (ord("p"), ord("P")):
                stats.report()

            try:
                if cv2.getWindowProperty(
                    WINDOW_NAME, cv2.WND_PROP_VISIBLE
                ) < 1:
                    break
            except cv2.error:
                break

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        reader.stop()
        cap.release()
        cv2.destroyAllWindows()

        if stats.frames > 0:
            stats.report()

        print("Stopped cleanly.")


if __name__ == "__main__":
    main()