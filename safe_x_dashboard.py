from __future__ import annotations

import json
import queue
import socket
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# SAFE-X configuration
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
WORKERS_FILE = ROOT / "data" / "workers.json"
MODEL_PATH = ROOT / "models" / "best.pt"
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "safe_x_audit.db"

RFID_HOST = "0.0.0.0"
RFID_PORT = 8765
PHONE_URL = "http://10.213.204.242:8080/video"
DEFAULT_CORRECTION_MINUTES = 5
DEFAULT_OVERRIDE_SECONDS = 60
FIRE_CLEAR_SECONDS = 10
MAX_HISTORY_ROWS = 500

DATA_DIR.mkdir(parents=True, exist_ok=True)
st.set_page_config(page_title="SAFE-X | Industrial Safety", page_icon="🦺", layout="wide")
st_autorefresh(interval=1000, key="safe_x_refresh")

# -----------------------------------------------------------------------------
# Database: persistent audit events, PPE correction windows, scoped overrides,
# and the latched emergency state. The original workers.json remains untouched.
# -----------------------------------------------------------------------------
def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=8, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db_connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_ts TEXT NOT NULL,
                epoch REAL NOT NULL,
                event_type TEXT NOT NULL,
                worker TEXT DEFAULT '',
                uid TEXT DEFAULT '',
                device_ip TEXT DEFAULT '',
                identity TEXT DEFAULT '',
                ppe_status TEXT DEFAULT '',
                detections TEXT DEFAULT '[]',
                decision TEXT DEFAULT '',
                reason TEXT DEFAULT '',
                deadline_epoch REAL,
                operator TEXT DEFAULT '',
                override_reason TEXT DEFAULT '',
                expires_epoch REAL,
                extra TEXT DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                uid TEXT PRIMARY KEY,
                worker TEXT NOT NULL,
                deadline_epoch REAL NOT NULL,
                denied_event_id INTEGER,
                status TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS overrides (
                uid TEXT PRIMARY KEY,
                worker TEXT NOT NULL,
                operator TEXT NOT NULL,
                reason TEXT NOT NULL,
                expires_epoch REAL NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_epoch REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.execute("INSERT OR IGNORE INTO app_state(key, value) VALUES('fire_latched', '0')")
        conn.execute("INSERT OR IGNORE INTO app_state(key, value) VALUES('last_fire_seen_epoch', '0')")


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_event(
    event_type: str,
    *,
    worker: str = "",
    uid: str = "",
    device_ip: str = "",
    identity: str = "",
    ppe_status: str = "",
    detections: list[str] | None = None,
    decision: str = "",
    reason: str = "",
    deadline_epoch: float | None = None,
    operator: str = "",
    override_reason: str = "",
    expires_epoch: float | None = None,
    extra: dict[str, Any] | None = None,
) -> int:
    with db_connect() as conn:
        cursor = conn.execute(
            """INSERT INTO events(
                event_ts, epoch, event_type, worker, uid, device_ip, identity,
                ppe_status, detections, decision, reason, deadline_epoch,
                operator, override_reason, expires_epoch, extra
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                now_ts(), time.time(), event_type, worker, uid, device_ip,
                identity, ppe_status, json.dumps(detections or []), decision,
                reason, deadline_epoch, operator, override_reason,
                expires_epoch, json.dumps(extra or {}),
            ),
        )
        return int(cursor.lastrowid)


def get_state(key: str, default: str = "") -> str:
    with db_connect() as conn:
        row = conn.execute("SELECT value FROM app_state WHERE key=?", (key,)).fetchone()
    return str(row["value"]) if row else default


def set_state(key: str, value: str) -> None:
    with db_connect() as conn:
        conn.execute(
            "INSERT INTO app_state(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )


def get_correction(uid: str) -> dict[str, Any] | None:
    with db_connect() as conn:
        row = conn.execute("SELECT * FROM corrections WHERE uid=?", (uid,)).fetchone()
    return dict(row) if row else None


def upsert_correction(uid: str, worker: str, deadline: float, event_id: int) -> None:
    with db_connect() as conn:
        conn.execute(
            """INSERT INTO corrections(uid, worker, deadline_epoch, denied_event_id, status)
               VALUES(?,?,?,?, 'ACTIVE')
               ON CONFLICT(uid) DO UPDATE SET
                 worker=excluded.worker,
                 deadline_epoch=excluded.deadline_epoch,
                 denied_event_id=excluded.denied_event_id,
                 status='ACTIVE'""",
            (uid, worker, deadline, event_id),
        )


def update_correction_status(uid: str, status: str) -> None:
    with db_connect() as conn:
        conn.execute("UPDATE corrections SET status=? WHERE uid=?", (status, uid))


def get_override(uid: str) -> dict[str, Any] | None:
    with db_connect() as conn:
        row = conn.execute("SELECT * FROM overrides WHERE uid=?", (uid,)).fetchone()
    return dict(row) if row else None


def active_override(uid: str) -> dict[str, Any] | None:
    row = get_override(uid)
    if row and row["active"] and float(row["expires_epoch"]) > time.time():
        return row
    return None


def expire_persistent_states() -> None:
    now = time.time()
    with db_connect() as conn:
        expired_overrides = conn.execute(
            "SELECT * FROM overrides WHERE active=1 AND expires_epoch<=?", (now,)
        ).fetchall()
        for row in expired_overrides:
            conn.execute("UPDATE overrides SET active=0 WHERE uid=?", (row["uid"],))
            conn.execute(
                """INSERT INTO events(event_ts, epoch, event_type, worker, uid, identity,
                   decision, reason, operator, override_reason, expires_epoch, extra)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (now_ts(), now, "OVERRIDE_EXPIRED", row["worker"], row["uid"],
                 "REGISTERED", "OVERRIDE EXPIRED", "Manual override time limit expired",
                 row["operator"], row["reason"], row["expires_epoch"], "{}"),
            )
        expired_corrections = conn.execute(
            "SELECT * FROM corrections WHERE status='ACTIVE' AND deadline_epoch<=?", (now,)
        ).fetchall()
        for row in expired_corrections:
            conn.execute("UPDATE corrections SET status='EXPIRED' WHERE uid=?", (row["uid"],))
            conn.execute(
                """INSERT INTO events(event_ts, epoch, event_type, worker, uid,
                   identity, decision, reason, deadline_epoch, extra)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (now_ts(), now, "CORRECTION_WINDOW_EXPIRED", row["worker"], row["uid"],
                 "REGISTERED", "ACCESS NOT GRANTED", "PPE correction window expired",
                 row["deadline_epoch"], "{}"),
            )


init_db()

# -----------------------------------------------------------------------------
# RFID and model resources
# -----------------------------------------------------------------------------
def normalize_uid(uid: Any) -> str:
    return "".join(ch for ch in str(uid).upper() if ch in "0123456789ABCDEF")


def load_workers() -> tuple[dict[str, dict[str, Any]], str | None]:
    if not WORKERS_FILE.exists():
        return {}, f"Worker registry not found: {WORKERS_FILE}"
    try:
        raw = json.loads(WORKERS_FILE.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {}, f"Cannot read workers.json: {exc}"

    if isinstance(raw, dict) and isinstance(raw.get("workers"), dict):
        records = list(raw["workers"].items())
    elif isinstance(raw, dict) and isinstance(raw.get("workers"), list):
        records = []
        for item in raw["workers"]:
            if not isinstance(item, dict):
                continue
            uid = next((item.get(k) for k in (
                "uid", "rfid_uid", "card_uid", "rfid", "card", "card_id", "tag_uid"
            ) if item.get(k)), None)
            records.append((uid, item))
    elif isinstance(raw, dict):
        records = list(raw.items())
    elif isinstance(raw, list):
        records = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            uid = next((item.get(k) for k in (
                "uid", "rfid_uid", "card_uid", "rfid", "card", "card_id", "tag_uid"
            ) if item.get(k)), None)
            records.append((uid, item))
    else:
        return {}, "Unsupported workers.json format"

    workers: dict[str, dict[str, Any]] = {}
    for uid, person in records:
        if uid and isinstance(person, dict):
            norm = normalize_uid(uid)
            if norm:
                workers[norm] = person
    return workers, None


def worker_name(worker: dict[str, Any] | None) -> str:
    if not worker:
        return "Unknown worker"
    for key in ("name", "worker_name", "full_name", "employee_name"):
        if worker.get(key):
            return str(worker[key])
    return "Registered worker"


@st.cache_resource
def load_model(path: str):
    if not Path(path).is_file():
        raise FileNotFoundError(f"YOLO model not found: {path}")
    return YOLO(path)


@st.cache_resource
def get_rfid_queue() -> queue.Queue:
    return queue.Queue()


@st.cache_resource
def get_rfid_server() -> dict[str, Any]:
    state: dict[str, Any] = {"status": "STARTING", "error": "", "thread": None}
    events = get_rfid_queue()

    def server_loop() -> None:
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((RFID_HOST, RFID_PORT))
            sock.listen(5)
            sock.settimeout(1.0)
            state["status"] = "LISTENING"
            while True:
                try:
                    conn, address = sock.accept()
                except socket.timeout:
                    continue
                except OSError as exc:
                    state["status"] = "ERROR"
                    state["error"] = str(exc)
                    break
                state["status"] = "DEVICE CONNECTED"
                try:
                    with conn:
                        conn.settimeout(10.0)
                        buffer = ""
                        while True:
                            try:
                                data = conn.recv(1024)
                            except socket.timeout:
                                continue
                            if not data:
                                break
                            buffer += data.decode("utf-8", errors="ignore")
                            while "\n" in buffer or "\r" in buffer:
                                positions = [p for p in (buffer.find("\n"), buffer.find("\r")) if p >= 0]
                                cut = min(positions)
                                uid = buffer[:cut].strip()
                                buffer = buffer[cut + 1:]
                                if uid:
                                    events.put({
                                        "UID": uid,
                                        "Device IP": address[0],
                                        "Time": now_ts(),
                                    })
                        if buffer.strip():
                            events.put({"UID": buffer.strip(), "Device IP": address[0], "Time": now_ts()})
                except Exception as exc:
                    state["error"] = f"RFID connection error: {exc}"
                finally:
                    if state.get("status") != "ERROR":
                        state["status"] = "LISTENING"
        except Exception as exc:
            state["status"] = "ERROR"
            state["error"] = str(exc)
            events.put({"SERVER_ERROR": str(exc), "Time": now_ts()})
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

    thread = threading.Thread(target=server_loop, daemon=True, name="SAFE-X-RFID")
    state["thread"] = thread
    thread.start()
    return state


@st.cache_resource
def get_camera_manager() -> dict[str, Any]:
    # One shared capture manager prevents Streamlit reruns from opening
    # duplicate camera handles. Failed sources are retried after a cooldown.
    return {
        "source": None,
        "capture": None,
        "backend": "",
        "read_failures": 0,
        "retry_after": 0.0,
    }


def release_camera(manager: dict[str, Any]) -> None:
    """Release the current capture safely and reset its frame-failure state."""
    capture = manager.get("capture")
    if capture is not None:
        try:
            capture.release()
        except Exception:
            pass
    manager["capture"] = None
    manager["backend"] = ""
    manager["read_failures"] = 0


def open_camera(source_choice: str, phone_url: str):
    """Open and test a camera using Windows-friendly backend fallbacks."""
    if source_choice == "Laptop webcam":
        source: Any = 0
        backends = [("DirectShow", cv2.CAP_DSHOW), ("Media Foundation", cv2.CAP_MSMF), ("Default", cv2.CAP_ANY)]
    else:
        source = phone_url.strip()
        # FFmpeg is generally a better first choice for HTTP/MJPEG IP streams.
        backends = [("FFmpeg", cv2.CAP_FFMPEG), ("Default", cv2.CAP_ANY)]
        if not source:
            return None, "No mobile camera URL was provided"

    failures: list[str] = []
    for backend_name, backend_id in backends:
        cap = None
        try:
            cap = cv2.VideoCapture(source, backend_id)
            if not cap.isOpened():
                failures.append(f"{backend_name}: could not open source")
                cap.release()
                continue

            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            # Some Windows camera drivers report isOpened() even when they
            # cannot produce frames. Test a few reads before keeping a handle.
            got_frame = False
            for _ in range(3):
                ok, test_frame = cap.read()
                if ok and test_frame is not None and test_frame.size:
                    got_frame = True
                    break
                time.sleep(0.12)

            if got_frame:
                return cap, backend_name

            failures.append(f"{backend_name}: opened but returned no frames")
            cap.release()
        except Exception as exc:
            failures.append(f"{backend_name}: {exc}")
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    return None, "; ".join(failures) if failures else "No compatible camera backend succeeded"


# -----------------------------------------------------------------------------
# PPE and experimental color-based emergency signal
# -----------------------------------------------------------------------------
def analyze_ppe(result, confidence_threshold: float) -> tuple[str, list[str], str, list[dict[str, Any]]]:
    """Uses only the known model classes; unknown/missing evidence is not a pass."""
    names_map = result.names if hasattr(result, "names") else {}
    detected: list[str] = []
    box_records: list[dict[str, Any]] = []

    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        for box in boxes:
            try:
                score = float(box.conf[0].item())
                class_index = int(box.cls[0].item())
                label = str(names_map.get(class_index, class_index)).strip().lower()
            except Exception:
                continue
            if score < confidence_threshold:
                continue
            detected.append(label)
            box_records.append({"class": label, "confidence": round(score, 3)})

    labels = set(detected)
    no_helmet_aliases = {"nohelmet", "no_helmet", "no helmet", "helmet_missing", "without_helmet"}
    no_vest_aliases = {"novest", "no_vest", "no vest", "vest_missing", "without_vest"}
    helmet_aliases = {"helmet", "hardhat", "hard_hat", "safety_helmet"}
    vest_aliases = {"vest", "safety_vest", "reflective_vest", "high_visibility_vest", "hi_vis_vest"}
    person_aliases = {"person", "worker", "human"}

    violations = []
    if labels.intersection(no_helmet_aliases):
        violations.append("No helmet detected")
    if labels.intersection(no_vest_aliases):
        violations.append("No safety vest detected")

    if violations:
        return "VIOLATION", detected, "; ".join(violations), box_records

    saw_helmet = bool(labels.intersection(helmet_aliases))
    saw_vest = bool(labels.intersection(vest_aliases))
    saw_person = bool(labels.intersection(person_aliases))
    if saw_person and saw_helmet and saw_vest:
        return "COMPLIANT", detected, "Person, helmet, and safety vest detected; no explicit PPE violation class detected", box_records

    if not detected:
        reason = "No confident detections in the current frame"
    else:
        missing = []
        if not saw_person:
            missing.append("person not confidently detected")
        if not saw_helmet:
            missing.append("helmet not confirmed")
        if not saw_vest:
            missing.append("safety vest not confirmed")
        reason = "PPE verification incomplete: " + ", ".join(missing)
    return "PENDING", detected, reason, box_records


def detect_red_orange_region(frame: np.ndarray, min_fraction: float) -> tuple[bool, float]:
    """Experimental color heuristic only; it does NOT recognize fire as an object."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_low = cv2.inRange(hsv, np.array([0, 110, 130]), np.array([12, 255, 255]))
    orange_yellow = cv2.inRange(hsv, np.array([13, 120, 150]), np.array([38, 255, 255]))
    red_high = cv2.inRange(hsv, np.array([165, 110, 130]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(cv2.bitwise_or(red_low, orange_yellow), red_high)
    # Remove isolated pixels so tiny compression artifacts don't trigger the demo alarm.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    fraction = float(cv2.countNonZero(mask)) / float(mask.shape[0] * mask.shape[1])
    return fraction >= min_fraction, fraction


def load_audit_rows(limit: int = MAX_HISTORY_ROWS) -> pd.DataFrame:
    with db_connect() as conn:
        df = pd.read_sql_query(
            "SELECT id, event_ts, event_type, worker, uid, identity, ppe_status, decision, reason, operator, override_reason, expires_epoch, deadline_epoch, device_ip "
            "FROM events ORDER BY id DESC LIMIT ?", conn, params=(limit,)
        )
    return df


def get_today_metrics() -> dict[str, int]:
    today = datetime.now().strftime("%Y-%m-%d") + "%"
    with db_connect() as conn:
        total = conn.execute(
            "SELECT COUNT(*) n FROM events WHERE event_ts LIKE ? AND event_type IN ('ACCESS_GRANTED','ACCESS_DENIED','ACCESS_PENDING','MANUAL_OVERRIDE_GRANTED')",
            (today,),
        ).fetchone()["n"]
        grants = conn.execute(
            "SELECT COUNT(*) n FROM events WHERE event_ts LIKE ? AND decision LIKE 'ACCESS GRANTED%'",
            (today,),
        ).fetchone()["n"]
        denies = conn.execute(
            "SELECT COUNT(*) n FROM events WHERE event_ts LIKE ? AND decision LIKE 'ACCESS DENIED%'",
            (today,),
        ).fetchone()["n"]
        ppe = conn.execute(
            "SELECT COUNT(*) n FROM events WHERE event_ts LIKE ? AND event_type='PPE_VIOLATION'",
            (today,),
        ).fetchone()["n"]
        fire = conn.execute(
            "SELECT COUNT(*) n FROM events WHERE event_ts LIKE ? AND event_type='FIRE_ALERT_ACTIVATED'",
            (today,),
        ).fetchone()["n"]
        overrides = conn.execute(
            "SELECT COUNT(*) n FROM events WHERE event_ts LIKE ? AND event_type='MANUAL_OVERRIDE_GRANTED'",
            (today,),
        ).fetchone()["n"]
    return {"scans": int(total), "grants": int(grants), "denies": int(denies),
            "ppe": int(ppe), "fire": int(fire), "overrides": int(overrides)}


def format_remaining(deadline: float) -> str:
    secs = max(0, int(deadline - time.time()))
    return f"{secs // 60:02d}:{secs % 60:02d}"


def trigger_fire_alert(source: str, confidence: float | None = None) -> None:
    if get_state("fire_latched", "0") != "1":
        set_state("fire_latched", "1")
        set_state("last_fire_seen_epoch", str(time.time()))
        log_event(
            "FIRE_ALERT_ACTIVATED", decision="FIRE ALERT — ACCESS LOCKDOWN",
            reason=source, extra={"source": source, "confidence_or_color_fraction": confidence},
        )
    else:
        set_state("last_fire_seen_epoch", str(time.time()))


def css_theme(fire_active: bool) -> None:
    if fire_active:
        app_bg = "#b00000"
        sidebar_bg = "#780000"
        panel_bg = "#3b0909"
        border = "#ff5555"
    else:
        app_bg = "#0b1220"
        sidebar_bg = "#07101d"
        panel_bg = "#111c2f"
        border = "#26364d"
    st.markdown(
        f"""
        <style>
        :root {{ color-scheme: dark; }}
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background: {app_bg} !important;
        }}
        [data-testid="stMainBlockContainer"] {{ padding-top: 1.2rem; padding-bottom: 2.8rem; max-width: 1600px; }}
        [data-testid="stSidebar"] {{ background: {sidebar_bg} !important; border-right: 1px solid {border}; }}
        .stApp, .stApp p, .stApp label, .stApp div, .stApp span {{ color: #f3f6fb; }}
        h1, h2, h3 {{ letter-spacing: -0.025em; }}
        .sx-eyebrow {{ color: #93a7c2 !important; letter-spacing: .18em; text-transform: uppercase; font-size: .72rem; font-weight: 800; }}
        .sx-header {{ font-size: 2.5rem; font-weight: 900; line-height: 1.05; margin: 0; color: #ffffff; }}
        .sx-subtitle {{ color: #b4c0d1 !important; margin-top: .45rem; }}
        .sx-panel {{ background: {panel_bg}; border: 1px solid {border}; border-radius: 16px; padding: 1rem 1.1rem; margin-bottom: .8rem; box-shadow: 0 12px 30px rgba(0,0,0,.14); }}
        .sx-kpi-label {{ color: #9fb0c8 !important; font-size: .75rem; font-weight: 750; text-transform: uppercase; letter-spacing: .07em; }}
        .sx-kpi-value {{ color: #ffffff; font-size: 1.85rem; font-weight: 850; line-height: 1.15; padding-top: .35rem; }}
        .sx-status {{ border-radius: 14px; padding: 1.1rem 1.2rem; font-size: 1.6rem; font-weight: 950; text-align: center; border: 1px solid rgba(255,255,255,.25); }}
        .sx-ok {{ background: #064e3b; color: #e7fff5 !important; }}
        .sx-deny {{ background: #5b1018; color: #fff1f2 !important; }}
        .sx-warn {{ background: #653b08; color: #fff7e7 !important; }}
        .sx-pending {{ background: #25344a; color: #f0f7ff !important; }}
        .sx-fire {{ background: #ff1010; color: #ffffff !important; animation: sx-fire-pulse .7s ease-in-out infinite alternate; border: 2px solid white; }}
        .sx-ppe-flash {{ color: #fff !important; background: #65000a; border: 3px solid #ff2525; border-radius: 14px; padding: 1rem; text-align: center; font-size: 1.35rem; font-weight: 950; animation: sx-ppe-pulse .65s step-end infinite; }}
        .sx-muted {{ color: #a9b8ce !important; font-size: .88rem; }}
        .sx-tag {{ display: inline-block; border: 1px solid #41536d; padding: .25rem .55rem; border-radius: 999px; font-size: .78rem; margin: .12rem .2rem .12rem 0; background: #17253a; }}
        @keyframes sx-ppe-pulse {{ 0%, 49% {{ background: #a40015; }} 50%, 100% {{ background: #29040a; }} }}
        @keyframes sx-fire-pulse {{ from {{ box-shadow: 0 0 12px rgba(255,0,0,.45); }} to {{ box-shadow: 0 0 38px rgba(255,70,70,.9); }} }}
        div[data-testid="stMetric"] {{ background: {panel_bg}; border: 1px solid {border}; border-radius: 14px; padding: .7rem .9rem; }}
        button[kind="primary"] {{ background: #d82232; border: 1px solid #ff5966; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# App startup and sidebar settings
# -----------------------------------------------------------------------------
workers, worker_error = load_workers()
rfid_server = get_rfid_server()
expire_persistent_states()

with st.sidebar:
    st.markdown("<div class='sx-eyebrow'>System controls</div>", unsafe_allow_html=True)
    st.header("Monitoring configuration")
    camera_choice = st.selectbox("Camera source", ["Laptop webcam", "Mobile IP camera"], index=0)
    phone_url = st.text_input("Mobile camera stream URL", value=PHONE_URL)
    confidence = st.slider("YOLO confidence", min_value=0.10, max_value=0.90, value=0.35, step=0.05)
    monitoring_enabled = st.toggle("Enable camera / PPE monitoring", value=True)
    st.divider()
    st.subheader("Experimental fire-color alert")
    color_alert_enabled = st.toggle("Enable red/orange color trigger", value=True)
    color_threshold_percent = st.slider(
        "Red/orange image area needed (%)", min_value=1, max_value=30, value=4,
        help="A color-only heuristic. Red clothing, signage, lights, or objects may trigger it. It is not a validated fire detector.",
    )
    correction_minutes = st.number_input(
        "PPE correction window (minutes)", min_value=1, max_value=30,
        value=DEFAULT_CORRECTION_MINUTES, step=1,
    )
    st.divider()
    st.subheader("Subsystem status")
    st.write(f"**RFID listener:** {rfid_server.get('status', 'UNKNOWN')}")
    if rfid_server.get("error"):
        st.warning(rfid_server["error"])
    st.caption(f"TCP port: {RFID_PORT}")
    st.caption(f"Worker records loaded: {len(workers)}")
    st.caption(f"AI weights: {'FOUND' if MODEL_PATH.is_file() else 'MISSING'}")

# -----------------------------------------------------------------------------
# Capture one current frame and run inference once per refresh.
# -----------------------------------------------------------------------------
manager = get_camera_manager()
source_key = "0" if camera_choice == "Laptop webcam" else phone_url.strip()
frame = None
annotated_frame = None
result = None
camera_message = "Camera monitoring is disabled."
ppe_status = "PENDING"
ppe_reason = "PPE has not been assessed in a current frame."
detected_classes: list[str] = []
box_records: list[dict[str, Any]] = []
color_alert_now = False
color_fraction = 0.0
model_fire_now = False
model_fire_confidence: float | None = None
model_names: dict[int, str] = {}

# A disabled camera is released, avoiding a camera that remains locked by this app.
if not monitoring_enabled:
    release_camera(manager)
    manager["source"] = None
    manager["retry_after"] = 0.0
else:
    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model file missing: {MODEL_PATH}")

        # If the source changed, dispose of the previous capture before opening another.
        if manager.get("source") != source_key:
            release_camera(manager)
            manager["source"] = source_key
            manager["retry_after"] = 0.0

        if manager.get("capture") is None and time.time() >= float(manager.get("retry_after", 0.0)):
            cap, backend_or_error = open_camera(camera_choice, phone_url)
            manager["capture"] = cap
            if cap is None:
                manager["backend"] = ""
                manager["retry_after"] = time.time() + 3.0
                camera_message = f"Camera unavailable; retrying shortly. Details: {backend_or_error}"
            else:
                manager["backend"] = backend_or_error
                manager["read_failures"] = 0

        camera = manager.get("capture")
        if camera is None:
            if not camera_message.startswith("Camera unavailable"):
                wait = max(0, int(float(manager.get("retry_after", 0.0)) - time.time()))
                camera_message = f"Camera reconnect cooldown ({wait}s). Check camera permissions, URL, and whether another app is using it."
        elif not camera.isOpened():
            release_camera(manager)
            manager["retry_after"] = time.time() + 3.0
            camera_message = "Camera disconnected; reconnecting shortly."
        else:
            ok, frame = camera.read()
            if not ok or frame is None or not getattr(frame, "size", 0):
                manager["read_failures"] = int(manager.get("read_failures", 0)) + 1
                camera_message = (
                    f"Camera is not returning frames ({manager['read_failures']}/3). "
                    "Check permissions, camera URL, and other apps using the camera."
                )
                frame = None
                if manager["read_failures"] >= 3:
                    release_camera(manager)
                    manager["retry_after"] = time.time() + 3.0
                    camera_message = "Camera frame reads failed repeatedly; released the camera and will retry in 3 seconds."
            else:
                manager["read_failures"] = 0
                camera_message = f"LIVE · {manager.get('backend') or 'camera backend'}"
                if color_alert_enabled:
                    color_alert_now, color_fraction = detect_red_orange_region(
                        frame, color_threshold_percent / 100.0
                    )
                model = load_model(str(MODEL_PATH))
                model_names = model.names if isinstance(model.names, dict) else {}
                result = model.predict(frame, conf=confidence, verbose=False, imgsz=640)[0]
                ppe_status, detected_classes, ppe_reason, box_records = analyze_ppe(result, confidence)
                fire_labels = {"fire", "flame", "smoke", "wildfire"}
                for item in box_records:
                    if item["class"].lower() in fire_labels:
                        model_fire_now = True
                        model_fire_confidence = max(model_fire_confidence or 0.0, float(item["confidence"]))
                annotated_frame = result.plot()
    except Exception as exc:
        camera_message = f"Monitoring error: {exc}"
        frame = None
        annotated_frame = None

# Latch a visible emergency on a supported model fire class or the explicitly
# experimental red/orange color heuristic. The current known model has no fire class.
if model_fire_now:
    trigger_fire_alert("YOLO fire/flame/smoke class detected", model_fire_confidence)
if color_alert_now:
    trigger_fire_alert(
        "EXPERIMENTAL red/orange color heuristic (not validated fire detection)",
        color_fraction,
    )

fire_active = get_state("fire_latched", "0") == "1"
css_theme(fire_active)

# Save/expire state after capture, before processing queued RFID reads.
expire_persistent_states()

if "scan_history_session" not in st.session_state:
    st.session_state.scan_history_session = []
if "last_attempt" not in st.session_state:
    st.session_state.last_attempt = None
if "last_color_fraction" not in st.session_state:
    st.session_state.last_color_fraction = 0.0
if "last_frame_epoch" not in st.session_state:
    st.session_state.last_frame_epoch = 0.0
if frame is not None:
    st.session_state.last_frame_epoch = time.time()
st.session_state.last_color_fraction = color_fraction

# -----------------------------------------------------------------------------
# Process RFID scans after current camera inference. A worker-to-person match is
# not available in this model; demonstrate one worker at the entry camera at a time.
# -----------------------------------------------------------------------------
rfid_events = get_rfid_queue()
while True:
    try:
        incoming = rfid_events.get_nowait()
    except queue.Empty:
        break

    if "SERVER_ERROR" in incoming:
        log_event("RFID_SERVER_ERROR", decision="RFID LISTENER ERROR", reason=str(incoming["SERVER_ERROR"]))
        continue

    raw_uid = str(incoming.get("UID", "")).strip()
    uid = normalize_uid(raw_uid)
    device_ip = str(incoming.get("Device IP", ""))
    if not uid:
        log_event("INVALID_RFID_SCAN", uid=raw_uid, device_ip=device_ip,
                  decision="ACCESS DENIED", reason="RFID payload did not contain a valid hexadecimal UID")
        continue

    worker_record = workers.get(uid)
    person = worker_name(worker_record) if worker_record else "UNKNOWN CARD"
    identity = "REGISTERED" if worker_record else "UNKNOWN"
    current_fire_active = get_state("fire_latched", "0") == "1"
    current_ppe_status = ppe_status if frame is not None else "PENDING"
    current_ppe_reason = ppe_reason if frame is not None else "Camera evidence unavailable; PPE cannot be verified"
    deadline = None
    override_row = active_override(uid)

    if not worker_record:
        decision = "ACCESS DENIED"
        reason = "Unknown RFID card — worker is not in the local registry."
        event_type = "ACCESS_DENIED"
    elif current_fire_active:
        decision = "ACCESS DENIED"
        reason = "Critical hazard lockout active. Manual PPE override cannot bypass fire lockdown."
        event_type = "ACCESS_DENIED"
    elif override_row:
        decision = "ACCESS GRANTED — MANUAL OVERRIDE"
        reason = f"Authorized override by {override_row['operator']}: {override_row['reason']}"
        event_type = "ACCESS_GRANTED"
    elif current_ppe_status == "VIOLATION":
        decision = "ACCESS DENIED"
        reason = current_ppe_reason
        event_type = "ACCESS_DENIED"
    elif current_ppe_status == "COMPLIANT":
        decision = "ACCESS GRANTED"
        reason = "Registered worker; current frame shows person, helmet, and vest with no explicit no-PPE detections."
        event_type = "ACCESS_GRANTED"
    else:
        decision = "ACCESS DENIED — VERIFY PPE"
        reason = current_ppe_reason + ". Access remains blocked until fresh PPE verification succeeds."
        event_type = "ACCESS_PENDING"

    scan_log_id = log_event(
        event_type, worker=person, uid=uid, device_ip=device_ip, identity=identity,
        ppe_status=current_ppe_status, detections=detected_classes, decision=decision,
        reason=reason, expires_epoch=(override_row["expires_epoch"] if override_row else None),
        extra={"raw_uid": raw_uid, "camera_source": camera_choice,
               "frame_epoch": st.session_state.last_frame_epoch},
    )

    if worker_record and current_ppe_status == "VIOLATION" and not current_fire_active:
        existing_correction = get_correction(uid)
        now_epoch = time.time()
        if existing_correction and existing_correction["status"] == "ACTIVE" and existing_correction["deadline_epoch"] > now_epoch:
            deadline = float(existing_correction["deadline_epoch"])
        else:
            deadline = now_epoch + int(correction_minutes) * 60
        upsert_correction(uid, person, deadline, scan_log_id)
        log_event("PPE_VIOLATION", worker=person, uid=uid, device_ip=device_ip,
                  identity=identity, ppe_status=current_ppe_status, detections=detected_classes,
                  decision="ACCESS DENIED", reason=current_ppe_reason,
                  deadline_epoch=deadline, extra={"linked_access_event_id": scan_log_id})
    elif worker_record and current_ppe_status == "COMPLIANT" and not current_fire_active:
        correction = get_correction(uid)
        if correction and correction["status"] == "ACTIVE":
            update_correction_status(uid, "RESOLVED")
            log_event("CORRECTION_COMPLETED", worker=person, uid=uid, identity=identity,
                      ppe_status=current_ppe_status, detections=detected_classes,
                      decision=decision, reason="Worker returned and the current PPE assessment passed.",
                      deadline_epoch=correction["deadline_epoch"],
                      extra={"linked_denial_event_id": correction["denied_event_id"]})

    attempt = {
        "worker": person, "uid": uid, "device_ip": device_ip,
        "identity": identity, "ppe_status": current_ppe_status,
        "detected_classes": detected_classes.copy(), "decision": decision,
        "reason": reason, "timestamp": now_ts(), "deadline_epoch": deadline,
        "event_id": scan_log_id, "fire_active": current_fire_active,
    }
    st.session_state.last_attempt = attempt
    st.session_state.scan_history_session.insert(0, attempt)
    st.session_state.scan_history_session = st.session_state.scan_history_session[:50]

# -----------------------------------------------------------------------------
# Header and emergency state
# -----------------------------------------------------------------------------
fire_active = get_state("fire_latched", "0") == "1"
now_epoch = time.time()
last_fire_seen = float(get_state("last_fire_seen_epoch", "0") or 0)
fire_clear_ready = fire_active and not color_alert_now and not model_fire_now and last_fire_seen > 0 and (now_epoch - last_fire_seen) >= FIRE_CLEAR_SECONDS

st.markdown("<div class='sx-eyebrow'>Industrial safety intelligence / live operations</div>", unsafe_allow_html=True)
st.markdown("<div class='sx-header'>SAFE-X <span style='font-size:1rem;color:#9fb0c8;font-weight:700'>COMMAND CENTER</span></div>", unsafe_allow_html=True)
st.markdown("<div class='sx-subtitle'>RFID Worker Verification · PPE Compliance · Hazard Monitoring · Audit Trail</div>", unsafe_allow_html=True)

header_cols = st.columns([1, 1, 1, 1])
header_cols[0].markdown(f"**SYSTEM**  \n{'🚨 EMERGENCY LOCKDOWN' if fire_active else '🟢 ONLINE'}")
header_cols[1].markdown(f"**CAMERA**  \n{'🟢 ' + camera_message if camera_message.startswith('LIVE') else '🟠 ' + camera_message}")
header_cols[2].markdown(f"**RFID**  \n{rfid_server.get('status', 'UNKNOWN')}")
header_cols[3].markdown(f"**LOCAL TIME**  \n{now_ts()}")

if fire_active:
    st.markdown(
        "<div class='sx-status sx-fire'>🔥 FIRE / CRITICAL HAZARD ALERT<br>ACCESS LOCKDOWN ACTIVE</div>",
        unsafe_allow_html=True,
    )
    st.error(
        "Emergency is latched. Normal access and the ordinary manual override are blocked. "
        "The current model has no fire class; if this was triggered by red/orange pixels, "
        "it is an experimental color heuristic and can produce false alarms."
    )
elif color_alert_enabled and frame is not None:
    st.caption(f"Experimental red/orange coverage in current frame: {color_fraction * 100:.2f}% (trigger: {color_threshold_percent}%). This is not validated fire recognition.")

if worker_error:
    st.error(worker_error)
    st.warning("Worker verification fails closed until the local data/workers.json file is available and readable.")

metrics = get_today_metrics()
metric_cols = st.columns(6)
metric_cols[0].metric("Registered workers", len(workers))
metric_cols[1].metric("Access attempts today", metrics["scans"])
metric_cols[2].metric("Granted today", metrics["grants"])
metric_cols[3].metric("Denied today", metrics["denies"])
metric_cols[4].metric("PPE violations", metrics["ppe"])
metric_cols[5].metric("Manual overrides", metrics["overrides"])

# -----------------------------------------------------------------------------
# Live camera and latest access attempt
# -----------------------------------------------------------------------------
left, right = st.columns([1.65, 1], gap="large")
with left:
    st.subheader("Live PPE monitoring")
    if annotated_frame is not None:
        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), width="stretch")
        st.caption(f"Camera: {camera_choice} · AI confidence threshold: {confidence:.2f}")
    elif frame is not None:
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch")
    else:
        st.warning(camera_message)
    tag_html = "".join(f"<span class='sx-tag'>{label}</span>" for label in sorted(set(detected_classes)))
    st.markdown("**Current detected classes**", unsafe_allow_html=True)
    st.markdown(tag_html or "<span class='sx-muted'>No confident detections in the current frame.</span>", unsafe_allow_html=True)
    if result is not None:
        names_display = ", ".join(f"{d['class']} ({d['confidence']:.2f})" for d in box_records) or "No confident boxes"
        st.caption(f"Detection evidence: {names_display}")
    st.info("PPE is assessed at scene level. The current model does not associate a person in the frame with an RFID card. For the demo, assess one worker at the entry camera at a time.")

with right:
    st.subheader("Entry decision")
    last_attempt = st.session_state.last_attempt
    if fire_active:
        st.markdown("<div class='sx-status sx-fire'>ACCESS DENIED<br>FIRE LOCKDOWN</div>", unsafe_allow_html=True)
        st.write("All normal entry attempts are blocked until an authorized hazard-clear process is completed.")
    elif last_attempt:
        decision = last_attempt["decision"]
        if "GRANTED" in decision:
            decision_class = "sx-ok"
        elif last_attempt["ppe_status"] == "VIOLATION":
            decision_class = "sx-deny"
        elif "DENIED" in decision:
            decision_class = "sx-deny"
        else:
            decision_class = "sx-pending"
        st.markdown(f"<div class='sx-status {decision_class}'>{decision}</div>", unsafe_allow_html=True)
        st.write(f"**Worker:** {last_attempt['worker']}")
        st.write(f"**RFID UID:** `{last_attempt['uid']}`")
        st.write(f"**Identity:** {last_attempt['identity']}")
        st.write(f"**PPE status:** {last_attempt['ppe_status']}")
        st.write(f"**Time:** {last_attempt['timestamp']}")
        st.write(f"**Reason:** {last_attempt['reason']}")
        if last_attempt["ppe_status"] == "VIOLATION" and last_attempt["identity"] == "REGISTERED":
            corr = get_correction(last_attempt["uid"])
            if corr and corr["status"] == "ACTIVE":
                remaining = format_remaining(float(corr["deadline_epoch"]))
                expiry = datetime.fromtimestamp(float(corr["deadline_epoch"])).strftime("%H:%M:%S")
                st.markdown(
                    f"<div class='sx-ppe-flash'>LOW / INCOMPLETE PPE KIT<br>ACCESS DENIED<br><span style='font-size:1rem'>RETURN WINDOW: {remaining}</span></div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"Correction deadline: {expiry}. The countdown is an opportunity to correct PPE, not permission to enter.")
            elif corr and corr["status"] == "EXPIRED":
                st.error("CORRECTION WINDOW EXPIRED — a fresh verification is required.")
        elif last_attempt["ppe_status"] == "VIOLATION":
            st.markdown("<div class='sx-ppe-flash'>LOW / INCOMPLETE PPE KIT<br>ACCESS DENIED</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='sx-status sx-pending'>AWAITING RFID SCAN</div>", unsafe_allow_html=True)
        st.caption("Scan a registered worker card at the entry point to record an access attempt.")

    # Valid, time-limited, worker-specific override. Fire lockout and unknown UID
    # are always higher priority and cannot be bypassed here.
    st.divider()
    st.subheader("Supervised manual override")
    can_override = bool(
        last_attempt and last_attempt.get("identity") == "REGISTERED"
        and "DENIED" in last_attempt.get("decision", "")
        and last_attempt.get("ppe_status") in {"VIOLATION", "PENDING"}
        and not fire_active
    )
    if not can_override:
        st.caption("A registered worker must have a current PPE-related denial. Overrides are disabled for unknown cards and active fire lockdown.")
    with st.form("manual_override_form", clear_on_submit=True):
        operator = st.text_input("Operator / supervisor ID", placeholder="Enter supervisor identifier")
        override_reason = st.text_area("Reason for override (required)", placeholder="Explain why an exception is authorized", height=70)
        override_minutes = st.selectbox("Override duration", [30, 60, 120, 300], index=1, format_func=lambda v: f"{v} seconds")
        confirmed = st.checkbox("I confirm this is an authorized, supervised exception.")
        override_submit = st.form_submit_button("AUTHORIZE MANUAL OVERRIDE", type="primary")
    if override_submit:
        latest = st.session_state.last_attempt
        if get_state("fire_latched", "0") == "1":
            st.error("Override refused: fire/critical hazard lockdown is active.")
        elif not latest or latest.get("identity") != "REGISTERED" or "DENIED" not in latest.get("decision", ""):
            st.error("Override refused: scan a registered worker with a current PPE-related denial first.")
        elif latest.get("ppe_status") not in {"VIOLATION", "PENDING"}:
            st.error("Override refused: this decision is not eligible for the PPE exception workflow.")
        elif not operator.strip() or not override_reason.strip() or not confirmed:
            st.error("Enter an operator ID and reason, then confirm the authorization.")
        else:
            uid = latest["uid"]
            exp = time.time() + int(override_minutes)
            with db_connect() as conn:
                conn.execute(
                    """INSERT INTO overrides(uid, worker, operator, reason, expires_epoch, active, created_epoch)
                       VALUES(?,?,?,?,?,1,?)
                       ON CONFLICT(uid) DO UPDATE SET worker=excluded.worker, operator=excluded.operator,
                         reason=excluded.reason, expires_epoch=excluded.expires_epoch, active=1, created_epoch=excluded.created_epoch""",
                    (uid, latest["worker"], operator.strip(), override_reason.strip(), exp, time.time()),
                )
            log_event("MANUAL_OVERRIDE_GRANTED", worker=latest["worker"], uid=uid,
                      identity="REGISTERED", ppe_status=latest["ppe_status"],
                      detections=latest["detected_classes"], decision="ACCESS GRANTED — MANUAL OVERRIDE",
                      reason="Supervisor granted a time-limited exception for this worker.",
                      operator=operator.strip(), override_reason=override_reason.strip(), expires_epoch=exp,
                      extra={"duration_seconds": int(override_minutes), "related_attempt_event_id": latest["event_id"]})
            latest["decision"] = "ACCESS GRANTED — MANUAL OVERRIDE"
            latest["reason"] = f"Supervised override by {operator.strip()}: {override_reason.strip()}"
            latest["timestamp"] = now_ts()
            st.session_state.last_attempt = latest
            st.success(f"Time-limited override granted for {latest['worker']} until {datetime.fromtimestamp(exp).strftime('%H:%M:%S')}. Audit record saved.")
            st.rerun()

    latest = st.session_state.last_attempt
    if latest and latest.get("identity") == "REGISTERED":
        ov = get_override(latest["uid"])
        if ov and ov["active"] and float(ov["expires_epoch"]) > time.time():
            remaining = max(0, int(float(ov["expires_epoch"]) - time.time()))
            st.success(f"Manual override active for {ov['worker']} · {remaining}s remaining · operator: {ov['operator']}")
            if st.button("Revoke this worker's override", key="revoke_override"):
                with db_connect() as conn:
                    conn.execute("UPDATE overrides SET active=0 WHERE uid=?", (latest["uid"],))
                log_event("OVERRIDE_REVOKED", worker=ov["worker"], uid=latest["uid"],
                          identity="REGISTERED", decision="OVERRIDE REVOKED",
                          reason="Supervisor revoked the active override.", operator=ov["operator"],
                          override_reason=ov["reason"], expires_epoch=ov["expires_epoch"])
                st.success("Override revoked and recorded in history.")
                st.rerun()

# -----------------------------------------------------------------------------
# Emergency controls. Acknowledgement does not clear the latch. Clearing requires
# an operator, reason, explicit confirmation, and the configured clear interval.
# -----------------------------------------------------------------------------
st.divider()
st.subheader("Emergency management")
fire_active = get_state("fire_latched", "0") == "1"
last_fire_seen = float(get_state("last_fire_seen_epoch", "0") or 0)
fire_clear_ready = fire_active and not color_alert_now and not model_fire_now and last_fire_seen > 0 and (time.time() - last_fire_seen) >= FIRE_CLEAR_SECONDS
if fire_active:
    ack_cols = st.columns([1, 2])
    if ack_cols[0].button("Acknowledge alert", key="ack_fire"):
        log_event("FIRE_ALERT_ACKNOWLEDGED", decision="FIRE LOCKDOWN STILL ACTIVE",
                  reason="Operator acknowledged the emergency alert; acknowledgement does not clear the hazard.")
        st.session_state.fire_acknowledged = True
        st.success("Alert acknowledged. Lockdown remains active.")
    if fire_clear_ready:
        st.info(f"No red/orange heuristic trigger has been seen for at least {FIRE_CLEAR_SECONDS} seconds. Authorized reset is available, but confirm the scene is safe in person.")
    else:
        st.warning("Hazard reset is locked until the visual trigger has been absent for at least 10 seconds. Do not use this demo color heuristic as proof that a real fire is extinguished.")
    with st.form("fire_reset_form", clear_on_submit=True):
        reset_operator = st.text_input("Authorized operator ID", key="fire_reset_operator")
        reset_reason = st.text_input("Clearance reason", key="fire_reset_reason", placeholder="Describe how the area was checked")
        physically_checked = st.checkbox("I have independently verified that the area is safe.")
        reset_submit = st.form_submit_button("CONFIRM HAZARD CLEAR / RESET", type="primary")
    if reset_submit:
        if not fire_clear_ready:
            st.error("Reset blocked: the trigger has not been absent for the required interval.")
        elif not reset_operator.strip() or not reset_reason.strip() or not physically_checked:
            st.error("Enter operator ID and reason, and confirm the area was independently checked.")
        else:
            set_state("fire_latched", "0")
            log_event("FIRE_LOCKDOWN_CLEARED", decision="LOCKDOWN CLEARED",
                      reason=reset_reason.strip(), operator=reset_operator.strip(),
                      extra={"clearance_confirmed_by_operator": True})
            st.success("Hazard state reset and logged. Verify the physical area independently before any real-world entry.")
            st.rerun()
else:
    st.caption("No active latched emergency. If enabled, the experimental color trigger can still be activated by red/orange objects and must not be treated as genuine fire recognition.")

# -----------------------------------------------------------------------------
# Persistent audit history
# -----------------------------------------------------------------------------
st.divider()
st.subheader("Persistent access and safety audit history")
with db_connect() as conn:
    total_events = int(conn.execute("SELECT COUNT(*) n FROM events").fetchone()["n"])
st.caption(f"{total_events} events stored locally in {DB_PATH.name}. Records persist across app reruns and restarts.")

history = load_audit_rows()
if not history.empty:
    fcols = st.columns([1.5, 1, 1])
    search = fcols[0].text_input("Search name / UID / reason", key="history_search")
    decision_options = ["All"] + sorted([str(v) for v in history["decision"].dropna().unique() if str(v)])
    decision_filter = fcols[1].selectbox("Decision filter", decision_options)
    type_options = ["All"] + sorted([str(v) for v in history["event_type"].dropna().unique() if str(v)])
    type_filter = fcols[2].selectbox("Event type", type_options)
    shown = history.copy()
    if search.strip():
        query = search.strip().lower()
        mask = shown.astype(str).apply(lambda col: col.str.lower().str.contains(query, regex=False, na=False)).any(axis=1)
        shown = shown[mask]
    if decision_filter != "All":
        shown = shown[shown["decision"] == decision_filter]
    if type_filter != "All":
        shown = shown[shown["event_type"] == type_filter]
    st.dataframe(shown, width="stretch", hide_index=True)
    export_bytes = shown.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered audit history (CSV)", data=export_bytes,
                       file_name=f"safe_x_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                       mime="text/csv")
else:
    st.info("No events recorded yet. Scan an RFID card to begin the demo.")

with st.expander("Model capability and safety notes"):
    st.markdown(
        """
        **Verified current YOLO classes:** `helmet`, `nohelmet`, `novest`, `person`, `vest`.

        - Automated PPE logic checks these real classes. It does not check gloves, goggles, or other unsupported equipment.
        - The model currently has **no fire or smoke class**. The red/orange trigger is an experimental color-pixel heuristic, not a trained fire detector; red clothing, signs, lights, or objects can trigger it.
        - PPE status is scene-level. The current implementation cannot associate a particular person in a multi-person frame with an RFID UID. Use one worker at a time at the entry camera for the demonstration.
        - This application records decisions but does not operate a physical door lock. Do not use this prototype as the sole basis for real industrial access or emergency response.
        - Manual override is a supervised demo workflow, not secure production authentication. The operator ID field alone does not authenticate a supervisor.
        """
    )

st.caption("SAFE-X prototype · Local processing · Audit events stored on this computer")

