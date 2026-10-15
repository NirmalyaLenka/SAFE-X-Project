
import json
import socket
import threading
from pathlib import Path
from datetime import datetime

import cv2
from ultralytics import YOLO

# ============================================================
# SAFE-X: RFID WORKER IDENTIFICATION + LIVE PPE INSPECTION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(
    r"C:\Users\Nirmalya Lenka\runs\detect\results"
    r"\safe_x_baseline\weights\best.pt"
)

WORKERS_PATH = ROOT / "data" / "workers.json"

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8765
CAMERA_INDEX = 0
CONFIDENCE = 0.35

# Shared state between RFID receiver and camera
state_lock = threading.Lock()
current_scan = {
    "uid": None,
    "worker": None,
    "unknown": False,
    "scan_time": None,
}

stop_event = threading.Event()


def load_workers():
    with open(WORKERS_PATH, "r", encoding="utf-8-sig") as file:
        data = json.load(file)

    return {
        uid.strip().upper(): worker
        for uid, worker in data["workers"].items()
    }


def update_scan(uid, workers):
    uid = uid.strip().upper()

    with state_lock:
        current_scan["uid"] = uid
        current_scan["worker"] = workers.get(uid)
        current_scan["unknown"] = uid not in workers
        current_scan["scan_time"] = datetime.now()

    print(f"\nRFID CARD: {uid}")

    worker = workers.get(uid)

    if worker:
        print("IDENTITY: REGISTERED")
        print(f"Employee ID: {worker['employee_id']}")
        print(f"Name: {worker['name']}")
        print(f"Designation: {worker['designation']}")
        print("Required PPE: " + ", ".join(worker["required_ppe"]))
        print("PPE DECISION: CAMERA INSPECTION IN PROGRESS")
    else:
        print("IDENTITY: UNKNOWN CARD")
        print("ACCESS: DENY / MANUAL REVIEW")

    print("-" * 50)


def rfid_server(workers):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server.bind((SERVER_HOST, SERVER_PORT))
        server.listen(5)
        server.settimeout(1.0)

        print(f"RFID receiver listening on port {SERVER_PORT}")

        while not stop_event.is_set():
            try:
                conn, address = server.accept()
            except socket.timeout:
                continue

            print(f"NodeMCU connected: {address}")

            with conn:
                buffer = ""

                while not stop_event.is_set():
                    try:
                        data = conn.recv(1024)
                    except (ConnectionResetError, OSError):
                        break

                    if not data:
                        break

                    buffer += data.decode("utf-8", errors="replace")

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        uid = line.strip()

                        if uid:
                            update_scan(uid, workers)

    except OSError as error:
        if not stop_event.is_set():
            print(f"RFID SERVER ERROR: {error}")
            print("Check whether another receiver is using port 8765.")

    finally:
        server.close()


def snapshot_scan():
    with state_lock:
        return current_scan.copy()


def inspect_frame(result, scan):
    """Make conservative decisions from model detections."""

    boxes = result.boxes
    names = result.names

    detected = []

    for box in boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        class_name = str(names[class_id]).strip().lower()

        detected.append((class_name, confidence))

    people = [item for item in detected if item[0] == "person"]
    helmets = [item for item in detected if item[0] == "helmet"]
    vests = [item for item in detected if item[0] == "vest"]
    no_helmets = [item for item in detected if item[0] == "nohelmet"]
    no_vests = [item for item in detected if item[0] == "novest"]

    if scan["uid"] is None:
        return "SCAN RFID CARD", "WAITING"

    if scan["unknown"]:
        return "UNKNOWN CARD - ACCESS DENIED", "DENY"

    if not people:
        return "WAITING FOR WORKER IN CAMERA ZONE", "WAITING"

    # This demo requires one worker in a controlled inspection zone.
    if len(people) > 1:
        return "MULTIPLE PEOPLE - MANUAL REVIEW", "MANUAL REVIEW"

    # Explicit model detections of missing PPE are violations.
    if no_helmets or no_vests:
        violations = []

        if no_helmets:
            violations.append("NO HELMET")

        if no_vests:
            violations.append("NO VEST")

        return "DENY - " + " / ".join(violations), "DENY"

    # Do not treat a missing detection as proof of a violation,
    # and do not approve PPE the model cannot inspect.
    missing_confirmations = []

    if not helmets:
        missing_confirmations.append("HELMET UNCONFIRMED")

    if not vests:
        missing_confirmations.append("VEST UNCONFIRMED")

    if missing_confirmations:
        return (
            "MANUAL REVIEW - " + " / ".join(missing_confirmations),
            "MANUAL REVIEW",
        )

    worker = scan["worker"]
    required = set(worker.get("required_ppe", []))

    unsupported = []

    if "safety_footwear" in required:
        unsupported.append("FOOTWEAR NOT VERIFIED")

    if "task_specific_electrical_protection" in required:
        unsupported.append("ELECTRICAL PROTECTION NOT VERIFIED")

    if unsupported:
        return "MANUAL REVIEW - " + " / ".join(unsupported), "MANUAL REVIEW"

    # Conservative fallback: no automatic approval.
    return "MANUAL REVIEW REQUIRED", "MANUAL REVIEW"


def draw_panel(frame, scan, decision_text, decision):
    height, width = frame.shape[:2]
    panel_height = min(190, height)

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (width, panel_height),
        (25, 25, 25),
        -1,
    )
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    lines = []

    if scan["worker"]:
        worker = scan["worker"]
        lines.append(f"Worker: {worker['name']}")
        lines.append(f"Employee ID: {worker['employee_id']}")
        lines.append(f"Role: {worker['designation']}")
    elif scan["unknown"]:
        lines.append(f"Unknown RFID: {scan['uid']}")
    else:
        lines.append("Worker: Not identified")

    lines.append(f"Decision: {decision}")
    lines.append(decision_text)

    y = 28

    for line in lines:
        cv2.putText(
            frame,
            line[:100],
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 30

    cv2.putText(
        frame,
        "Q = QUIT | Scan a new RFID card to switch workers",
        (12, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

    if not WORKERS_PATH.is_file():
        raise FileNotFoundError(f"Worker database not found: {WORKERS_PATH}")

    workers = load_workers()

    print("=" * 55)
    print("SAFE-X INTEGRATED WORKER SAFETY DEMO")
    print(f"Registered worker profiles: {len(workers)}")
    print(f"Model: {MODEL_PATH}")
    print("=" * 55)

    model = YOLO(str(MODEL_PATH))

    receiver_thread = threading.Thread(
        target=rfid_server,
        args=(workers,),
        daemon=True,
    )
    receiver_thread.start()

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        stop_event.set()
        raise RuntimeError(
            "Could not open webcam. Close other camera apps and check CAMERA_INDEX."
        )

    print("Camera ready. Scan a registered RFID card.")
    print("Press Q in the camera window to exit.")

    try:
        while True:
            ok, frame = camera.read()

            if not ok:
                print("Camera frame unavailable.")
                break

            scan = snapshot_scan()

            results = model.predict(
                source=frame,
                imgsz=640,
                conf=CONFIDENCE,
                verbose=False,
            )

            result = results[0]
            annotated = result.plot()

            decision_text, decision = inspect_frame(result, scan)

            draw_panel(
                annotated,
                scan,
                decision_text,
                decision,
            )

            cv2.imshow("SAFE-X | RFID + PPE Inspection", annotated)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    except KeyboardInterrupt:
        print("\nStopping SAFE-X...")

    finally:
        stop_event.set()
        camera.release()
        cv2.destroyAllWindows()
        print("SAFE-X stopped.")


if __name__ == "__main__":
    main()