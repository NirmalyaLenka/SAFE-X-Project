

import cv2
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime

# ---------------- SAFE-X CONFIGURATION ----------------
MODEL_PATH = Path(
    r"C:\Users\Nirmalya Lenka\runs\detect\results\safe_x_baseline\weights\best.pt"
)
CAMERA_ID = 0
CONFIDENCE = 0.25
DEVICE = 0

CLASS_NAMES = {
    0: "helmet",
    1: "nohelmet",
    2: "novest",
    3: "person",
    4: "vest",
}

# OpenCV colors are BGR
COLORS = {
    "SAFE": (0, 200, 0),
    "NO HELMET": (0, 165, 255),
    "NO VEST": (0, 165, 255),
    "CRITICAL": (0, 0, 255),
    "CHECK": (0, 220, 255),
}


def center_of(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def inside_region(point, person_box, y_min, y_max):
    px, py = point
    x1, y1, x2, y2 = person_box

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    # Allow a small horizontal margin for PPE boxes.
    horizontal = x1 - width * 0.10 <= px <= x2 + width * 0.10
    vertical = y1 + height * y_min <= py <= y1 + height * y_max

    return horizontal and vertical


def evaluate_person(person_box, detections):
    helmet = False
    nohelmet = False
    vest = False
    novest = False

    for detection in detections:
        name = detection["name"]
        point = center_of(detection["box"])

        if name in ("helmet", "nohelmet"):
            if inside_region(point, person_box, 0.0, 0.48):
                if name == "helmet":
                    helmet = True
                else:
                    nohelmet = True

        elif name in ("vest", "novest"):
            if inside_region(point, person_box, 0.20, 0.90):
                if name == "vest":
                    vest = True
                else:
                    novest = True

    # Explicit violation detections take priority.
    if nohelmet and novest:
        status = "CRITICAL"
    elif nohelmet:
        status = "NO HELMET"
    elif novest:
        status = "NO VEST"
    elif helmet and vest:
        status = "SAFE"
    else:
        status = "CHECK"

    return status, helmet, nohelmet, vest, novest


def main():
    if not MODEL_PATH.is_file():
        print(f"ERROR: Model not found:\n{MODEL_PATH}")
        return

    print("Loading SAFE-X model...")
    model = YOLO(str(MODEL_PATH))

    print("Opening laptop webcam...")
    camera = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_ID)

    if not camera.isOpened():
        print("ERROR: Webcam could not be opened.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("SAFE-X LIVE MONITORING STARTED")
    print("Press Q in the video window to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Warning: Could not read webcam frame.")
                break

            result = model.predict(
                source=frame,
                conf=CONFIDENCE,
                device=DEVICE,
                verbose=False
            )[0]

            detections = []

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    coords = box.xyxy[0].cpu().tolist()

                    name = CLASS_NAMES.get(
                        class_id, result.names.get(class_id, str(class_id))
                    )

                    detections.append({
                        "name": name,
                        "confidence": confidence,
                        "box": coords,
                    })

                    x1, y1, x2, y2 = map(int, coords)
                    color = (255, 180, 0)

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), color, 2
                    )
                    cv2.putText(
                        frame,
                        f"{name} {confidence:.2f}",
                        (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

            people = [
                detection["box"]
                for detection in detections
                if detection["name"] == "person"
            ]

            statuses = []

            for index, person_box in enumerate(people, start=1):
                status, helmet, nohelmet, vest, novest = evaluate_person(
                    person_box, detections
                )
                statuses.append(status)

                x1, y1, x2, y2 = map(int, person_box)
                color = COLORS[status]

                # Draw a thicker outline around each person.
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), color, 3
                )

                label = f"Worker {index}: {status}"
                label_y = max(25, y1 - 30)

                cv2.putText(
                    frame,
                    label,
                    (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2
                )

            # Dashboard overlay
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 55),
                          (25, 25, 25), -1)
            cv2.putText(
                frame, "SAFE-X | LIVE PPE MONITOR",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2
            )
            cv2.putText(
                frame,
                f"People: {len(people)} | SAFE: {statuses.count('SAFE')} "
                f"| Violations: {sum(s in ('NO HELMET', 'NO VEST', 'CRITICAL') for s in statuses)}",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1
            )

            cv2.imshow("SAFE-X Live Compliance Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nStopping SAFE-X...")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("SAFE-X webcam stopped.")


if __name__ == "__main__":
    main()