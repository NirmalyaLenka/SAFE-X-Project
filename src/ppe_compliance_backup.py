from ultralytics import YOLO
from pathlib import Path

# =========================
# SAFE-X CONFIGURATION
# =========================

MODEL_PATH = Path(
    r"C:\Users\Nirmalya Lenka\runs\detect\results\safe_x_baseline\weights\best.pt"
)

IMAGE_SOURCE = Path(
    r"C:\Users\Nirmalya Lenka\Documents\SAFE-X\datasets\ppe\test\images"
)

CONFIDENCE = 0.25

CLASS_NAMES = {
    0: "helmet",
    1: "nohelmet",
    2: "novest",
    3: "person",
    4: "vest",
}


# =========================
# COMPLIANCE LOGIC
# =========================

def evaluate_compliance(detections):
    """
    Convert YOLO detections into a SAFE-X safety status.

    This is the first baseline version.
    """

    has_person = False
    has_helmet = False
    has_nohelmet = False
    has_vest = False
    has_novest = False

    for detection in detections:
        class_id = int(detection.cls[0])
        class_name = CLASS_NAMES.get(class_id, "unknown")

        if class_name == "person":
            has_person = True

        elif class_name == "helmet":
            has_helmet = True

        elif class_name == "nohelmet":
            has_nohelmet = True

        elif class_name == "vest":
            has_vest = True

        elif class_name == "novest":
            has_novest = True

    # No person detected
    if not has_person:
        return "NO PERSON"

    # Direct violation classes take priority
    if has_nohelmet and has_novest:
        return "CRITICAL"

    if has_nohelmet:
        return "NO HELMET"

    if has_novest:
        return "NO VEST"

    if has_helmet and has_vest:
        return "SAFE"

    return "CHECK"


# =========================
# MAIN
# =========================

def main():

    print("=" * 50)
    print("SAFE-X PPE COMPLIANCE ENGINE")
    print("=" * 50)

    print(f"Model: {MODEL_PATH}")
    print(f"Source: {IMAGE_SOURCE}")
    print()

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(IMAGE_SOURCE),
        conf=CONFIDENCE,
        device=0,
        verbose=False,
    )

    total = len(results)

    print(f"Processed images: {total}")
    print()

    for index, result in enumerate(results, start=1):

        detections = result.boxes

        status = evaluate_compliance(detections)

        image_name = Path(result.path).name

        print(
            f"[{index:03d}/{total:03d}] "
            f"{image_name:<35} -> {status}"
        )

    print()
    print("=" * 50)
    print("SAFE-X ANALYSIS COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()