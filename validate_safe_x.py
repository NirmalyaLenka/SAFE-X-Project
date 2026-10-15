
from pathlib import Path
import cv2
from ultralytics import YOLO

MODEL_PATH = Path(
    r"C:\Users\Nirmalya Lenka\runs\detect\results"
    r"\safe_x_baseline\weights\best.pt"
)

OUTPUT_DIR = Path("validation_results")
OUTPUT_DIR.mkdir(exist_ok=True)

CONFIDENCE = 0.25
CAMERA_INDEX = 0  # Laptop webcam

def main():
    if not MODEL_PATH.is_file():
        print(f"Model not found: {MODEL_PATH}")
        return

    model = YOLO(str(MODEL_PATH))
    print("Classes:", model.names)
    print("Controls: SPACE = capture/test, Q = quit")

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("Cannot open laptop webcam.")
        return

    count = 0

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Failed to read camera frame.")
                break

            cv2.imshow("SAFE-X Validation - Press SPACE to test", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == 32:  # Spacebar
                count += 1

                result = model.predict(
                    frame,
                    conf=CONFIDENCE,
                    verbose=False
                )[0]

                annotated = result.plot()
                image_path = OUTPUT_DIR / f"test_{count:03d}.jpg"
                cv2.imwrite(str(image_path), annotated)

                print(f"\nTest {count}: {image_path}")
                found = False

                if result.boxes is not None:
                    for box in result.boxes:
                        found = True
                        class_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())
                        name = model.names[class_id]
                        print(f"  {name}: confidence={confidence:.3f}")

                if not found:
                    print("  No detections")

                print("  Annotated image saved.")

    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()