from pathlib import Path
from ultralytics import YOLO

project = Path(r"C:\Users\Nirmalya Lenka\Documents\SAFE-X")
model_path = Path.home() / "runs" / "detect" / "runs" / "detect" / "safe_x_ppe_compliance_clean" / "weights" / "best.pt"
data_path = project / "datasets" / "ppe_compliance_clean" / "data_safe_x.yaml"
output_dir = project / "runs" / "safe_x_test_eval"

print("Model exists:", model_path.exists())
print("Dataset config exists:", data_path.exists())

if not model_path.exists() or not data_path.exists():
    raise FileNotFoundError("Model or dataset configuration was not found.")

output_dir.mkdir(parents=True, exist_ok=True)

model = YOLO(str(model_path))
print("\nStarting evaluation on test split...")

metrics = model.val(
    data=str(data_path),
    split="test",
    imgsz=640,
    plots=True,
    project=str(project / "runs"),
    name="safe_x_test_eval",
    exist_ok=True,
    workers=0
)

print("\n=== OVERALL TEST METRICS ===")
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)

print("\n=== PER-CLASS METRICS ===")
for i, name in model.names.items():
    if i < len(metrics.box.p) and i < len(metrics.box.r):
        print(
            f"{i}: {name} | "
            f"Precision={metrics.box.p[i]:.4f} | "
            f"Recall={metrics.box.r[i]:.4f}"
        )

print("\nEvaluation output:", metrics.save_dir)
