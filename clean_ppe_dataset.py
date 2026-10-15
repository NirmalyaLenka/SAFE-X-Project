from pathlib import Path
import shutil

source = Path(r"C:\Users\Nirmalya Lenka\Documents\SAFE-X\datasets\ppe_compliance")
target = Path(r"C:\Users\Nirmalya Lenka\Documents\SAFE-X\datasets\ppe_compliance_clean")

if target.exists():
    raise SystemExit(f"STOP: Destination already exists: {target}")

for split in ("train", "valid", "test"):
    src_images = source / split / "images"
    src_labels = source / split / "labels"
    dst_images = target / split / "images"
    dst_labels = target / split / "labels"

    if not src_images.is_dir() or not src_labels.is_dir():
        raise SystemExit(f"STOP: Missing image or label directory for {split}")

    dst_images.mkdir(parents=True)
    dst_labels.mkdir(parents=True)

    for image in src_images.iterdir():
        if image.is_file():
            shutil.copy2(image, dst_images / image.name)

    converted = 0
    unchanged = 0

    for label in src_labels.glob("*.txt"):
        output = []

        for line_number, line in enumerate(
            label.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue

            try:
                values = [float(x) for x in line.split()]
            except ValueError:
                raise SystemExit(f"Invalid number: {label}:{line_number}")

            if len(values) < 5:
                raise SystemExit(f"Too few values: {label}:{line_number}")

            class_id = values[0]
            coords = values[1:]

            if not class_id.is_integer() or not 0 <= class_id < 13:
                raise SystemExit(f"Invalid class ID: {label}:{line_number}")

            if len(values) == 5:
                if any(not 0 <= x <= 1 for x in coords):
                    raise SystemExit(f"Invalid box coordinate: {label}:{line_number}")
                output.append(line.strip())
                unchanged += 1
                continue

            # A polygon needs at least 3 (x, y) points.
            if len(coords) < 6 or len(coords) % 2 != 0:
                raise SystemExit(f"Unexpected annotation format: {label}:{line_number}")

            xs = coords[0::2]
            ys = coords[1::2]

            if any(not 0 <= x <= 1 for x in xs + ys):
                raise SystemExit(f"Polygon coordinate outside [0,1]: {label}:{line_number}")

            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            width, height = x2 - x1, y2 - y1

            if width <= 0 or height <= 0:
                raise SystemExit(f"Degenerate polygon: {label}:{line_number}")

            xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
            output.append(
                f"{int(class_id)} {xc:.8f} {yc:.8f} {width:.8f} {height:.8f}"
            )
            converted += 1

        (dst_labels / label.name).write_text(
            "\n".join(output) + ("\n" if output else ""),
            encoding="utf-8",
        )

    print(f"{split}: converted polygon rows={converted}; preserved box rows={unchanged}")

shutil.copy2(source / "data_safe_x.yaml", target / "data_safe_x.yaml")
print(f"Clean dataset created at: {target}")
