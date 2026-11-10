# SAFE-X — PPE Compliance Monitoring Prototype

SAFE-X explores a workflow for industrial PPE inspection by combining RFID-based worker identification with computer-vision detections. The current dashboard is a development prototype; it can display detection and audit information, but it must not be treated as a certified safety system.

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/Detection-Ultralytics%20YOLO-111827)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![ESP8266](https://img.shields.io/badge/RFID%20bridge-ESP8266-CC0000)](https://www.espressif.com/)

## What it does

- Receives worker-card UIDs from an ESP8266/NodeMCU RFID reader over TCP.
- Loads worker profiles from `data/workers.json`.
- Uses a YOLO model and OpenCV camera frames to detect people and PPE-related classes.
- Displays the current inspection state in a Streamlit dashboard.
- Records audit and event state in SQLite.

**Important:** the repository contains several development scripts and earlier implementations. The root-level `safe_x_dashboard.py` is the current dashboard entry point. Files under `src/` are standalone/earlier implementation and test modules; do not assume every script uses the same configuration or is production-ready.

## How it works

```text
RFID card -> MFRC522 reader -> ESP8266 -> TCP port 8765
                                      |
                                      v
                               Worker profile lookup
                                      |
Camera -> OpenCV frames -> YOLO PPE model -> Detection review
                                      |
                                      v
                         Streamlit dashboard + SQLite audit
```

The dashboard's configured RFID listener binds to `0.0.0.0:8765`. The camera source is configured in the script and may need to be changed for your device. RFID identity and PPE detection are separate inputs; a reliable worker-to-detection association must be validated in the actual camera layout.

## Prerequisites

- Python 3.11 or a compatible version for the pinned environment.
- A webcam or configured network camera.
- An ESP8266/NodeMCU and MFRC522 RFID reader for the RFID workflow.
- A compatible trained PPE model at `models/best.pt`.

The repository's root `yolo11n.pt` is a generic pretrained model; it is **not a substitute for the project's trained PPE weights**.

## Setup

1. Clone the repository and enter its directory.
2. Create and activate a virtual environment.
3. Install dependencies:

   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # macOS/Linux: source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Put your trained PPE weights at `models/best.pt`. Model weights are intentionally excluded from Git; see [models/README.md](models/README.md).
5. Review `data/workers.json` and replace demo worker records with authorized test records only. Avoid committing real personal data.
6. Configure the camera URL/index and RFID sender address in `safe_x_dashboard.py` for your local hardware.
7. Start the dashboard from the repository root:

   ```bash
   streamlit run safe_x_dashboard.py
   ```

If the model file, camera, or dependencies are unavailable, the dashboard will not operate normally. Resolve those prerequisites before testing the full workflow.

## Hardware and network notes

- Use the MFRC522 module's supported supply voltage and logic levels; verify the exact board before wiring.
- Keep the ESP8266 sender and dashboard host on a network that can reach TCP port `8765`.
- Do not expose the listener or dashboard directly to the public internet.
- Use a controlled test area and human review for all uncertain detections.

## Repository structure

```text
.
├── data/
│   ├── workers.json                 # Worker profile data; use non-sensitive test records
│   └── safe_x_audit.db              # Local SQLite audit database (runtime state)
├── models/
│   └── README.md                    # Instructions for supplying trained weights
├── src/
│   ├── ppe_compliance.py            # Standalone camera/PPE evaluation prototype
│   ├── ppe_compliance_backup.py     # Backup/earlier variant
│   ├── rfid_receiver.py             # Standalone TCP RFID receiver
│   ├── safe_x_detection_test.py     # Detection test utility
│   └── safe_x_integrated.py         # Earlier integrated RFID/camera prototype
├── validation_results/              # Saved validation images
├── clean_ppe_dataset.py              # Dataset cleanup helper
├── evaluate_safe_x.py                # Model evaluation helper
├── safe_x_dashboard.py               # Current Streamlit dashboard
├── test_two_cameras.py               # Camera test utility
├── validate_safe_x.py                # Interactive local-camera validation
├── requirements.txt
├── SAFE_X_System_Pipeline.png
└── README.md
```

## Skills demonstrated

- Python application structure and dependency management
- Computer vision with OpenCV and Ultralytics YOLO
- Streamlit dashboard development
- RFID integration and TCP socket communication
- SQLite event/audit storage
- Hardware integration, camera calibration, and test documentation

## Limitations and safety

- This is a prototype, not a certified PPE detection or access-control product.
- Detection quality depends on the trained weights, lighting, camera angle, distance, occlusion, and dataset coverage.
- A model prediction is not proof that PPE is correctly worn. False positives and false negatives are possible.
- Do not use it as the sole method for workplace safety decisions or to deny access without a validated human-review procedure.
- Scripts under `src/` may contain machine-specific paths from earlier experiments. Review them before running on another computer.

## Maintenance & validation

- **Documentation review:** 10 November 2026.
- **Scope:** README formatting, repository map, and model setup instructions were revised.
- **Validation status:** No end-to-end test or hardware run was performed as part of this documentation change. Compile/run checks and model validation remain necessary.
