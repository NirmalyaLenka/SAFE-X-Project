\# SAFE-X 🦺



\### AI-Assisted Worker PPE Compliance Monitoring Using RFID and Computer Vision



SAFE-X is an engineering prototype designed to help monitor personal protective equipment (PPE) compliance in industrial environments such as factories, construction sites, and warehouses.



The project combines \*\*RFID-based worker identification\*\* with \*\*YOLO11n computer vision\*\* to identify workers, detect PPE-related objects, and support the review of potential safety violations.



The goal is to make PPE inspections more organized, traceable, and easier to review.



> \*\*Project status:\*\* Prototype under development. RFID identification has been tested, and the object-detection model has recorded evaluation results. Reliable real-time linking of worker identity to PPE status is still being developed.



\---



\## What Problem Does SAFE-X Solve?



Workers in industrial environments may be required to wear safety helmets, reflective vests, and other protective equipment.



Manual inspections can be repetitive and difficult to monitor continuously. Camera-based detection alone also does not tell supervisors which registered worker is being inspected.



SAFE-X explores how two technologies can work together:



\* \*\*RFID:\*\* Identifies a registered worker using an RFID card.

\* \*\*Computer vision:\*\* Detects people and PPE-related objects in camera frames.

\* \*\*Dashboard:\*\* Provides an interface for displaying worker information and inspection data.



The intended result is a system that helps supervisors identify cases that may require further inspection.



\##  How It Works



```text

&#x20;      RFID CARD                     CAMERA

&#x20;          |                            |

&#x20;          v                            v

&#x20;    MFRC522 Reader                Video Frames

&#x20;          |                            |

&#x20;          v                            v

&#x20;    ESP8266 NodeMCU                  OpenCV

&#x20;          |                            |

&#x20;          v                            v

&#x20;      TCP/IP Message                 YOLO11n

&#x20;          |                            |

&#x20;          v                            v

&#x20;     Worker Profile               PPE Detections

&#x20;          |                            |

&#x20;          +-------------+--------------+

&#x20;                        |

&#x20;                        v

&#x20;              PPE Decision Logic

&#x20;                        |

&#x20;                        v

&#x20;                Streamlit Dashboard

&#x20;                        |

&#x20;                        v

&#x20;               Review and Monitoring

```



\*\*Note:\*\* The complete worker-to-PPE decision pipeline is the intended architecture. Reliable real-time integration of the two branches remains under development.



\##  Key Features



\* RFID-based identification of registered workers.

\* Worker profiles stored in a JSON file.

\* Configurable PPE requirements for each worker.

\* YOLO11n object detection for five PPE-related classes.

\* Camera-frame processing using OpenCV.

\* A Streamlit dashboard for monitoring.

\* TCP/IP communication between the ESP8266 and Python application.

\* Model evaluation using precision, recall, and mean Average Precision (mAP).

\* Support for further development of inspection records and review workflows.



\##  AI Model



SAFE-X uses \*\*YOLO11n\*\*, a lightweight object-detection model from Ultralytics.



The model has five configured classes:



| Class      | Purpose                                     |

| ---------- | ------------------------------------------- |

| `person`   | Detects people                              |

| `helmet`   | Detects safety helmets                      |

| `nohelmet` | Detects the labeled missing-helmet category |

| `vest`     | Detects safety vests                        |

| `novest`   | Detects the labeled missing-vest category   |



These are the model's configured labels. Actual detection quality depends on image conditions, training data, and the scene being monitored.



\### Dataset



The previously audited dataset contains the following image counts:



| Split      |    Images |

| ---------- | --------: |

| Training   |     1,278 |

| Validation |       121 |

| Test       |        61 |

| \*\*Total\*\*  | \*\*1,460\*\* |



The test-label audit found no missing labels, empty label files, malformed bounding boxes, or invalid class IDs. Two exact duplicate image pairs were identified across the training and validation splits and should be removed before a future evaluation.



\### Reported Model Evaluation



| Metric    | Reported result |

| --------- | --------------: |

| Precision |           92.1% |

| Recall    |           88.0% |

| mAP@50    |           92.6% |

| mAP@50–95 |           68.5% |



These are recorded model-evaluation results; the exact evaluation split should be confirmed from the saved training and evaluation files.



\*\*Important:\*\* These metrics describe object-detection performance, not the accuracy of the complete SAFE-X application. In particular, the reported `nohelmet` recall is 75.6%, so missing-helmet detection requires further improvement.



\## 🛠️ Technologies Used



\### Hardware



\* ESP8266 NodeMCU

\* MFRC522 RFID reader

\* RFID cards or tags

\* Camera or smartphone-based IP camera

\* Laptop for computer vision and dashboard processing



\### Software



\* Python

\* Ultralytics YOLO11n

\* OpenCV

\* Streamlit

\* Pandas

\* TCP/IP sockets

\* JSON

\* Arduino IDE



\##  Project Structure



```text

SAFE-X/

│

├── safe\_x\_dashboard.py

│

├── src/

│   ├── safe\_x\_integrated.py

│   └── rfid\_receiver.py

│

├── data/

│   └── workers.json

│

├── datasets/

│   └── ppe/

│       ├── train/

│       ├── valid/

│       ├── test/

│       └── data.yaml

│

├── validation\_results/

│   └── nohelmet\_review\_contact\_sheet.jpg

│

├── .venv/                  # Local Python environment

└── README.md

```



This shows the main project files and expected data folders. Some folders or optional files may not be included in every copy of the repository.



The trained `best.pt` model weights are stored separately in the current development setup. They must be obtained and configured before running model inference.



\##  Getting Started



\### 1. Install the Prerequisites



You will need:



\* Python installed on your computer.

\* Git, if you want to clone the repository.

\* Arduino IDE for working with the ESP8266 firmware.

\* A compatible RFID reader and card for hardware testing.

\* The trained YOLO model weights.

\* A camera stream if you want to test video detection.



\### 2. Clone the Repository



Open a terminal and run:



```bash

git clone https://github.com/NirmalyaLenka/SAFE-X.git

cd SAFE-X

```



Replace the repository URL if your actual GitHub repository uses a different name or location.



\### 3. Create a Python Environment



On Windows, run:



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



If PowerShell blocks activation, you can run commands using the virtual environment's Python directly instead:



```powershell

.\\.venv\\Scripts\\python.exe --version

```



\### 4. Install Python Dependencies



With the virtual environment active, run:



```bash

python -m pip install --upgrade pip

python -m pip install streamlit streamlit-autorefresh ultralytics opencv-python pandas

```



These are the main dependencies visible in the dashboard code. Install any additional packages required by the integrated application if they are not included here.



\### 5. Configure the Project Paths



The current development version contains Windows-specific paths. Before running it on another computer, open `safe\_x\_dashboard.py` and update:



\* `ROOT`: the location of your SAFE-X project.

\* `MODEL\_PATH`: the location of your trained `best.pt` file.

\* `PHONE\_URL`: the correct URL for your camera stream.



The integrated application may also have its own paths, so check its configuration before using it.



For example, if you place the model weights inside a `models` folder, configure the dashboard's model path accordingly.



\### 6. Configure Worker Profiles



Open `data/workers.json` and review its existing structure before adding or changing records.



Each registered worker profile should contain the fields expected by the application, including:



\* Employee ID

\* Name

\* Designation

\* Required PPE



Use sample data when testing or sharing the project. Do not publish real employee information or RFID card UIDs in a public repository.



\### 7. Run the Dashboard



From the project root, run:



```powershell

.\\.venv\\Scripts\\python.exe -m streamlit run .\\safe\_x\_dashboard.py

```



Streamlit will display a local URL, typically:



```text

http://localhost:8501

```



Open the displayed address in your browser.



\*\*Important:\*\* The dashboard starts its own RFID server on port `8765`. Do not run `src/rfid\_receiver.py` at the same time, because both applications would attempt to use the same port.



The standalone RFID receiver can be run separately when testing RFID identification without the dashboard:



```powershell

.\\.venv\\Scripts\\python.exe .\\src\\rfid\_receiver.py

```



\## 🔌 Hardware Workflow



1\. Connect the MFRC522 RFID reader to the ESP8266 NodeMCU according to the firmware wiring configuration.

2\. Upload the appropriate Arduino firmware.

3\. Configure the laptop's IP address and server port in the firmware.

4\. Connect the laptop and NodeMCU to a network that allows communication between them.

5\. Start the Python application that will receive RFID events.

6\. Scan a registered card and check whether the corresponding worker profile is displayed.



Use the voltage requirements of the individual modules and the correct wiring configuration for your hardware.



\##  Current Development Status



| Component                                    | Status                                |

| -------------------------------------------- | ------------------------------------- |

| RFID card reading                            | Tested                                |

| Registered worker lookup                     | Tested                                |

| Python TCP receiver                          | Tested                                |

| YOLO11n model evaluation                     | Results available                     |

| Streamlit dashboard                          | Runs locally                          |

| Real-time camera-to-worker PPE association   | Needs further integration and testing |

| Reliable end-to-end PPE compliance decisions | Not yet established                   |



\##  Future Improvements



\* Improve detection of missing helmets and vests.

\* Remove duplicate images from dataset splits and repeat evaluation.

\* Expand the dataset to cover different lighting, camera angles, distances, and occlusion.

\* Stabilize live video processing and worker-to-detection association.

\* Add consistent, timestamped inspection records.

\* Add an appropriate manual-review workflow for uncertain cases.

\* Improve RFID security and protect worker data.

\* Measure inference speed and system latency on the target hardware.

\* Conduct more extensive field testing before practical deployment.



\##  Limitations and Safety Notice



SAFE-X is an experimental engineering prototype, not a certified workplace safety system.



Computer vision can miss objects or misclassify them. An absent `nohelmet` detection does not prove that a worker is wearing a helmet. Camera failures, network interruptions, identity mismatches, and environmental conditions may affect results.



All uncertain or potentially non-compliant cases should be handled through an appropriate human-review process. Do not rely on this prototype as the sole method of enforcing PPE requirements or protecting workers.



\##  References



1\. \[Ultralytics YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/)

2\. \[Ultralytics Model Performance Metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/)

3\. \[OpenCV VideoCapture Documentation](https://docs.opencv.org/4.x/d8/dfe/classcv\_1\_1VideoCapture.html)

4\. \[Streamlit Documentation](https://docs.streamlit.io/get-started)

5\. \[MFRC522 Arduino RFID Library](https://github.com/miguelbalboa/rfid)

6\. \[OSHA — Personal Protective Equipment in Construction](https://www.osha.gov/personal-protective-equipment/construction)



\##  Author



\*\*Nirmalya Lenka\*\*



GitHub: \[@NirmalyaLenka](https://github.com/NirmalyaLenka)



\---



\*SAFE-X — Exploring a more traceable and reviewable approach to PPE monitoring through embedded systems and computer vision.\*





---

## Maintenance & Audit

- **Audit date:** 10 November 2026
- **Documentation status:** README reviewed and audit record added on this date.
- **Scope and evidence:** This is a documentation-level maintenance pass. It does not establish that source code compiles, dependencies are current, security has been tested, or runtime behavior has been verified.
- **Code and tests:** No new functional code changes or test results are claimed by this README update. Run project-specific build, test, and security checks before relying on the project.
- **Maintenance baseline:** Documentation status recorded through 10 November 2026; revisit setup instructions, dependencies, security, known limitations, and tests when the project changes.
