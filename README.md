<div align="center">

# 🔩 Steel Defect Vision

**Real-time steel surface defect detection powered by YOLOv8**

Detects 15 types of industrial steel surface defects using a custom-trained YOLOv8s model, deployed as an interactive web application.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B.svg)](https://streamlit.io/)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 🚀 Quick Start

### 🌐 Live Demo
👉 **[https://steel-defect-vision.streamlit.app/](https://steel-defect-vision.streamlit.app/)**

Upload any steel surface image and get instant defect detection results with bounding boxes, confidence scores, and defect classification — no setup required.

---

## 📊 Model Performance

| Metric | Value |
|:---:|---:|
| **mAP@50** | 0.7948 |
| **mAP@50-95** | 0.4558 |
| **Precision** | 0.7784 |
| **Recall** | 0.7365 |
| **Training Time** | 1.126 hours |
| **Inference Speed** | ~3.7 ms / image |

---

## 💡 What This Project Does

Steel manufacturing plants lose millions annually due to surface defects that slip through manual quality inspection. This project automates that process — a computer vision model scans steel surfaces and instantly identifies defects with pixel-level accuracy.

### ✨ Key Features

- **🖼️ Single Image Inspection** — Upload one image, get annotated results instantly
- **📦 Batch Inspection** — Upload multiple images, get a full defect summary report
- **📈 Model Performance Tab** — View mAP, precision-recall curves, and confusion matrix
- **⚙️ Adjustable Thresholds** — Tune confidence, IoU, and inference resolution for your use case

> 💡 **Tip for real-world images:** For glossy or reflective steel photos, lower the confidence threshold to 0.15–0.25 and set resolution to 832 for best results. The model was trained on controlled laboratory conditions.

---

## 🏷️ Defect Classes (15 Types)

| ID | Class | Source Dataset |
|:--:|:---|:---|
| 0 | Crazing | NEU-DET |
| 1 | Inclusion | NEU-DET + GC10-DET |
| 2 | Patches | NEU-DET |
| 3 | Pitted Surface | NEU-DET |
| 4 | Rolled-in Scale | NEU-DET |
| 5 | Scratches | NEU-DET |
| 6 | Punching Hole | GC10-DET |
| 7 | Welding Line | GC10-DET |
| 8 | Crescent Gap | GC10-DET |
| 9 | Water Spot | GC10-DET |
| 10 | Oil Spot | GC10-DET |
| 11 | Silk Spot | GC10-DET |
| 12 | Rolled Pit | GC10-DET |
| 13 | Crease | GC10-DET |
| 14 | Waist Folding | GC10-DET |

---

## 📦 Dataset Information

| Dataset | Classes | Images |
|:---|:---:|---:|
| [NEU-DET](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) | 6 | 1,799 |
| [GC10-DET](https://www.kaggle.com/datasets/alex000kim/gc10-det) | 10 | 2,161 |
| **Combined Total** | **15** | **3,960** |

**Dataset Details:**
- **Train / Val Split:** 80% / 20%
- **Annotation Format:** Pascal VOC XML → converted to YOLO TXT
- **Training Notebook:** [View on Kaggle](https://www.kaggle.com/ahsanneural)

---

## 🤖 Model Architecture

| Parameter | Value |
|:---|:---|
| **Architecture** | YOLOv8s |
| **Framework** | Ultralytics 8.4.46 |
| **Input Size** | 640 × 640 |
| **Epochs** | 50 |
| **Hardware** | Tesla T4 GPU (Kaggle) |
| **Weights File** | `best.pt` (22.5 MB) |

---

## 🗂️ Project Structure

```
steel-defect-vision/
├── app.py                      # Streamlit web application
├── simulate.py                 # Conveyor belt simulation script
├── best.pt                     # Trained YOLOv8s weights
├── requirements.txt            # Python dependencies
├── packages.txt                # System-level apt dependencies
├── results.png                 # Training curves visualization
├── confusion_matrix.png        # Confusion matrix visualization
├── README.md                   # This file
└── .streamlit/
    └── config.toml             # Streamlit theme configuration
```

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Local Installation

```bash
# Clone the repository
git clone https://github.com/Ahsan-Neural/steel-defect-vision.git
cd steel-defect-vision

# Install Python dependencies
pip install -r requirements.txt

# Launch the web application
streamlit run app.py

# (Optional) Run conveyor belt simulation
python simulate.py --source data/samples --limit 20
```

---

## 📋 Dependencies

### Python Packages (`requirements.txt`)
```
streamlit
ultralytics
opencv-python-headless
pillow
numpy
pandas
```

### System Packages (`packages.txt`)
```
libgl1
libglx-mesa0
```

---

## 👤 Author

**Muhammad Ahsan** — BS Artificial Intelligence Student

- 📊 **Kaggle:** [ahsanneural](https://www.kaggle.com/ahsanneural)
- 🌐 **Live App:** [steel-defect-vision.streamlit.app](https://steel-defect-vision.streamlit.app/)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If you found this project helpful, please consider giving it a star!**

</div>
