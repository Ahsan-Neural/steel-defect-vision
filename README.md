<div align="center">

# 🔩 Steel Defect Vision

**Real-time steel surface defect detection powered by YOLOv8**

Detects 15 types of industrial steel surface defects using a custom-trained YOLOv8s model, deployed as an interactive web application.



</div>

***

## 🌐 Live Demo

👉 **[https://steel-defect-vision.streamlit.app/](https://steel-defect-vision.streamlit.app/)**

Upload any steel surface image and get instant defect detection results with bounding boxes, confidence scores, and defect classification — no setup required.

***

##  Model Performance

| Metric | Value |
|---|---|
| **mAP@50** | 0.7948 |
| **mAP@50-95** | 0.4558 |
| **Precision** | 0.7784 |
| **Recall** | 0.7365 |
| **Training Time** | 1.126 hours |
| **Inference Speed** | ~3.7 ms / image |




***

##  What This Project Does

Steel manufacturing plants lose millions annually due to surface defects that slip through manual quality inspection. This project automates that process — a computer vision model scans steel surface images and instantly flags defects with bounding boxes, class labels, and confidence scores.

**Key features of the app:**
- **Single Image Inspection** — upload one image, get annotated results instantly
- **Batch Inspection** — upload multiple images, get a full defect summary report
- **Model Performance Tab** — view mAP, precision-recall curves, and confusion matrix
- **Adjustable thresholds** — tune confidence, IoU, and inference resolution for your use case

> 💡 **Tip for real-world images:** For glossy or reflective steel photos, lower the confidence threshold to 0.15–0.25 and set resolution to 832 for best results. The model was trained on controlled industrial dataset images, so domain gap may reduce confidence scores on real-world photos.

***

## 🏷️ Defect Classes (15)

| ID | Class | Source Dataset |
|---|---|---|
| 0 | crazing | NEU-DET |
| 1 | inclusion | NEU-DET + GC10-DET |
| 2 | patches | NEU-DET |
| 3 | pitted_surface | NEU-DET |
| 4 | rolled-in_scale | NEU-DET |
| 5 | scratches | NEU-DET |
| 6 | punching_hole | GC10-DET |
| 7 | welding_line | GC10-DET |
| 8 | crescent_gap | GC10-DET |
| 9 | water_spot | GC10-DET |
| 10 | oil_spot | GC10-DET |
| 11 | silk_spot | GC10-DET |
| 12 | rolled_pit | GC10-DET |
| 13 | crease | GC10-DET |
| 14 | waist_folding | GC10-DET |

***

## 📦 Dataset

| Dataset | Classes | Images |
|---|---|---|
| [NEU-DET](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) | 6 | 1,799 |
| [GC10-DET](https://www.kaggle.com/datasets/alex000kim/gc10-det) | 10 | 2,161 |
| **Combined** | **15** | **3,960** |

- **Train / Val split:** 80% / 20%
- **Annotation format:** Pascal VOC XML → converted to YOLO TXT
- **Training notebook:** [View on Kaggle](https://www.kaggle.com/ahsanneural)

***

##  Model Details

| Parameter | Value |
|---|---|
| Architecture | YOLOv8s |
| Framework | Ultralytics 8.4.46 |
| Input size | 640 × 640 |
| Epochs | 50 |
| Hardware | Tesla T4 GPU (Kaggle) |
| Weights file | `best.pt` (22.5 MB) |

***

## 🗂️ Project Structure

```
steel-defect-vision/
├── app.py                  # Streamlit web application
├── simulate.py             # Conveyor belt simulation script
├── best.pt                 # Trained YOLOv8s weights
├── requirements.txt        # Python dependencies
├── packages.txt            # System-level apt dependencies
├── results.png             # Training curves
├── confusion_matrix.png    # Confusion matrix
└── .streamlit/
    └── config.toml         # Streamlit theme config
```

***

##  Run Locally

```bash
git clone https://github.com/Ahsan-Neural/steel-defect-vision.git
cd steel-defect-vision

pip install -r requirements.txt

# Launch web app
streamlit run app.py

# Run conveyor belt simulation
python simulate.py --source data/samples --limit 20
```

***

## 📋 Requirements

```
streamlit
ultralytics
opencv-python-headless
pillow
numpy
pandas
```

**System packages (`packages.txt`):**
```
libgl1
libglx-mesa0
```

***

## 👤 Author

**Muhammad Ahsan** — BS Artificial Intelligence Student

- 📊 Kaggle: [ahsanneural](https://www.kaggle.com/ahsanneural)
- 🌐 Live App: [steel-defect-vision.streamlit.app](https://steel-defect-vision.streamlit.app/)

***

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
