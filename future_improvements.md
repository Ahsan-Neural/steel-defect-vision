# Future Improvements – Steel Defect Vision

This document tracks concrete, next-step upgrades planned for **Steel Defect Vision**, focusing on model robustness, real-world usability, and production readiness.

## 1. Model & Training

- Train separate variants for:
  - High-precision (factory QA) vs. high-recall (early defect screening).
- Experiment with larger YOLOv8 variants (m / l) and compare:
  - mAP@50 and mAP@50-95 against current YOLOv8s.
  - Inference speed trade-offs on CPU and GPU.
- Add more targeted augmentation for:
  - Reflections, motion blur, and low-light conditions.
  - Different steel textures and background noise.
- Implement proper experiment tracking:
  - Save configs, seeds, and training logs for each run.

## 2. Real-World Robustness

- Build a small “real-world” test set from:
  - Phone camera images from factory-like environments.
  - Mixed lighting, angle, and distance variations.
- Evaluate domain gap:
  - Compare metrics on NEU-DET / GC10-DET vs. real-world images.
- Add an optional pre-processing pipeline:
  - Contrast normalization, glare reduction, or denoising.
- Document recommended settings for:
  - Different camera setups and conveyor speeds.

## 3. App UX & Features

- Add a **defect summary report** section:
  - Per-class counts, total defects per image/batch.
  - Simple CSV export of detections.
- Add **example test images** in the repo:
  - One image per key defect type with expected outputs.
- Provide a “factory mode” preset:
  - Predefined confidence / IoU / resolution settings.
- Improve error handling:
  - Clear messages for unsupported file types and corrupted images.

## 4. Monitoring & Analytics

- Log anonymized usage stats locally (optional):
  - Number of images processed, average inference time.
- Add performance dashboard:
  - Show moving average of inference speed.
  - Basic hardware info and environment summary.
- Create a simple benchmarking script:
  - Run a fixed batch of images and report timings.

## 5. Deployment & Scaling

- Package the app as a Docker image:
  - One command deployment for local server / edge device.
- Explore alternative deployment targets:
  - Cloud VM, on-premise server, or industrial PC.
- Add **GPU vs. CPU** deployment notes:
  - Recommended hardware, expected throughput, and latency.
- Document environment variables:
  - For paths, thresholds, and feature toggles.

## 6. Documentation & Examples

- Add a “Failure Cases” section with:
  - Images where the model missed or misclassified defects.
  - Short analysis of why and how to improve.
- Add a “Use Cases” section:
  - Inline inspection, final QA, offline analysis from stored images.
- Provide a short “Integration Guide”:
  - How a factory could integrate this with an existing pipeline.
- Record a short walkthrough GIF or screenshots:
  - Upload → detection → summary flow.

## 7. Codebase Quality

- Refactor `app.py`:
  - Separate UI code, model loading, and inference into modules.
- Add type hints and docstrings for key functions.
- Add a minimal test suite:
  - Smoke tests for model loading and single-image inference.
- Simplify configuration:
  - Centralize thresholds, paths, and model options in one config file.
