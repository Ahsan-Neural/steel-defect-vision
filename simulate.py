import argparse
import os
import time
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

CLASSES = [
    "crazing", "inclusion", "patches", "pitted_surface",
    "rolled-in_scale", "scratches", "punching_hole", "welding_line",
    "crescent_gap", "water_spot", "oil_spot", "silk_spot",
    "rolled_pit", "crease", "waist_folding"
]


def load_model(weights_path: str = "best.pt") -> YOLO:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at '{weights_path}'")
    return YOLO(weights_path)


def iter_images(source_dir: str, limit=None):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    all_paths = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(exts):
                all_paths.append(os.path.join(root, f))
    all_paths.sort()
    if limit is not None:
        all_paths = all_paths[:limit]
    for p in all_paths:
        yield p


def run_simulation(
    source_dir,
    weights_path="best.pt",
    output_dir="outputs/simulated",
    conf=0.35,
    iou=0.45,
    limit=None,
):
    os.makedirs(output_dir, exist_ok=True)
    model = load_model(weights_path)

    total = rejected = passed = 0
    all_defect_classes = []

    print("\n" + "=" * 70)
    print("STEEL DEFECT VISION — CONVEYOR BELT SIMULATION")
    print("=" * 70)
    print(f"Source directory : {source_dir}")
    print(f"Model weights    : {weights_path}")
    print(f"Output directory : {output_dir}")
    print(f"Confidence       : {conf}")
    print(f"IoU (NMS)        : {iou}")
    print("=" * 70 + "\n")

    for idx, img_path in enumerate(iter_images(source_dir, limit=limit), start=1):
        total += 1
        img_name = os.path.basename(img_path)

        img    = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        t0 = time.time()
        results = model.predict(
            source=img_np,
            conf=conf,
            iou=iou,
            imgsz=640,
            verbose=False
        )
        elapsed = (time.time() - t0) * 1000

        result    = results[0]
        boxes     = result.boxes
        n_defects = len(boxes)

        defect_classes = [CLASSES[int(b.cls[0])] for b in boxes]
        all_defect_classes.extend(defect_classes)

        verdict = "REJECT" if n_defects > 0 else "PASS"
        if verdict == "REJECT":
            rejected += 1
        else:
            passed += 1

        ann_bgr = result.plot(line_width=2, font_size=12)
        out_name = f"{idx:04d}_{verdict.lower()}_{img_name}"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, ann_bgr)

        print(f"[Sheet #{idx:03d}] {img_name}")
        print(f"  Inference time : {elapsed:.1f} ms")
        print(f"  Detected       : {n_defects} defect(s)")
        if n_defects > 0:
            for j, (box, cls_name) in enumerate(zip(boxes, defect_classes), start=1):
                conf_j = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                print(f"    #{j:02d}  {cls_name:<15}  conf={conf_j:.2f}  box=[{x1}, {y1}, {x2}, {y2}]")
        else:
            print("    No defects detected.")
        print(f"  Verdict        : {verdict}")
        print("-" * 70)

    print("\n" + "=" * 70)
    print("FACTORY INSPECTION SUMMARY")
    print("=" * 70)
    print(f"Total sheets inspected : {total}")
    print(f"Passed (no defects)    : {passed}")
    print(f"Rejected (defects)     : {rejected}")
    if total > 0:
        print(f"Rejection rate         : {rejected / total * 100:.1f}%")

    if all_defect_classes:
        counts = Counter(all_defect_classes)
        top, freq = counts.most_common(1)[0]
        print(f"Most common defect     : {top} ({freq} occurrences)")
        print("\nDefect frequency table:")
        for cls_name, c in counts.most_common():
            print(f"  - {cls_name:<15}: {c}")
    else:
        print("No defects detected in this run.")

    print("=" * 70)
    print(f"\nAnnotated images saved to: {output_dir}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Conveyor-belt style steel defect detection simulation."
    )
    parser.add_argument("--source",  type=str,   default="data/samples", help="Folder containing input images")
    parser.add_argument("--weights", type=str,   default="best.pt",      help="Path to model weights")
    parser.add_argument("--output",  type=str,   default="outputs/simulated", help="Folder to save annotated images")
    parser.add_argument("--conf",    type=float, default=0.35,           help="Confidence threshold")
    parser.add_argument("--iou",     type=float, default=0.45,           help="IoU threshold for NMS")
    parser.add_argument("--limit",   type=int,   default=None,           help="Max number of images to process")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_simulation(
        source_dir=args.source,
        weights_path=args.weights,
        output_dir=args.output,
        conf=args.conf,
        iou=args.iou,
        limit=args.limit,
    )
