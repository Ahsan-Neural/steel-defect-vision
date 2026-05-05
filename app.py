import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time
import os
import pandas as pd
from collections import Counter

st.set_page_config(
    page_title="SteelGuard — Defect Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    [data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
    }

    .section-header {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8b949e;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid #21262d;
    }

    .verdict-reject {
        background-color: #3d1c1c;
        border: 1px solid #f85149;
        border-left: 4px solid #f85149;
        border-radius: 6px;
        padding: 16px 20px;
        font-size: 16px;
        font-weight: 600;
        color: #f85149;
    }

    .verdict-pass {
        background-color: #1c3d2a;
        border: 1px solid #3fb950;
        border-left: 4px solid #3fb950;
        border-radius: 6px;
        padding: 16px 20px;
        font-size: 16px;
        font-weight: 600;
        color: #3fb950;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        color: #8b949e;
        border-radius: 6px 6px 0 0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00b4d8;
        color: #000000;
        font-weight: 700;
    }

    h1, h2, h3 { color: #e6edf3; }
    p, li { color: #8b949e; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return YOLO("best.pt")


CLASSES = [
    "crazing", "inclusion", "patches", "pitted_surface",
    "rolled-in_scale", "scratches", "punching_hole", "welding_line",
    "crescent_gap", "water_spot", "oil_spot", "silk_spot",
    "rolled_pit", "crease", "waist_folding"
]

with st.sidebar:
    st.markdown("## SteelGuard")
    st.markdown('<p style="color:#8b949e; font-size:13px;">Industrial Surface Defect Detection</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-header">Inference Settings</div>', unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence Threshold", 0.10, 0.95, 0.35, 0.05)
    iou_threshold  = st.slider("IoU Threshold (NMS)", 0.10, 0.90, 0.45, 0.05)
    imgsz          = st.selectbox("Inference Resolution", [416, 640, 832], index=1)

    st.markdown("---")
    st.markdown('<div class="section-header">Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:13px; color:#8b949e; line-height:1.8">
    Architecture: YOLOv8s<br>
    Classes: 15 defect types<br>
    Training: NEU-DET + GC10-DET<br>
    Images: 3,960 total<br>
    mAP50: 0.7948<br>
    Inference: ~3.7 ms / image
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Defect Classes</div>', unsafe_allow_html=True)
    for c in CLASSES:
        st.markdown(f'<p style="font-size:12px; color:#8b949e; margin:2px 0">— {c}</p>', unsafe_allow_html=True)


st.markdown("# Steel Surface Defect Detection")
st.markdown('<p style="color:#8b949e;">Upload a steel surface image to detect and classify surface defects in real time.</p>', unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Single Image Inspection", "Batch Inspection", "Model Performance"])

# ── TAB 1 ─────────────────────────────────────────────────────
with tab1:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="section-header">Upload Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop a steel surface image here",
            type=["jpg", "jpeg", "png", "bmp"],
            key="single"
        )
        if uploaded:
            img_pil = Image.open(uploaded).convert("RGB")
            st.image(img_pil, caption="Original Image", use_container_width=True)
            st.markdown(
                f'<p style="font-size:12px; color:#8b949e;">Size: {img_pil.width} x {img_pil.height} px</p>',
                unsafe_allow_html=True
            )

    with col_result:
        st.markdown('<div class="section-header">Detection Results</div>', unsafe_allow_html=True)

        if uploaded:
            model = load_model()
            with st.spinner("Running inference..."):
                t0 = time.time()
                results = model.predict(
                    source=np.array(img_pil),
                    conf=conf_threshold,
                    iou=iou_threshold,
                    imgsz=imgsz,
                    verbose=False
                )
                elapsed = (time.time() - t0) * 1000

            result  = results[0]
            ann_rgb = result.plot(line_width=2, font_size=12)[:, :, ::-1]
            st.image(ann_rgb, caption="Detected Defects", use_container_width=True)

            boxes     = result.boxes
            n_defects = len(boxes)
            m1, m2, m3 = st.columns(3)
            m1.metric("Defects Found", n_defects)
            m2.metric("Inference Time", f"{elapsed:.1f} ms")
            m3.metric("Verdict", "REJECT" if n_defects > 0 else "PASS")

            if n_defects > 0:
                st.markdown('<div class="verdict-reject">REJECT — Defects detected on this sheet</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="verdict-pass">PASS — No defects detected</div>', unsafe_allow_html=True)

            if n_defects > 0:
                st.markdown('<div class="section-header" style="margin-top:20px">Defect Breakdown</div>', unsafe_allow_html=True)
                det_data = []
                for i, box in enumerate(boxes):
                    cls_id   = int(box.cls[0])
                    cls_name = CLASSES[cls_id]
                    conf_val = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    det_data.append({
                        "No.": i + 1,
                        "Class": cls_name,
                        "Confidence": f"{conf_val:.2%}",
                        "Bounding Box": f"[{x1}, {y1}, {x2}, {y2}]"
                    })
                st.dataframe(pd.DataFrame(det_data), use_container_width=True, hide_index=True)
        else:
            st.markdown(
                '<p style="color:#8b949e; margin-top:40px; text-align:center;">Upload an image to begin inspection.</p>',
                unsafe_allow_html=True
            )

# ── TAB 2 ─────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Batch Upload</div>', unsafe_allow_html=True)
    batch_files = st.file_uploader(
        "Upload multiple steel surface images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="batch"
    )

    if batch_files:
        model    = load_model()
        progress = st.progress(0)
        summary  = []
        cols     = st.columns(3)

        for idx, f in enumerate(batch_files):
            img = Image.open(f).convert("RGB")
            res = model.predict(
                source=np.array(img),
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=imgsz,
                verbose=False
            )[0]
            n       = len(res.boxes)
            ann     = res.plot(line_width=2)[:, :, ::-1]
            verdict = "REJECT" if n > 0 else "PASS"
            color   = "#f85149" if n > 0 else "#3fb950"

            with cols[idx % 3]:
                st.image(ann, use_container_width=True)
                st.markdown(
                    f'<p style="font-size:12px; color:{color}; text-align:center; font-weight:600;">{verdict} — {n} defect(s)</p>',
                    unsafe_allow_html=True
                )

            defect_classes = [CLASSES[int(b.cls[0])] for b in res.boxes]
            summary.append({
                "File": f.name,
                "Defects": n,
                "Verdict": verdict,
                "Classes Found": ", ".join(defect_classes) if defect_classes else "None"
            })
            progress.progress((idx + 1) / len(batch_files))

        st.markdown("---")
        st.markdown('<div class="section-header">Batch Summary Report</div>', unsafe_allow_html=True)

        total    = len(summary)
        rejected = sum(1 for s in summary if s["Verdict"] == "REJECT")
        passed   = total - rejected
        all_cls  = [c for s in summary for c in s["Classes Found"].split(", ") if c != "None"]
        top_cls  = Counter(all_cls).most_common(1)[0][0] if all_cls else "None"

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Total Inspected", total)
        r2.metric("Passed", passed)
        r3.metric("Rejected", rejected)
        r4.metric("Most Common Defect", top_cls)

        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

# ── TAB 3 ─────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Training Results</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("mAP50",     "0.7948")
    c2.metric("mAP50-95",  "0.4558")
    c3.metric("Precision", "0.7784")
    c4.metric("Recall",    "0.7365")

    st.markdown("---")

    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown('<div class="section-header">Training Curves</div>', unsafe_allow_html=True)
        if os.path.exists("results.png"):
            st.image("results.png", use_container_width=True)
        else:
            st.markdown('<p style="color:#8b949e;">results.png not found.</p>', unsafe_allow_html=True)

    with ic2:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_container_width=True)
        else:
            st.markdown('<p style="color:#8b949e;">confusion_matrix.png not found.</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Per-Class Performance</div>', unsafe_allow_html=True)

    class_perf = pd.DataFrame([
        {"Class": "punching_hole",   "mAP50": 0.975, "Precision": 0.979, "Recall": 0.984},
        {"Class": "scratches",       "mAP50": 0.970, "Precision": 0.885, "Recall": 0.968},
        {"Class": "patches",         "mAP50": 0.961, "Precision": 0.929, "Recall": 0.941},
        {"Class": "welding_line",    "mAP50": 0.932, "Precision": 0.844, "Recall": 0.921},
        {"Class": "crescent_gap",    "mAP50": 0.927, "Precision": 0.820, "Recall": 0.979},
        {"Class": "pitted_surface",  "mAP50": 0.898, "Precision": 0.899, "Recall": 0.790},
        {"Class": "inclusion",       "mAP50": 0.867, "Precision": 0.824, "Recall": 0.804},
        {"Class": "crease",          "mAP50": 0.838, "Precision": 0.858, "Recall": 0.753},
        {"Class": "waist_folding",   "mAP50": 0.828, "Precision": 0.733, "Recall": 0.500},
        {"Class": "rolled-in_scale", "mAP50": 0.804, "Precision": 0.787, "Recall": 0.695},
        {"Class": "water_spot",      "mAP50": 0.756, "Precision": 0.796, "Recall": 0.699},
        {"Class": "crazing",         "mAP50": 0.703, "Precision": 0.671, "Recall": 0.635},
        {"Class": "oil_spot",        "mAP50": 0.654, "Precision": 0.690, "Recall": 0.596},
        {"Class": "silk_spot",       "mAP50": 0.645, "Precision": 0.674, "Recall": 0.559},
        {"Class": "rolled_pit",      "mAP50": 0.161, "Precision": 0.288, "Recall": 0.223},
    ])
    st.dataframe(
        class_perf.style.background_gradient(subset=["mAP50"], cmap="RdYlGn"),
        use_container_width=True,
        hide_index=True
    )
