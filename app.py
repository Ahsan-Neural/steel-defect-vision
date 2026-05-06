import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time
import os
import pandas as pd
from collections import Counter
import io
import base64

st.set_page_config(
    page_title="SteelGuard — Defect Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }

    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    [data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px 18px;
    }

    .section-header {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #6e7681;
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid #21262d;
    }

    .verdict-reject {
        background: linear-gradient(135deg, #2d1117 0%, #3d1c1c 100%);
        border: 1px solid #f85149;
        border-radius: 8px;
        padding: 14px 20px;
        font-size: 15px;
        font-weight: 700;
        color: #f85149;
        letter-spacing: 0.04em;
        margin-top: 8px;
    }

    .verdict-pass {
        background: linear-gradient(135deg, #0d2117 0%, #1c3d2a 100%);
        border: 1px solid #3fb950;
        border-radius: 8px;
        padding: 14px 20px;
        font-size: 15px;
        font-weight: 700;
        color: #3fb950;
        letter-spacing: 0.04em;
        margin-top: 8px;
    }

    .defect-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin: 2px;
    }

    .badge-high { background-color: #3d1c1c; color: #f85149; border: 1px solid #f85149; }
    .badge-mid  { background-color: #2d2416; color: #e3b341; border: 1px solid #e3b341; }
    .badge-low  { background-color: #1c2d2d; color: #58a6ff; border: 1px solid #58a6ff; }

    .info-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }

    .tip-card {
        background-color: #1a2332;
        border: 1px solid #388bfd;
        border-left: 4px solid #388bfd;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }

    .log-entry {
        background-color: #161b22;
        border-left: 3px solid #30363d;
        padding: 8px 12px;
        margin-bottom: 6px;
        border-radius: 0 6px 6px 0;
        font-size: 13px;
        color: #8b949e;
    }

    .log-reject { border-left-color: #f85149; }
    .log-pass   { border-left-color: #3fb950; }

    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        color: #8b949e;
        border-radius: 6px 6px 0 0;
        font-weight: 500;
        font-size: 13px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f2937;
        color: #f0f6fc;
        font-weight: 700;
        border-bottom: 2px solid #58a6ff;
    }

    h1, h2, h3 { color: #e6edf3; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not os.path.exists("best.pt"):
        st.error("Model file 'best.pt' not found.")
        st.stop()
    return YOLO("best.pt")


CLASSES = [
    "crazing", "inclusion", "patches", "pitted_surface",
    "rolled-in_scale", "scratches", "punching_hole", "welding_line",
    "crescent_gap", "water_spot", "oil_spot", "silk_spot",
    "rolled_pit", "crease", "waist_folding"
]

SEVERITY = {
    "crazing": "High", "inclusion": "High", "pitted_surface": "High",
    "rolled-in_scale": "High", "punching_hole": "High", "crescent_gap": "High",
    "scratches": "Medium", "patches": "Medium", "welding_line": "Medium",
    "crease": "Medium", "waist_folding": "Medium", "rolled_pit": "Medium",
    "water_spot": "Low", "oil_spot": "Low", "silk_spot": "Low"
}

SEVERITY_COLOR = {"High": "badge-high", "Medium": "badge-mid", "Low": "badge-low"}

if "inspection_log" not in st.session_state:
    st.session_state.inspection_log = []
if "total_inspected" not in st.session_state:
    st.session_state.total_inspected = 0
if "total_rejected" not in st.session_state:
    st.session_state.total_rejected = 0


def get_severity_badge(cls_name):
    sev = SEVERITY.get(cls_name, "Low")
    css = SEVERITY_COLOR[sev]
    return f'<span class="defect-badge {css}">{sev}</span>'


def df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### SteelGuard")
    st.markdown('<p style="color:#6e7681;font-size:12px;margin-top:-8px;">Industrial AI Inspection System</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-header">Inference Settings</div>', unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence Threshold", 0.10, 0.95, 0.35, 0.05)
    iou_threshold  = st.slider("IoU Threshold (NMS)",  0.10, 0.90, 0.45, 0.05)
    imgsz          = st.selectbox("Inference Resolution", [416, 640, 832], index=1)

    st.markdown("""
    <div class="tip-card">
        <p style="font-size:11px;color:#388bfd;font-weight:700;margin:0 0 4px 0">💡 Real-World Image Tip</p>
        <p style="font-size:11px;color:#8b949e;margin:0;line-height:1.6">
        For glossy or reflective steel photos, lower <b style="color:#c9d1d9">Confidence</b> to 0.15–0.25
        and set <b style="color:#c9d1d9">Resolution</b> to 832 for best results.
        Domain gap may reduce confidence scores on real-world photos.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Session Stats</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    s1.metric("Inspected", st.session_state.total_inspected)
    s2.metric("Rejected",  st.session_state.total_rejected)

    if st.session_state.total_inspected > 0:
        rate = st.session_state.total_rejected / st.session_state.total_inspected * 100
        st.progress(int(rate), text=f"Rejection Rate: {rate:.1f}%")

    if st.button("Reset Session", use_container_width=True):
        st.session_state.inspection_log   = []
        st.session_state.total_inspected  = 0
        st.session_state.total_rejected   = 0
        st.rerun()

    st.markdown("---")
    st.markdown('<div class="section-header">Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
    <p style="font-size:12px;color:#8b949e;line-height:2;margin:0">
    <b style="color:#c9d1d9">Architecture</b> YOLOv8s<br>
    <b style="color:#c9d1d9">Classes</b> 15 defect types<br>
    <b style="color:#c9d1d9">Datasets</b> NEU-DET + GC10-DET<br>
    <b style="color:#c9d1d9">Images</b> 3,960 total<br>
    <b style="color:#c9d1d9">mAP50</b> 0.7948<br>
    <b style="color:#c9d1d9">Precision</b> 0.7784<br>
    <b style="color:#c9d1d9">Recall</b> 0.7365<br>
    <b style="color:#c9d1d9">Inference</b> ~3.7 ms / image
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Defect Severity Guide</div>', unsafe_allow_html=True)
    for cls in CLASSES:
        sev = SEVERITY[cls]
        badge = SEVERITY_COLOR[sev]
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin:3px 0">'
            f'<span style="font-size:12px;color:#8b949e">{cls}</span>'
            f'<span class="defect-badge {badge}">{sev}</span></div>',
            unsafe_allow_html=True
        )

# ─── HEADER ─────────────────────────────────────────────────────────────────
st.markdown("## Steel Surface Defect Detection")
st.markdown('<p style="color:#6e7681;margin-top:-10px;font-size:14px;">AI-powered quality control for industrial steel inspection</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "Single Inspection", "Batch Inspection",
    "Inspection Log", "Model Performance"
])

# ─── TAB 1: SINGLE ──────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-header">Upload Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop a steel surface image",
            type=["jpg", "jpeg", "png", "bmp"],
            key="single"
        )
        if uploaded:
            img_pil = Image.open(uploaded).convert("RGB")
            st.image(img_pil, caption="Original", use_container_width=True)

            col_meta1, col_meta2, col_meta3 = st.columns(3)
            col_meta1.metric("Width",  f"{img_pil.width}px")
            col_meta2.metric("Height", f"{img_pil.height}px")
            col_meta3.metric("Format", uploaded.type.split("/")[-1].upper())

    with col_right:
        st.markdown('<div class="section-header">Detection Result</div>', unsafe_allow_html=True)

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
            st.image(ann_rgb, caption="Annotated Output", use_container_width=True)

            boxes     = result.boxes
            n_defects = len(boxes)
            verdict   = "REJECT" if n_defects > 0 else "PASS"

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Defects",    n_defects)
            m2.metric("Time",       f"{elapsed:.1f} ms")
            m3.metric("Verdict",    verdict)
            avg_conf = float(np.mean([float(b.conf[0]) for b in boxes])) if n_defects > 0 else 0
            m4.metric("Avg Conf",   f"{avg_conf:.0%}" if n_defects > 0 else "—")

            if n_defects > 0:
                st.markdown('<div class="verdict-reject">⛔ REJECT — Surface defects detected</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="verdict-pass">✅ PASS — No defects detected</div>', unsafe_allow_html=True)

            ts = time.strftime("%H:%M:%S")
            st.session_state.total_inspected += 1
            if n_defects > 0:
                st.session_state.total_rejected += 1
            defect_cls_list = [CLASSES[int(b.cls[0])] for b in boxes]
            st.session_state.inspection_log.append({
                "Time": ts,
                "File": uploaded.name,
                "Defects": n_defects,
                "Verdict": verdict,
                "Classes": ", ".join(defect_cls_list) if defect_cls_list else "None",
                "Avg Conf": f"{avg_conf:.2%}" if n_defects > 0 else "—",
                "Inf Time": f"{elapsed:.1f} ms"
            })

            if n_defects > 0:
                st.markdown("---")
                st.markdown('<div class="section-header">Defect Breakdown</div>', unsafe_allow_html=True)

                det_data = []
                for i, box in enumerate(boxes):
                    cls_id   = int(box.cls[0])
                    cls_name = CLASSES[cls_id]
                    conf_val = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w = x2 - x1
                    h = y2 - y1

                    det_data.append({
                        "#":          i + 1,
                        "Class":      cls_name,
                        "Severity":   SEVERITY.get(cls_name, "Low"),
                        "Confidence": f"{conf_val:.2%}",
                        "Size (px)":  f"{w}×{h}",
                        "Location":   f"({x1},{y1})"
                    })

                df_det = pd.DataFrame(det_data)
                st.dataframe(df_det, use_container_width=True, hide_index=True)

                badge_html = "".join(
                    get_severity_badge(CLASSES[int(b.cls[0])]) for b in boxes
                )
                st.markdown(f'<div style="margin-top:8px">{badge_html}</div>', unsafe_allow_html=True)

                csv_bytes = df_to_csv(df_det)
                st.download_button(
                    "Download Report (CSV)",
                    data=csv_bytes,
                    file_name=f"report_{uploaded.name}_{ts}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.markdown("""
            <div class="info-card" style="margin-top:12px;text-align:center;padding:40px 20px">
                <p style="font-size:14px;color:#6e7681;margin:0">
                Upload a steel surface image to begin inspection
                </p>
                <p style="font-size:12px;color:#484f58;margin-top:8px">
                Supports JPG, PNG, BMP
                </p>
            </div>
            """, unsafe_allow_html=True)

# rest of your code stays unchanged...
