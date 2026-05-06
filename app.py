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

            # Log
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

                # Severity badges summary
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

# ─── TAB 2: BATCH ───────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Batch Upload</div>', unsafe_allow_html=True)
    batch_files = st.file_uploader(
        "Upload multiple steel images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="batch"
    )

    if batch_files:
        model     = load_model()
        progress  = st.progress(0, text="Starting batch inspection...")
        summary   = []
        cols      = st.columns(3)

        for idx, f in enumerate(batch_files):
            img = Image.open(f).convert("RGB")
            t0  = time.time()
            res = model.predict(
                source=np.array(img),
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=imgsz,
                verbose=False
            )[0]
            inf_ms = (time.time() - t0) * 1000

            n       = len(res.boxes)
            ann     = res.plot(line_width=2)[:, :, ::-1]
            verdict = "REJECT" if n > 0 else "PASS"
            color   = "#f85149" if n > 0 else "#3fb950"
            avg_c   = float(np.mean([float(b.conf[0]) for b in res.boxes])) if n > 0 else 0
            def_cls = [CLASSES[int(b.cls[0])] for b in res.boxes]

            with cols[idx % 3]:
                st.image(ann, use_container_width=True)
                st.markdown(
                    f'<p style="font-size:11px;color:{color};text-align:center;font-weight:700;margin-top:2px">'
                    f'{verdict} — {n} defect(s)</p>',
                    unsafe_allow_html=True
                )

            summary.append({
                "File":       f.name,
                "Defects":    n,
                "Verdict":    verdict,
                "Avg Conf":   f"{avg_c:.2%}" if n > 0 else "—",
                "Inf Time":   f"{inf_ms:.1f} ms",
                "Severity":   max((SEVERITY.get(c, "Low") for c in def_cls), key=["Low","Medium","High"].index, default="None") if def_cls else "None",
                "Classes":    ", ".join(def_cls) if def_cls else "None"
            })
            progress.progress(
                (idx + 1) / len(batch_files),
                text=f"Processing {idx+1}/{len(batch_files)}: {f.name}"
            )

        # Update session
        st.session_state.total_inspected += len(summary)
        batch_rejected = sum(1 for s in summary if s["Verdict"] == "REJECT")
        st.session_state.total_rejected  += batch_rejected

        st.markdown("---")
        st.markdown('<div class="section-header">Batch Summary</div>', unsafe_allow_html=True)

        total    = len(summary)
        rejected = sum(1 for s in summary if s["Verdict"] == "REJECT")
        passed   = total - rejected
        all_cls  = [c for s in summary for c in s["Classes"].split(", ") if c != "None"]
        top_cls  = Counter(all_cls).most_common(1)[0][0] if all_cls else "None"
        avg_time = np.mean([float(s["Inf Time"].replace(" ms","")) for s in summary])

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Total",       total)
        r2.metric("Passed",      passed)
        r3.metric("Rejected",    rejected)
        r4.metric("Top Defect",  top_cls)
        r5.metric("Avg Inf Time", f"{avg_time:.1f} ms")

        reject_rate = rejected / total * 100
        st.progress(int(reject_rate), text=f"Rejection Rate: {reject_rate:.1f}%")

        df_summary = pd.DataFrame(summary)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        st.download_button(
            "Download Batch Report (CSV)",
            data=df_to_csv(df_summary),
            file_name=f"batch_report_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.markdown("""
        <div class="info-card" style="text-align:center;padding:40px 20px">
            <p style="font-size:14px;color:#6e7681;margin:0">
            Upload multiple images to run batch inspection
            </p>
            <p style="font-size:12px;color:#484f58;margin-top:8px">
            All images are processed sequentially with the same inference settings
            </p>
        </div>
        """, unsafe_allow_html=True)

# ─── TAB 3: LOG ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Session Inspection Log</div>', unsafe_allow_html=True)

    if st.session_state.inspection_log:
        log_df = pd.DataFrame(st.session_state.inspection_log)

        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Total Inspected", len(log_df))
        l2.metric("Rejected",        int((log_df["Verdict"] == "REJECT").sum()))
        l3.metric("Passed",          int((log_df["Verdict"] == "PASS").sum()))
        rate = int((log_df["Verdict"] == "REJECT").sum()) / len(log_df) * 100
        l4.metric("Rejection Rate",  f"{rate:.1f}%")

        st.markdown("---")

        for entry in reversed(st.session_state.inspection_log[-20:]):
            css = "log-reject" if entry["Verdict"] == "REJECT" else "log-pass"
            color = "#f85149" if entry["Verdict"] == "REJECT" else "#3fb950"
            st.markdown(
                f'<div class="log-entry {css}">'
                f'<span style="color:#6e7681">[{entry["Time"]}]</span> '
                f'<b style="color:#c9d1d9">{entry["File"]}</b> — '
                f'<span style="color:{color};font-weight:700">{entry["Verdict"]}</span> '
                f'· {entry["Defects"]} defect(s) · {entry["Classes"]} · {entry["Inf Time"]}'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.dataframe(log_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download Full Session Log (CSV)",
            data=df_to_csv(log_df),
            file_name=f"session_log_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.markdown("""
        <div class="info-card" style="text-align:center;padding:40px 20px">
            <p style="font-size:14px;color:#6e7681;margin:0">
            No inspections recorded yet in this session
            </p>
            <p style="font-size:12px;color:#484f58;margin-top:8px">
            Use Single or Batch inspection tabs to start logging results
            </p>
        </div>
        """, unsafe_allow_html=True)

# ─── TAB 4: PERFORMANCE ─────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Model Performance Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("mAP50",     "0.7948")
    c2.metric("mAP50-95",  "0.4558")
    c3.metric("Precision", "0.7784")
    c4.metric("Recall",    "0.7365")
    c5.metric("Inference", "~3.7 ms")

    st.markdown("---")

    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown('<div class="section-header">Training Curves</div>', unsafe_allow_html=True)
        if os.path.exists("results.png"):
            st.image("results.png", use_container_width=True)
        else:
            st.markdown('<div class="info-card"><p style="color:#6e7681;font-size:13px;margin:0">Place results.png in repo root to display training curves.</p></div>', unsafe_allow_html=True)

    with ic2:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_container_width=True)
        else:
            st.markdown('<div class="info-card"><p style="color:#6e7681;font-size:13px;margin:0">Place confusion_matrix.png in repo root to display matrix.</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Per-Class Performance</div>', unsafe_allow_html=True)

    class_perf = pd.DataFrame([
        {"Class": "punching_hole",  "mAP50": 0.975, "Precision": 0.979, "Recall": 0.984, "Severity": "High"},
        {"Class": "scratches",      "mAP50": 0.970, "Precision": 0.885, "Recall": 0.968, "Severity": "Medium"},
        {"Class": "patches",        "mAP50": 0.961, "Precision": 0.929, "Recall": 0.941, "Severity": "Medium"},
        {"Class": "welding_line",   "mAP50": 0.932, "Precision": 0.844, "Recall": 0.921, "Severity": "Medium"},
        {"Class": "crescent_gap",   "mAP50": 0.927, "Precision": 0.820, "Recall": 0.979, "Severity": "High"},
        {"Class": "pitted_surface", "mAP50": 0.898, "Precision": 0.899, "Recall": 0.790, "Severity": "High"},
        {"Class": "inclusion",      "mAP50": 0.867, "Precision": 0.824, "Recall": 0.804, "Severity": "High"},
        {"Class": "crease",         "mAP50": 0.838, "Precision": 0.858, "Recall": 0.753, "Severity": "Medium"},
        {"Class": "waist_folding",  "mAP50": 0.828, "Precision": 0.733, "Recall": 0.500, "Severity": "Medium"},
        {"Class": "rolled-in_scale","mAP50": 0.804, "Precision": 0.787, "Recall": 0.695, "Severity": "High"},
        {"Class": "water_spot",     "mAP50": 0.756, "Precision": 0.796, "Recall": 0.699, "Severity": "Low"},
        {"Class": "crazing",        "mAP50": 0.703, "Precision": 0.671, "Recall": 0.635, "Severity": "High"},
        {"Class": "oil_spot",       "mAP50": 0.654, "Precision": 0.690, "Recall": 0.596, "Severity": "Low"},
        {"Class": "silk_spot",      "mAP50": 0.645, "Precision": 0.674, "Recall": 0.559, "Severity": "Low"},
        {"Class": "rolled_pit",     "mAP50": 0.161, "Precision": 0.288, "Recall": 0.223, "Severity": "Medium"},
    ])

    st.dataframe(
        class_perf.style.background_gradient(subset=["mAP50", "Precision", "Recall"], cmap="RdYlGn"),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    pa1, pa2 = st.columns(2)

    with pa1:
        st.markdown('<div class="section-header">Defects by Severity</div>', unsafe_allow_html=True)
        sev_counts = class_perf["Severity"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]
        st.dataframe(sev_counts, use_container_width=True, hide_index=True)

    with pa2:
        st.markdown('<div class="section-header">Top 5 Best Detected Classes</div>', unsafe_allow_html=True)
        top5 = class_perf.nlargest(5, "mAP50")[["Class", "mAP50", "Precision"]]
        st.dataframe(top5, use_container_width=True, hide_index=True)
