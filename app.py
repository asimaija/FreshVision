"""
FreshVision — Streamlit App
Upload a fruit/vegetable image, click Detect, get back a clean grouped
summary of detections plus the annotated image.

Run:
    streamlit run streamlit_app.py
"""

from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_ROOT / "models" / "runs" / "freshvision_yolo26m_v4" / "weights" / "best.pt"

# ── Inference settings ───────────────────────────────────
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.30

st.set_page_config(page_title="FreshVision", page_icon="🥦", layout="centered")

# ── Styling ──────────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 760px; }

    .fv-header {
        text-align: center;
        margin-bottom: 1.6rem;
    }
    .fv-header h1 {
        font-size: 2rem;
        margin-bottom: 0.1rem;
        color: #1b5e20;
    }
    .fv-header p {
        color: #6b7a6c;
        font-size: 0.95rem;
        margin-top: 0;
    }

    div[data-testid="stMetric"] {
        background: #f3f8f0;
        border: 1px solid #dfe9db;
        border-radius: 12px;
        padding: 14px 10px;
        text-align: center;
    }
    div[data-testid="stMetricValue"] { color: #1b5e20; }

    .item-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #ffffff;
        border: 1px solid #e4e9e2;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .item-card .name {
        font-weight: 600;
        text-transform: capitalize;
        font-size: 0.95rem;
        color: #1b2b1c;
    }
    .item-card .meta {
        font-size: 0.8rem;
        color: #8a988b;
        margin-top: 2px;
    }
    .item-card .count-badge {
        background: #1b5e20;
        color: white;
        font-weight: 700;
        font-size: 0.85rem;
        min-width: 34px;
        text-align: center;
        border-radius: 999px;
        padding: 4px 10px;
    }
    .conf-bar-track {
        background: #eef2ec;
        border-radius: 999px;
        height: 6px;
        width: 100%;
        margin-top: 6px;
        overflow: hidden;
    }
    .conf-bar-fill {
        background: #4caf50;
        height: 100%;
        border-radius: 999px;
    }

    .stButton > button {
        border-radius: 999px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not WEIGHTS_PATH.exists():
        st.error(f"Weights not found at: {WEIGHTS_PATH}")
        st.stop()
    return YOLO(str(WEIGHTS_PATH))


model = load_model()

st.markdown("""
<div class="fv-header">
    <h1>🥦 FreshVision</h1>
    <p>Upload a photo of fruits or vegetables, then click Detect</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    detect_clicked = st.button("🔍 Detect", type="primary", use_container_width=True)

    if detect_clicked:
        with st.spinner("Running detection..."):
            results = model.predict(
                source=image,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                agnostic_nms=True,
                save=False,
                verbose=False,
            )
            result = results[0]

            annotated_bgr = result.plot()
            annotated_rgb = annotated_bgr[:, :, ::-1]
            annotated_img = Image.fromarray(annotated_rgb)

        raw_detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            raw_detections.append((model.names[cls_id], conf))

        if not raw_detections:
            st.warning(
                "⚠️ This doesn't look like a fruit or vegetable — "
                "no confident detections found. Try a clearer photo, "
                "or a closer shot of the item."
            )
        else:
            # ── Group by class: count + avg/max confidence ──
            grouped = defaultdict(list)
            for name, conf in raw_detections:
                grouped[name].append(conf)

            summary = []
            for name, confs in grouped.items():
                summary.append({
                    "name": name,
                    "count": len(confs),
                    "avg_conf": sum(confs) / len(confs),
                    "max_conf": max(confs),
                })
            summary.sort(key=lambda d: (d["count"], d["max_conf"]), reverse=True)

            st.markdown("### Detected")
            st.image(annotated_img, use_container_width=True)

            # ── Summary metrics ──
            total_items = len(raw_detections)
            unique_types = len(grouped)

            c1, c2 = st.columns(2)
            c1.metric("Total items", total_items)
            c2.metric("Unique types", unique_types)

            st.markdown("### Breakdown")

            for item in summary:
                pct = int(item["avg_conf"] * 100)
                st.markdown(f"""
                <div class="item-card">
                    <div style="flex:1;">
                        <div class="name">{item['name']}</div>
                        <div class="meta">avg {pct}% confidence &middot; best {int(item['max_conf']*100)}%</div>
                        <div class="conf-bar-track">
                            <div class="conf-bar-fill" style="width:{pct}%;"></div>
                        </div>
                    </div>
                    <div class="count-badge">×{item['count']}</div>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("Show raw detection list"):
                df = pd.DataFrame(raw_detections, columns=["Item", "Confidence"])
                df["Item"] = df["Item"].str.title()
                df["Confidence"] = (df["Confidence"] * 100).round(1).astype(str) + "%"
                df = df.sort_values("Confidence", ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("Upload an image above to get started.")