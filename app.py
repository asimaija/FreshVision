"""
FreshVision — Streamlit App
Upload a fruit/vegetable image, click Detect, get back a table of detections
plus the annotated image.

Run:
    streamlit run streamlit_app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_ROOT / "models" / "runs" / "freshvision_yolo26s_v2" / "weights" / "best.pt"

# ── Inference settings ───────────────────────────────────
CONF_THRESHOLD = 0.35   # raised from 0.25 — fewer, more confident boxes
IOU_THRESHOLD = 0.30    # lowered from 0.45 — merges overlapping duplicate boxes more aggressively

st.set_page_config(page_title="FreshVision", page_icon="🥦", layout="centered")


@st.cache_resource
def load_model():
    if not WEIGHTS_PATH.exists():
        st.error(f"Weights not found at: {WEIGHTS_PATH}")
        st.stop()
    return YOLO(str(WEIGHTS_PATH))


model = load_model()

st.title("🥦 FreshVision")
st.caption("Upload a photo of fruits or vegetables, then click Detect")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
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
                agnostic_nms=True,   # suppress overlapping boxes across classes too, not just within-class
                save=False,
                verbose=False,
            )
            result = results[0]

            annotated_bgr = result.plot()
            annotated_rgb = annotated_bgr[:, :, ::-1]
            annotated_img = Image.fromarray(annotated_rgb)

        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append((model.names[cls_id], conf))

        detections.sort(key=lambda d: d[1], reverse=True)

        if not detections:
            st.warning(
                "⚠️ This doesn't look like a fruit or vegetable — "
                "no confident detections found. Try a clearer photo, "
                "or a closer shot of the item."
            )
        else:
            st.subheader("Detected")
            st.image(annotated_img, use_container_width=True)

            st.subheader(f"Detections ({len(detections)})")

            df = pd.DataFrame(detections, columns=["Item", "Confidence"])
            df["Item"] = df["Item"].str.title()
            df["Confidence"] = (df["Confidence"] * 100).round(1).astype(str) + "%"

            st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("Upload an image above to get started.")