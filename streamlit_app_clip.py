"""
FreshVision - YOLO boxes + CLIP names (experimental, nothing hardcoded)

YOLO finds WHERE the items are. CLIP decides WHAT each one is.
Everything is derived at runtime:
  - weights   : picked from models/runs/*/weights/best.pt (newest by default)
  - imgsz     : read from the checkpoint's own training args
  - class names + synonyms : read from the model itself (no manual mapping)
  - thresholds: sidebar sliders

Run:
    pip install transformers
    streamlit run streamlit_app_clip.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="FreshVision + CLIP", page_icon="🥦", layout="centered")


# ── Helpers (all generic) ────────────────────────────────
def key_of(name):
    """Display name = first synonym of the dataset's own class name."""
    return name.split("/")[0].strip().lower()


def build_classes(names):
    """Group dataset classes by display name (this also merges Tomato/tomato).
    Returns display names and, per class, every synonym found in the dataset name."""
    groups = {}
    for n in names.values():
        groups.setdefault(key_of(n), set()).update(s.strip().lower() for s in n.split("/") if s.strip())
    keys = sorted(groups)
    return keys, [sorted(groups[k]) for k in keys]


@st.cache_resource
def load_yolo(path):
    return YOLO(str(path))


@st.cache_resource
def load_clip(clip_id):
    return CLIPModel.from_pretrained(clip_id).eval(), CLIPProcessor.from_pretrained(clip_id)


def crop_box(img, xyxy, pad):
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    box = tuple(int(v) for v in (max(0, x1 - w * pad), max(0, y1 - h * pad),
                                 min(img.width, x2 + w * pad), min(img.height, y2 + h * pad)))
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        return None
    return img.crop(box)


def clip_classify(clip_id, crops, syns, extras):
    """Score each crop against every synonym of every class (max over synonyms),
    plus optional user-supplied 'not produce' labels. Returns (class_index, prob)."""
    model, processor = load_clip(clip_id)
    prompts, cls_idx = [], []
    for i, ss in enumerate(syns):
        for s in ss:
            prompts.append(f"a photo of {s}")
            cls_idx.append(i)
    for j, e in enumerate(extras):
        prompts.append(f"a photo of {e}")
        cls_idx.append(len(syns) + j)
    n_cls = len(syns) + len(extras)

    inputs = processor(text=prompts, images=crops, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image
    idx = torch.tensor(cls_idx).unsqueeze(0).expand(logits.shape[0], -1)
    per_class = torch.full((logits.shape[0], n_cls), float("-inf")).scatter_reduce(
        1, idx, logits, reduce="amax", include_self=False)
    conf, best = per_class.softmax(dim=-1).max(dim=-1)
    return [(int(b), float(c)) for b, c in zip(best, conf)]


def draw(img, items):
    out = img.copy()
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", max(14, img.width // 50))
    except OSError:
        font = ImageFont.load_default()
    for it in items:
        x1, y1, x2, y2 = it["box"]
        d.rectangle([x1, y1, x2, y2], outline=(30, 160, 60), width=max(2, img.width // 300))
        d.text((x1 + 3, y1 + 2), f'{it["name"]} {it["conf"]:.0%}', fill=(255, 255, 255),
               font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return out


# ── Sidebar: every setting comes from here or from the model ──
weights = sorted(PROJECT_ROOT.glob("models/runs/*/weights/best.pt"),
                 key=lambda p: p.stat().st_mtime, reverse=True)
if not weights:
    st.error("models/runs/*/weights/best.pt nahi mila.")
    st.stop()

st.sidebar.header("Settings")
weights_path = st.sidebar.selectbox("Model", weights, format_func=lambda p: p.parent.parent.name)
yolo = load_yolo(weights_path)
train_imgsz = int((getattr(yolo, "ckpt", None) or {}).get("train_args", {}).get("imgsz", 640))

mode = st.sidebar.radio("Naming", ["YOLO + CLIP (naya)", "YOLO only (purana)"])
conf_thr = st.sidebar.slider("Box confidence", 0.05, 0.90, 0.30, 0.05)
imgsz = st.sidebar.select_slider("Image size", [320, 480, 640, 800, 960, 1280], value=min(
    [320, 480, 640, 800, 960, 1280], key=lambda v: abs(v - train_imgsz)),
    help=f"Model {train_imgsz} par train hua tha")
iou = st.sidebar.slider("NMS IoU", 0.10, 0.90, 0.30, 0.05)
max_boxes = st.sidebar.slider("Max boxes", 5, 100, 40, 5)
if mode.startswith("YOLO + CLIP"):
    clip_id = st.sidebar.text_input("CLIP model", "openai/clip-vit-base-patch32")
    min_clip = st.sidebar.slider("Min CLIP confidence", 0.0, 0.9, 0.30, 0.05)
    pad = st.sidebar.slider("Crop padding", 0.0, 0.5, 0.10, 0.05)
    extras_text = st.sidebar.text_input(
        "Non-produce labels (comma se, optional)", "",
        help="Jo cheezein reject karni hon, jaise: a human hand, cooked food")
    extras = [e.strip() for e in extras_text.split(",") if e.strip()]

keys, syns = build_classes(yolo.names)

# ── Main ─────────────────────────────────────────────────
st.title("🥦 FreshVision")
st.caption("YOLO boxes dhoondta hai, CLIP naam batata hai")

file = st.file_uploader("Image chuno", type=["jpg", "jpeg", "png", "webp"])
if file is None:
    st.info("Upload an image to get started.")
    st.stop()

img = Image.open(file).convert("RGB")
st.image(img, caption="Uploaded", width="stretch")

if st.button("🔍 Detect", type="primary", width="stretch"):
    with st.spinner("Detecting..."):
        res = yolo.predict(img, conf=conf_thr, iou=iou, imgsz=imgsz, agnostic_nms=True, verbose=False)[0]
        boxes = sorted(zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()),
                       key=lambda b: -b[1])[:max_boxes]

        items = []
        if mode.startswith("YOLO + CLIP"):
            kept, crops = [], []
            for xyxy, c, k in boxes:
                crop = crop_box(img, xyxy, pad)
                if crop is not None:
                    kept.append((xyxy, c, k))
                    crops.append(crop)
            preds = clip_classify(clip_id, crops, syns, extras) if crops else []
            for (xyxy, c, k), (ci, cc) in zip(kept, preds):
                if ci >= len(keys) or cc < min_clip:
                    continue   # non-produce label won, or CLIP unsure
                items.append({"box": xyxy, "name": keys[ci], "conf": cc,
                              "yolo": key_of(yolo.names[int(k)])})
        else:
            for xyxy, c, k in boxes:
                n = key_of(yolo.names[int(k)])
                items.append({"box": xyxy, "name": n, "conf": c, "yolo": n})

    if not items:
        st.warning("Koi fruit/vegetable nahi mila. Sidebar mein confidence kam karke dekho.")
        st.stop()

    st.image(draw(img, items), caption="Detected", width="stretch")

    df = pd.DataFrame(items)
    counts = (df.groupby("name").agg(count=("name", "size"), avg_conf=("conf", "mean"))
                .sort_values("count", ascending=False).reset_index())
    counts["avg_conf"] = (counts["avg_conf"] * 100).round(0).astype(int).astype(str) + "%"
    st.subheader(f"Breakdown ({len(items)} items, {len(counts)} types)")
    st.dataframe(counts.rename(columns={"name": "Item", "count": "Count", "avg_conf": "Avg conf"}),
                 width="stretch", hide_index=True)

    with st.expander("YOLO ne kya kaha tha (comparison)"):
        cmp = df[["name", "yolo", "conf"]].copy()
        cmp["conf"] = (cmp["conf"] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(cmp.rename(columns={"name": "Final", "yolo": "YOLO", "conf": "Conf"}),
                     width="stretch", hide_index=True)