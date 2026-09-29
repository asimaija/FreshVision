"""
FreshVision - YOLO boxes + Classifier + CLIP names

YOLO finds WHERE the items are. A crop classifier (trained on your own data,
scripts/train_classifier.py) and CLIP together decide WHAT each one is.

  - weights    : models/runs/*/weights/best.pt      (detector, newest by default)
  - classifier : models/cls/*/weights/best.pt       (optional, newest by default)
  - imgsz      : read from the checkpoints' own training args
  - class names + synonyms : read from the detector itself

Run:
    pip install transformers
    streamlit run streamlit_app_clip.py
"""

import re
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="FreshVision + CLIP", page_icon="🥦", layout="centered")

# Several prompt styles per label -> averaged. More stable than one "a photo of X".
TEMPLATES = [
    "a photo of {}, a type of fruit or vegetable.",
    "a close-up photo of {}, fresh food.",
    "a photo of cooked or sliced {}.",
]

# Things that are NOT produce. If CLIP thinks a crop is one of these, the box is dropped.
DEFAULT_EXTRAS = ("spice powder, chili flakes, seasoning, a plate, a bowl, a fork, "
                  "a leaf, a wooden table, a human hand, food packaging")


# ── Helpers ──────────────────────────────────────────────
def key_of(name):
    """Display name = first synonym of the dataset's own class name."""
    return name.split("/")[0].strip().lower()


def folder_name(key):
    """Same rule as scripts/extract_crops.py, so classifier names map back to keys."""
    return re.sub(r"[^a-z0-9]+", "_", key).strip("_")


def build_classes(names):
    """Group dataset classes by display name (this also merges Tomato/tomato)."""
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


def clip_probs(clip_id, crops, syns, extras):
    """Probability of every class (+ every non-produce label) for every crop.
    Each label is scored with several prompt templates (averaged); a class takes
    the best of its synonyms. Returns tensor (n_crops, n_classes + n_extras)."""
    model, processor = load_clip(clip_id)
    T = len(TEMPLATES)
    prompts, groups = [], []
    for i, ss in enumerate(syns):
        for s in ss:
            prompts += [t.format(s) for t in TEMPLATES]
            groups.append(i)
    for j, e in enumerate(extras):
        prompts += [f"a photo of {e}"] * T
        groups.append(len(syns) + j)
    n_cls = len(syns) + len(extras)

    inputs = processor(text=prompts, images=crops, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image              # (N, groups*T)
    logits = logits.view(logits.shape[0], -1, T).mean(-1)      # average templates
    idx = torch.tensor(groups).unsqueeze(0).expand(logits.shape[0], -1)
    per_class = torch.full((logits.shape[0], n_cls), float("-inf")).scatter_reduce(
        1, idx, logits, reduce="amax", include_self=False)
    return per_class.softmax(dim=-1)


def clf_probs(cls_model, crops, keys, cls_imgsz):
    """Classifier probabilities re-indexed to the app's `keys` order.
    Classes the classifier does not know stay at 0."""
    res = cls_model.predict(crops, imgsz=cls_imgsz, verbose=False)
    out = torch.zeros(len(crops), len(keys))
    pos = {folder_name(k): i for i, k in enumerate(keys)}
    for r_i, r in enumerate(res):
        p = r.probs.data.cpu()
        for c_i, name in cls_model.names.items():
            k = pos.get(folder_name(name))
            if k is not None:
                out[r_i, k] = p[int(c_i)]
    return out


def draw(img, items):
    out = img.copy()
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", max(14, img.width // 50))
    except OSError:
        font = ImageFont.load_default()
    for it in items:
        x1, y1, x2, y2 = it["box"]
        color = (230, 160, 30) if it["uncertain"] else (30, 160, 60)
        d.rectangle([x1, y1, x2, y2], outline=color, width=max(2, img.width // 300))
        label = f'{it["name"]}{"?" if it["uncertain"] else ""} {it["conf"]:.0%}'
        d.text((x1 + 3, y1 + 2), label, fill=(255, 255, 255),
               font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return out


# ── Sidebar ──────────────────────────────────────────────
weights = sorted(PROJECT_ROOT.glob("models/runs/*/weights/best.pt"),
                 key=lambda p: p.stat().st_mtime, reverse=True)
if not weights:
    st.error("models/runs/*/weights/best.pt nahi mila.")
    st.stop()
cls_weights = sorted(PROJECT_ROOT.glob("models/cls/*/weights/best.pt"),
                     key=lambda p: p.stat().st_mtime, reverse=True)

st.sidebar.header("Settings")
weights_path = st.sidebar.selectbox("Model", weights, format_func=lambda p: p.parent.parent.name)
yolo = load_yolo(weights_path)
train_imgsz = int((getattr(yolo, "ckpt", None) or {}).get("train_args", {}).get("imgsz", 640))

mode = st.sidebar.radio("Naming", ["YOLO + Classifier + CLIP (naya)", "YOLO only (purana)"])
conf_thr = st.sidebar.slider("Box confidence", 0.05, 0.90, 0.25, 0.05,
                             help="Kam rakho: zyada items milenge, naming filter CLIP/classifier karega")
imgsz = st.sidebar.select_slider("Image size", [320, 480, 640, 800, 960, 1280], value=min(
    [320, 480, 640, 800, 960, 1280], key=lambda v: abs(v - train_imgsz)),
    help=f"Model {train_imgsz} par train hua tha")
iou = st.sidebar.slider("NMS IoU", 0.10, 0.90, 0.30, 0.05)
max_boxes = st.sidebar.slider("Max boxes", 5, 100, 40, 5)

use_naming = mode.startswith("YOLO + ")
cls_model, cls_imgsz, w_clf = None, 224, 0.0
if use_naming:
    clip_id = st.sidebar.text_input("CLIP model", "openai/clip-vit-large-patch14",
                                    help="Tez chahiye to: openai/clip-vit-base-patch32")
    if cls_weights:
        cls_path = st.sidebar.selectbox("Classifier", cls_weights,
                                        format_func=lambda p: p.parent.parent.name)
        cls_model = load_yolo(cls_path)
        cls_imgsz = int((getattr(cls_model, "ckpt", None) or {}).get("train_args", {}).get("imgsz", 224))
        w_clf = st.sidebar.slider("Classifier weight", 0.0, 1.0, 0.6, 0.05,
                                  help="0 = sirf CLIP, 1 = sirf classifier")
    else:
        st.sidebar.info("Classifier nahi mila (models/cls). scripts/train_classifier.py chalao. Abhi sirf CLIP.")
    min_conf = st.sidebar.slider("Min confidence", 0.0, 0.9, 0.30, 0.05)
    min_margin = st.sidebar.slider("Min margin (top1 - top2)", 0.0, 0.5, 0.08, 0.02,
                                   help="Isse kam farq ho to label par '?' aata hai (uncertain)")
    pad = st.sidebar.slider("Crop padding", 0.0, 0.5, 0.10, 0.05,
                            help="Classifier train karte waqt extract_crops.py --pad se match rakho")
    extras_text = st.sidebar.text_area("Non-produce labels (comma se)", DEFAULT_EXTRAS,
                                       help="Jo cheezein reject karni hon (masale, plate, fork...)")
    extras = [e.strip() for e in extras_text.split(",") if e.strip()]

keys, syns = build_classes(yolo.names)

# ── Main ─────────────────────────────────────────────────
st.title("🥦 FreshVision")
st.caption("YOLO boxes dhoondta hai, classifier + CLIP naam batate hain")

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
        if use_naming:
            kept, crops = [], []
            for xyxy, c, k in boxes:
                crop = crop_box(img, xyxy, pad)
                if crop is not None:
                    kept.append((xyxy, c, k))
                    crops.append(crop)

            if crops:
                n_keys = len(keys)
                p_clip = clip_probs(clip_id, crops, syns, extras)              # (N, keys+extras)
                p_clf = clf_probs(cls_model, crops, keys, cls_imgsz) if cls_model else None

                # Blend: everything from CLIP scaled by (1-w), classifier mass added on top.
                fused = p_clip * (1.0 - w_clf)
                if p_clf is not None:
                    fused[:, :n_keys] += w_clf * p_clf

                for n_i, (xyxy, c, k) in enumerate(kept):
                    row = fused[n_i]
                    top2 = row.topk(min(2, row.numel()))
                    best = int(top2.indices[0])
                    conf = float(top2.values[0])
                    margin = conf - float(top2.values[1]) if row.numel() > 1 else conf
                    if best >= n_keys or conf < min_conf:
                        continue   # non-produce label won, or not sure enough
                    items.append({
                        "box": xyxy, "name": keys[best], "conf": conf,
                        "uncertain": margin < min_margin,
                        "yolo": key_of(yolo.names[int(k)]),
                        "clip": keys[int(p_clip[n_i, :n_keys].argmax())],
                        "clf": keys[int(p_clf[n_i].argmax())] if p_clf is not None else "-",
                    })
        else:
            for xyxy, c, k in boxes:
                n = key_of(yolo.names[int(k)])
                items.append({"box": xyxy, "name": n, "conf": c, "uncertain": False,
                              "yolo": n, "clip": "-", "clf": "-"})

    if not items:
        st.warning("Koi fruit/vegetable nahi mila. Sidebar mein confidence kam karke dekho.")
        st.stop()

    st.image(draw(img, items), caption="Detected (orange = uncertain)", width="stretch")

    df = pd.DataFrame(items)
    counts = (df.groupby("name").agg(count=("name", "size"), avg_conf=("conf", "mean"))
                .sort_values("count", ascending=False).reset_index())
    counts["avg_conf"] = (counts["avg_conf"] * 100).round(0).astype(int).astype(str) + "%"
    st.subheader(f"Breakdown ({len(items)} items, {len(counts)} types)")
    st.dataframe(counts.rename(columns={"name": "Item", "count": "Count", "avg_conf": "Avg conf"}),
                 width="stretch", hide_index=True)

    with st.expander("Kis model ne kya kaha tha (comparison)"):
        cmp = df[["name", "yolo", "clf", "clip", "conf"]].copy()
        cmp["conf"] = (cmp["conf"] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(cmp.rename(columns={"name": "Final", "yolo": "YOLO", "clf": "Classifier",
                                         "clip": "CLIP", "conf": "Conf"}),
                     width="stretch", hide_index=True)