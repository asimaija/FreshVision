"""
FreshVision - YOLO boxes + Classifier + CLIP naming + Nutrition (exact CSV values)

YOLO finds WHERE the items are. A crop classifier (trained on your own data)
and CLIP together decide WHAT each one is. configs/nutrition.csv supplies
per-100g nutrition, shown EXACTLY as stored in the CSV (no scaling).

CHANGES in this version:
  * Works on a PC WITHOUT a GPU. Device is chosen automatically, and the slow
    settings (imgsz 1280, test-time augmentation, CLIP-large) are only used
    when a GPU is present.
  * Detector is chosen by TASK ("detect"), the classifier by TASK ("classify"),
    so a classifier can never be loaded as the detector.
  * You can pin exact weights with CLASSIFIER_PATHS / DETECTOR_PATH below.
  * If NO detector is found, the app still runs: "whole-photo mode" names the
    single main item, and a "grid scan" checkbox finds several kinds of produce
    by classifying overlapping tiles (approximate, no boxes per item).
  * The class list now includes every class the classifier knows (before,
    classes missing from the detector were silently dropped).
  * UNSUPPORTED_PRODUCE no longer lists radish / zucchini / eggplant
    (the classifier knows them now).
  * CLIP no longer vetoes a confident classifier: if the classifier is
    >= NAME_CLF_TRUST sure, its answer is used even when CLIP disagrees.

Run:
    pip install transformers
    streamlit run app.py
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

st.set_page_config(
    page_title="FreshVision",
    page_icon="🌿",
    layout="centered",
)

# ── Device ───────────────────────────────────────────────
HAS_GPU = torch.cuda.is_available()
DEVICE = "cuda" if HAS_GPU else "cpu"      # for CLIP (torch)
YOLO_DEVICE = 0 if HAS_GPU else "cpu"      # for ultralytics

# ── Model files ──────────────────────────────────────────
# Pin exact weights here (paths relative to this file). If a path does not
# exist, the app falls back to searching models/ by task.
# First existing + valid file in this list wins (a newly trained classifier goes in
# models/cls_new/best.pt via install_detector.bat; the second is your current one).
CLASSIFIER_PATHS = ["models/cls_new/best.pt",
                    "models/runs/runs/freshvision_cls_v26l_more-2/weights/best.pt"]
DETECTOR_PATH = None        # e.g. "models/det/best.pt"; None = search models/

# ── Fixed detection / naming settings (no UI controls) ──────────────────
DETECT_CONF = 0.25
DETECT_IOU = 0.35
DETECT_IMGSZ = 1280 if HAS_GPU else 640     # 1280 is far too slow on CPU
DETECT_MAX_BOXES = 100
DETECT_AUGMENT = HAS_GPU                    # test-time augmentation: GPU only

# CLIP-large is about 10x slower than CLIP-base on CPU.
CLIP_MODEL_ID = ("openai/clip-vit-large-patch14" if HAS_GPU
                 else "openai/clip-vit-base-patch32")
NAME_MIN_CONF = 0.40
NAME_MIN_MARGIN = 0.06     # below this, a label is shown with "?" but still kept
NAME_CROP_PAD = 0.12
NAME_CLASSIFIER_WEIGHT = 0.6   # 0 = CLIP only, 1 = classifier only
NAME_CLF_TRUST = 0.80      # classifier this sure beats a disagreeing CLIP
                           # (NOT used in grid-scan mode: there both must agree)

DEFAULT_EXTRAS = ("spice powder, chili flakes, seasoning, a plate, a bowl, a fork, "
                  "a leaf, a wooden table, a human hand, food packaging")

# Real produce that is NOT in the trained classes. CLIP can "claim" a box with
# one of these, so the app skips it (or tells you) instead of forcing a wrong
# label like okra -> green bean. Add more names here as you spot misses.
UNSUPPORTED_PRODUCE = ("okra, bitter gourd, bitter melon, cassava root, yuca, taro root, "
                       "yam, turnip, daikon, lotus root, "
                       "bottle gourd, ridge gourd, snake gourd, drumstick, "
                       "a root vegetable, a gourd")

# ── Look & feel ──────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
    --ink:        #20201C;
    --basil:      #25402C;
    --basil-dim:  #3D5C46;
    --parchment:  #F6EFE1;
    --parchment2: #EFE5CF;
    --tomato:     #C1432B;
    --mustard:    #E0A13B;
    --line:       #D8CBA8;
}
html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
.stApp { background: var(--parchment); color: var(--ink); }
h1, h2, h3, .fv-serif { font-family: 'Fraunces', serif; }
/* header crate label */
.fv-hero {
    background: var(--basil);
    color: var(--parchment);
    border-radius: 4px;
    padding: 2rem 2rem 1.6rem 2rem;
    margin-bottom: 1.6rem;
    border: 2px solid var(--basil);
    position: relative;
}
.fv-hero::before, .fv-hero::after {
    content: "";
    position: absolute; left: 14px; right: 14px; height: 1px;
    background: rgba(246,239,225,0.35);
}
.fv-hero::before { top: 10px; }
.fv-hero::after { bottom: 10px; }
.fv-hero h1 {
    font-size: 2.1rem; font-weight: 700; margin: 0 0 0.35rem 0; letter-spacing: 0.2px;
    color: var(--parchment);
}
.fv-hero p {
    margin: 0; font-size: 0.98rem; color: var(--parchment2); max-width: 46ch;
}
/* section titles */
.fv-section-title {
    font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.35rem;
    color: var(--basil); margin: 1.6rem 0 0.6rem 0; display: flex; align-items: baseline; gap: 0.5rem;
}
.fv-section-title .fv-tag {
    font-family: 'Inter', sans-serif; font-weight: 500; font-size: 0.85rem;
    color: var(--basil-dim); border-left: 2px solid var(--line); padding-left: 0.6rem;
}
/* receipt-style breakdown */
.fv-receipt {
    background: #fffdf8; border: 1px solid var(--line); border-radius: 3px;
    padding: 1rem 1.2rem; font-family: 'Inter', sans-serif;
}
.fv-receipt-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.45rem 0; border-bottom: 1px dashed var(--line); font-size: 0.95rem;
}
.fv-receipt-row:last-child { border-bottom: none; }
.fv-receipt-name { font-weight: 600; color: var(--ink); }
.fv-receipt-meta { color: var(--basil-dim); font-size: 0.85rem; }
.fv-receipt-count {
    font-family: 'Fraunces', serif; font-weight: 600; color: var(--tomato); font-size: 1.05rem;
}
/* per-item nutrition card (one per detected class) */
.fv-item-nutrition {
    background: #fffdf8; border: 1.5px solid var(--ink); border-radius: 3px;
    padding: 0.7rem 1rem; margin-bottom: 0.7rem;
}
.fv-item-nutrition .fv-in-head {
    display: flex; justify-content: space-between; align-items: baseline;
    border-bottom: 3px solid var(--ink); padding-bottom: 0.3rem; margin-bottom: 0.4rem;
}
.fv-item-nutrition .fv-in-name {
    font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.05rem; color: var(--ink);
}
.fv-item-nutrition .fv-in-count {
    font-size: 0.8rem; color: var(--basil-dim);
}
.fv-item-nutrition .fv-in-cal {
    font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.3rem; color: var(--tomato);
}
.fv-item-nutrition .fv-in-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.3rem 0.6rem; font-size: 0.85rem;
}
.fv-item-nutrition .fv-in-grid .val { color: var(--basil); font-weight: 600; }
/* uploader + button */
[data-testid="stFileUploaderDropzone"] {
    background: #fffdf8 !important; border: 2px dashed var(--basil-dim) !important; border-radius: 6px !important;
}
.stButton > button {
    background: var(--tomato); color: #fff; border: none; border-radius: 4px;
    font-weight: 600; padding: 0.55rem 1.2rem; font-family: 'Inter', sans-serif;
}
.stButton > button:hover { background: #a53622; color: #fff; }
.fv-uncertain-note { font-size: 0.82rem; color: var(--basil-dim); margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# Several prompt styles per label - averaged. More stable than one "a photo of X".
TEMPLATES = [
    "a photo of {}, a type of fruit or vegetable.",
    "a close-up photo of {}, fresh food.",
    "a photo of cooked or sliced {}.",
]


# ── Helpers ──────────────────────────────────────────────
def norm(s):
    """Lowercase + collapse whitespace/underscores, so 'Sweet Potato' /
    ' sweet potato ' / 'sweet_potato' all match the same key."""
    return re.sub(r"[\s_]+", " ", str(s).strip().lower())


def key_of(name):
    """Display name = first synonym of the dataset's own class name."""
    return norm(name.split("/")[0])


def folder_name(key):
    """Same rule as the crop-extraction folders, so classifier names map back to keys."""
    return re.sub(r"[^a-z0-9]+", "_", key).strip("_")


# Detector (LVIS) names that mean the same as one of your classifier classes.
NAME_ALIASES = {"tomatoes": "tomato", "capsicum": "bell pepper", "bell peppers": "bell pepper"}


def build_classes(det_names, cls_names):
    """Group classes by display name. Uses the detector's classes AND every
    class the classifier knows. A detector class whose name (or any synonym,
    e.g. 'tomatoes' / 'capsicum') matches a classifier class is merged into
    that class, so the detector, classifier and CLIP all speak the same names."""
    cls_keys = {key_of(n) for n in cls_names.values()}

    def canon(s):
        s = NAME_ALIASES.get(s, s)
        if s in cls_keys:
            return s
        for cut in (1, 2):                       # tomatoes -> tomato, apples -> apple
            if len(s) > cut + 2 and s[:-cut] in cls_keys and s.endswith(("s", "es")[cut - 1]):
                return s[:-cut]
        return None

    groups = {}
    for n in det_names.values():
        syn = [norm(x) for x in n.split("/") if x.strip()]
        k = next((c for c in (canon(x) for x in syn) if c), None) or key_of(n)
        groups.setdefault(k, set()).update(syn)
    for n in cls_names.values():
        k = key_of(n)
        groups.setdefault(k, set()).add(k)
    keys = sorted(groups)
    return keys, [sorted(groups[k]) for k in keys]


@st.cache_resource
def load_yolo(path):
    return YOLO(str(path))


def checkpoint_paths(root):
    """All best.pt / last.pt under `root`, newest file first, best.pt before
    last.pt. Files with any other name (best_old.pt, ...) are ignored."""
    if not root.exists():
        return []
    paths = [p for p in root.rglob("*.pt") if p.name in ("best.pt", "last.pt")]
    return sorted(paths, key=lambda p: (-p.stat().st_mtime, p.name != "best.pt"))


def find_model(paths, task):
    """Return (model, path) for the first checkpoint whose task matches."""
    for p in paths:
        try:
            m = load_yolo(p)
        except Exception:
            continue
        if getattr(m, "task", None) == task:
            return m, p
    return None, None


def resolve_model(explicit, task, paths):
    """Use a pinned path (str or list of str) if it exists and has the right
    task, else search by task."""
    if isinstance(explicit, str):
        explicit = [explicit]
    for e in explicit or []:
        p = PROJECT_ROOT / e
        if p.exists():
            try:
                m = load_yolo(p)
                if getattr(m, "task", None) == task:
                    return m, p
            except Exception:
                pass
    return find_model(paths, task)


def describe_checkpoints(paths):
    lines = []
    for p in paths:
        try:
            lines.append(f"- `{p.relative_to(PROJECT_ROOT)}` -> {load_yolo(p).task}")
        except Exception as e:
            lines.append(f"- `{p.relative_to(PROJECT_ROOT)}` -> ERROR: {e}")
    return lines or ["- (no best.pt / last.pt files found under models/)"]


@st.cache_resource
def load_clip(clip_id, device):
    model = CLIPModel.from_pretrained(clip_id).eval().to(device)
    return model, CLIPProcessor.from_pretrained(clip_id)


@st.cache_resource
def load_nutrition():
    """normalized class key -> {calories_kcal, protein_g, ...} per 100g,
    from configs/nutrition.csv."""
    path = PROJECT_ROOT / "configs" / "nutrition.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "class" not in df.columns:
        return {}
    df["class"] = df["class"].map(norm)
    df = df.drop_duplicates(subset="class", keep="first").set_index("class")
    return df.to_dict(orient="index")


def nutrition_for(name, nutrition_db):
    """Return the CSV row exactly as stored (per 100 g). No scaling, no rounding."""
    row = nutrition_db.get(norm(name))
    if row is None:
        return None
    est = {}
    for k, v in row.items():
        if k in ("matched_food", "fdc_id", "data_type"):
            continue
        try:
            fv = float(v)
            est[k] = fv if pd.notna(fv) else None
        except (TypeError, ValueError):
            est[k] = None
    return {"per_100g": row, "serving_g": 100, "estimated": est}


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
    Returns tensor (n_crops, n_classes + n_extras) on the CPU."""
    model, processor = load_clip(clip_id, DEVICE)
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

    inputs = processor(text=prompts, images=crops, return_tensors="pt",
                       padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image.cpu()        # (N, groups*T)
    logits = logits.view(logits.shape[0], -1, T).mean(-1)      # average templates
    idx = torch.tensor(groups).unsqueeze(0).expand(logits.shape[0], -1)
    per_class = torch.full((logits.shape[0], n_cls), float("-inf")).scatter_reduce(
        1, idx, logits, reduce="amax", include_self=False)
    return per_class.softmax(dim=-1)


def clf_probs(cls_model, crops, keys, cls_imgsz):
    """Classifier probabilities re-indexed to the app's `keys` order.
    Classes the classifier does not know stay at 0."""
    res = cls_model.predict(crops, imgsz=cls_imgsz, device=YOLO_DEVICE, verbose=False)
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
        color = (224, 161, 59) if it["uncertain"] else (37, 64, 44)   # mustard / basil
        d.rectangle([x1, y1, x2, y2], outline=color, width=max(2, img.width // 300))
        label = f'{it["name"]}{"?" if it["uncertain"] else ""} {it["conf"]:.0%}'
        d.text((x1 + 3, y1 + 2), label, fill=(255, 255, 255),
               font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return out


def grid_boxes(w, h, max_tiles=36):
    """Overlapping square tiles that cover the photo (used when there is no
    detector). Tile = ~30% of the short side, 50% overlap, at most max_tiles."""
    tile = min(max(64, int(0.30 * min(w, h))), w, h)
    stride = max(32, tile // 2)

    def starts(n):
        xs = list(range(0, n - tile + 1, stride)) or [0]
        if xs[-1] + tile < n:
            xs.append(n - tile)
        return xs

    xs, ys = starts(w), starts(h)
    while len(xs) * len(ys) > max_tiles:
        stride = int(stride * 1.25)
        xs, ys = starts(w), starts(h)
    return [[float(x), float(y), float(x + tile), float(y + tile)] for y in ys for x in xs]


def merge_same_label(items):
    """Merge overlapping tiles that got the same name into one region."""
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def overlap(a, b):
        return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i]["name"] == items[j]["name"] and overlap(items[i]["box"], items[j]["box"]):
                parent[find(i)] = find(j)
    groups = {}
    for i, it in enumerate(items):
        groups.setdefault(find(i), []).append(it)
    out = []
    for g in groups.values():
        bx = [x["box"] for x in g]
        out.append({**g[0],
                    "box": [min(b[0] for b in bx), min(b[1] for b in bx),
                            max(b[2] for b in bx), max(b[3] for b in bx)],
                    "conf": sum(x["conf"] for x in g) / len(g),
                    "uncertain": all(x["uncertain"] for x in g)})
    return out


# ── Model loading ────────────────────────────────────────
all_ckpts = checkpoint_paths(PROJECT_ROOT / "models")

yolo, det_path = resolve_model(DETECTOR_PATH, "detect", all_ckpts)
cls_model, cls_path = resolve_model(CLASSIFIER_PATHS, "classify", all_ckpts)

if yolo is None and cls_model is None:
    st.error("No usable weights found under models/. Checkpoints seen:\n\n"
             + "\n".join(describe_checkpoints(all_ckpts))
             + "\n\nPut a classifier best.pt in models/ (and a detector best.pt "
               "for multi-item photos).")
    st.stop()

cls_imgsz = 256
if cls_model is not None:
    try:
        cls_imgsz = int((getattr(cls_model, "ckpt", None) or {}).get("train_args", {}).get("imgsz", 256))
    except Exception:
        pass

keys, syns = build_classes(yolo.names if yolo is not None else {},
                           cls_model.names if cls_model is not None else {})

# Non-produce / unsupported labels CLIP may claim. A label that is also one of
# our real classes is removed from the lists, so it can never be both.
key_set = set(keys)
extras = [e.strip() for e in DEFAULT_EXTRAS.split(",") if e.strip()]
unsupported_set = {e.strip() for e in UNSUPPORTED_PRODUCE.split(",")
                   if e.strip() and norm(e) not in key_set}
extras = [e for e in extras if norm(e) not in key_set] + sorted(unsupported_set)

nutrition_db = load_nutrition()

# ── Main ─────────────────────────────────────────────────
st.markdown("""
<div class="fv-hero">
  <h1>🌿 FreshVision</h1>
  <p>Point it at your produce. It finds every item, names it, and shows its nutrition - like a receipt for what's actually in the photo.</p>
</div>
""", unsafe_allow_html=True)

if yolo is None:
    st.info("No detector installed, so this is not an error: the app names the main "
            "item in the photo. For a photo with several items, tick the grid-scan "
            "box below. A trained detector (models/det/best.pt) gives exact boxes.")
if not HAS_GPU:
    st.caption("Running on CPU (no GPU found) with fast settings.")
if not nutrition_db:
    st.info("Nutrition data isn't loaded yet - add configs/nutrition.csv to see values.")

file = st.file_uploader("Upload a photo of fruits or vegetables", type=["jpg", "jpeg", "png", "webp"])
if file is None:
    st.stop()

img = Image.open(file).convert("RGB")
st.image(img, caption="Uploaded", width="stretch")

grid_mode = False
if yolo is None:
    grid_mode = st.checkbox("This photo has several items (approximate grid scan)")

if st.button("Scan this photo", type="primary", width="stretch"):
    with st.spinner("Looking closely..."):
        if yolo is not None:
            res = yolo.predict(
                img,
                conf=DETECT_CONF,
                iou=DETECT_IOU,
                imgsz=DETECT_IMGSZ,
                agnostic_nms=True,
                augment=DETECT_AUGMENT,
                device=YOLO_DEVICE,
                verbose=False,
            )[0]

            # Guard: no boxes attribute (wrong task) or zero detections.
            if getattr(res, "boxes", None) is None or len(res.boxes) == 0:
                st.warning("Nothing was found in that photo. Try a clearer or closer shot.")
                st.stop()

            boxes = sorted(zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()),
                           key=lambda b: -b[1])[:DETECT_MAX_BOXES]
        else:
            if grid_mode:
                boxes = [(b, 1.0, -1) for b in grid_boxes(img.width, img.height)]
            else:
                # Whole-photo mode: one box covering the full image.
                boxes = [([0.0, 0.0, float(img.width), float(img.height)], 1.0, -1)]

        items = []
        unsupported_seen = set()
        kept, crops = [], []
        for xyxy, c, k in boxes:
            crop = crop_box(img, xyxy, NAME_CROP_PAD)
            if crop is not None:
                kept.append((xyxy, c, k))
                crops.append(crop)

        if crops:
            n_keys = len(keys)
            p_clip = clip_probs(CLIP_MODEL_ID, crops, syns, extras)
            p_clf = clf_probs(cls_model, crops, keys, cls_imgsz) if cls_model else None

            fused = p_clip * (1.0 - NAME_CLASSIFIER_WEIGHT if p_clf is not None else 1.0)
            if p_clf is not None:
                fused[:, :n_keys] += NAME_CLASSIFIER_WEIGHT * p_clf

            for n_i, (xyxy, c, k) in enumerate(kept):
                row = fused[n_i]
                top2 = row.topk(min(2, row.numel()))
                best = int(top2.indices[0])
                conf = float(top2.values[0])
                margin = conf - float(top2.values[1]) if row.numel() > 1 else conf

                # CLIP says it is not (supported) produce -> skip.
                if best >= n_keys:
                    label = extras[best - n_keys]
                    if label in unsupported_set:
                        unsupported_seen.add(label)
                    continue

                clip_idx = int(p_clip[n_i, :n_keys].argmax())
                clf_idx = None
                if p_clf is not None:
                    clf_idx = int(p_clf[n_i].argmax())
                    clf_conf = float(p_clf[n_i, clf_idx])
                    if clip_idx != clf_idx:
                        if clf_conf >= NAME_CLF_TRUST and not grid_mode:
                            # Confident trained classifier beats a disagreeing CLIP.
                            best, conf, margin = clf_idx, clf_conf, clf_conf
                        else:
                            continue   # they disagree and neither is sure: don't guess

                if conf < NAME_MIN_CONF:
                    continue

                items.append({
                    "box": xyxy, "name": keys[best], "conf": conf,
                    "uncertain": margin < NAME_MIN_MARGIN,
                    "yolo": key_of(yolo.names[int(k)]) if (yolo is not None and int(k) >= 0) else "-",
                    "clip": keys[clip_idx],
                    "clf": keys[clf_idx] if clf_idx is not None else "-",
                })

    if grid_mode and items:
        items = merge_same_label(items)
        st.caption("Grid scan is approximate: it shows which kinds of produce are "
                   "present. Counts are regions, not exact items.")

    if unsupported_seen:
        st.info("This looks like produce the app doesn't know yet: "
                + ", ".join(sorted(unsupported_seen))
                + ". It was skipped rather than given a wrong name.")

    if not items:
        st.warning("No known produce could be named in that photo. Try a clearer or closer shot.")
        st.stop()

    st.image(draw(img, items), caption="Detected", width="stretch")

    df = pd.DataFrame(items)
    counts = (df.groupby("name").agg(count=("name", "size"), avg_conf=("conf", "mean"))
                .sort_values("count", ascending=False).reset_index())

    # ── Breakdown, styled as a receipt ───────────────────
    st.markdown(f'<div class="fv-section-title">The tally <span class="fv-tag">{len(items)} items · {len(counts)} kinds</span></div>',
               unsafe_allow_html=True)
    rows_html = "".join(
        f'<div class="fv-receipt-row"><div><span class="fv-receipt-name">{r["name"].title()}</span> '
        f'<span class="fv-receipt-meta">· {r["avg_conf"]*100:.0f}% sure</span></div>'
        f'<div class="fv-receipt-count">×{int(r["count"])}</div></div>'
        for _, r in counts.iterrows()
    )
    st.markdown(f'<div class="fv-receipt">{rows_html}</div>', unsafe_allow_html=True)
    if df["uncertain"].any():
        st.markdown('<div class="fv-uncertain-note">Mustard outline in the photo above = not fully sure about that one.</div>',
                   unsafe_allow_html=True)

    with st.expander("What each model said (for debugging)"):
        st.caption(f"Detector: {det_path if det_path else 'none (whole-photo mode)'}")
        st.caption(f"Classifier: {cls_path if cls_path else 'none loaded'}")
        st.caption(f"Device: {DEVICE} | CLIP: {CLIP_MODEL_ID}")
        cmp = df[["name", "yolo", "clf", "clip", "conf"]].copy()
        cmp["conf"] = (cmp["conf"] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(cmp.rename(columns={"name": "Final", "yolo": "YOLO", "clf": "Classifier",
                                         "clip": "CLIP", "conf": "Conf"}),
                     width="stretch", hide_index=True)

    # ── Nutrition: exact values from configs/nutrition.csv (per 100 g) ─
    if nutrition_db:
        nut_rows, missing = [], []
        for name, cnt in counts[["name", "count"]].values:
            info = nutrition_for(name, nutrition_db)
            if info is None:
                missing.append(name)
                continue
            # one row per class (NOT multiplied by count)
            nut_rows.append({"Item": name, **info["estimated"]})

        st.markdown('<div class="fv-section-title">Nutrition <span class="fv-tag">per 100 g</span></div>',
                   unsafe_allow_html=True)

        if nut_rows:
            nut_df = pd.DataFrame(nut_rows)
            per_item = nut_df.groupby("Item").sum(numeric_only=True)
            item_counts = counts.set_index("name")["count"]

            for name, row in per_item.iterrows():
                n = int(item_counts.get(name, 1))
                st.markdown(f"""
                <div class="fv-item-nutrition">
                  <div class="fv-in-head">
                    <div>
                      <span class="fv-in-name">{name.title()}</span>
                      <span class="fv-in-count"> · per 100 g · {n} detected</span>
                    </div>
                    <span class="fv-in-cal">{row.get('calories_kcal', 0):.2f} kcal</span>
                  </div>
                  <div class="fv-in-grid">
                    <div>Protein <span class="val">{row.get('protein_g', 0):.2f} g</span></div>
                    <div>Carbs <span class="val">{row.get('carbs_g', 0):.2f} g</span></div>
                    <div>Fat <span class="val">{row.get('fat_g', 0):.2f} g</span></div>
                    <div>Fiber <span class="val">{row.get('fiber_g', 0):.2f} g</span></div>
                    <div>Sugar <span class="val">{row.get('sugar_g', 0):.2f} g</span></div>
                    <div>Sodium <span class="val">{row.get('sodium_mg', 0):.2f} mg</span></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            st.caption("Values are per 100 g, taken directly from configs/nutrition.csv.")
        else:
            st.info("None of the detected items were found in configs/nutrition.csv.")

        if missing:
            st.caption(f"No nutrition data yet for: {', '.join(m.title() for m in missing)}.")