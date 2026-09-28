"""
extract_crops.py - cut classifier training crops out of the YOLO-format dataset.

The detector finds fruit boxes well but names them badly. A classifier trained
on the SAME boxes (cropped out) fixes the naming without any new labeling.

Selection is diversity-first: each class takes one crop per source image, then a
second crop per image, and so on - so 800 crops come from as many DIFFERENT
photos as possible instead of from a few crowded ones. The report shows how many
distinct images each class really has (that, not the box count, is the true size
of the data).

    python scripts/extract_crops.py
    python scripts/extract_crops.py --cap 1000 --min-images 20 --min-size 24

Output (Ultralytics classify format):
    data/crops/train/<class>/*.jpg
    data/crops/val/<class>/*.jpg
"""

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def key_of(name):
    """Class key = first synonym; also merges Tomato/tomato style duplicates."""
    return name.split("/")[0].strip().lower()


def folder_name(key):
    return re.sub(r"[^a-z0-9]+", "_", key).strip("_")


def label_dir_for(img_dir):
    parts = list(img_dir.parts)
    i = max(i for i, p in enumerate(parts) if p == "images")
    parts[i] = "labels"
    return Path(*parts)


def collect(img_dir, id2key, per_image, rng):
    """Label files -> {class: [(image, cx, cy, w, h), ...]} ordered diversity-first:
    every image's 1st box, then every image's 2nd box, ... (no image is read here)."""
    lbl_dir = label_dir_for(img_dir)
    images = {p.stem: p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT}
    by_class = defaultdict(lambda: defaultdict(list))            # class -> image -> boxes
    for lf in lbl_dir.rglob("*.txt"):
        img = images.get(lf.stem)
        if img is None:
            continue
        for line in lf.read_text().splitlines():
            p = line.split()
            if len(p) == 5 and int(p[0]) in id2key:
                by_class[id2key[int(p[0])]][img].append((img, *map(float, p[1:])))

    cands = {}
    for k, per_img in by_class.items():
        ranked = []
        for rows in per_img.values():
            rng.shuffle(rows)
            for rank, row in enumerate(rows[:per_image] if per_image else rows):
                ranked.append((rank, rng.random(), row))
        ranked.sort(key=lambda t: (t[0], t[1]))
        cands[k] = [t[2] for t in ranked]
    return cands


def crop_split(cands, keys, cap, min_size, pad, out_dir):
    """Pick up to `cap` crops per class in diversity order (size filter applied),
    then open each source image once and save its crops."""
    sizes, selected = {}, defaultdict(list)
    counts, sources = defaultdict(int), defaultdict(set)
    for k in keys:
        for img, cx, cy, w, h in cands.get(k, []):
            if counts[k] >= cap:
                break
            if img not in sizes:
                try:
                    with Image.open(img) as im:
                        sizes[img] = im.size                       # header only
                except Exception:
                    sizes[img] = None
            if sizes[img] is None:
                continue
            W, H = sizes[img]
            bw, bh = w * W, h * H
            x1, y1 = max(0, cx * W - bw / 2 - bw * pad), max(0, cy * H - bh / 2 - bh * pad)
            x2, y2 = min(W, cx * W + bw / 2 + bw * pad), min(H, cy * H + bh / 2 + bh * pad)
            if min(x2 - x1, y2 - y1) < min_size:
                continue
            selected[img].append((k, (int(x1), int(y1), int(x2), int(y2))))
            counts[k] += 1
            sources[k].add(img)

    for img_path, boxes in selected.items():
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            for n, (k, box) in enumerate(boxes):
                d = out_dir / folder_name(k)
                d.mkdir(parents=True, exist_ok=True)
                im.crop(box).save(d / f"{img_path.stem[:40]}_{n}.jpg", quality=95)
    return counts, {k: len(v) for k, v in sources.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "configs" / "dataset.yaml"))
    ap.add_argument("--out", default=str(ROOT / "data" / "crops"))
    ap.add_argument("--cap", type=int, default=800, help="max train crops per class")
    ap.add_argument("--val-cap", type=int, default=200, help="max val crops per class")
    ap.add_argument("--per-image", type=int, default=0,
                    help="hard limit of crops of one class from one image (0 = none)")
    ap.add_argument("--min-size", type=int, default=32, help="skip crops with a side below this (px)")
    ap.add_argument("--min-instances", type=int, default=100, help="skip classes with fewer train boxes")
    ap.add_argument("--min-images", type=int, default=30, help="skip classes seen in fewer distinct train images")
    ap.add_argument("--pad", type=float, default=0.05, help="context added around each box")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    cfg = yaml.safe_load(Path(a.data).read_text())
    root = Path(cfg["path"])
    id2key = {int(i): key_of(n) for i, n in cfg["names"].items()}
    all_keys = set(id2key.values())

    train_c = collect(root / cfg["train"], id2key, a.per_image, rng)
    stats = {k: (len(train_c.get(k, [])), len({r[0] for r in train_c.get(k, [])})) for k in all_keys}
    keys = sorted(k for k in all_keys if stats[k][0] >= a.min_instances and stats[k][1] >= a.min_images)
    skipped = sorted((stats[k][1], stats[k][0], k) for k in all_keys if k not in keys)

    out = Path(a.out)
    tr, tr_imgs = crop_split(train_c, keys, a.cap, a.min_size, a.pad, out / "train")
    va, _ = crop_split(collect(root / cfg["val"], id2key, a.per_image, rng),
                       keys, a.val_cap, a.min_size, a.pad, out / "val")

    print(f"\n{'class':<26}{'train crops':>12}{'from images':>13}{'val crops':>11}")
    print("-" * 62)
    for k in sorted(keys, key=lambda k: tr[k]):
        weak = tr[k] < a.min_instances or tr_imgs.get(k, 0) < a.min_images
        print(f"{k[:25]:<26}{tr[k]:>12}{tr_imgs.get(k, 0):>13}{va[k]:>11}{'  <-- weak' if weak else ''}")
    print(f"\nkept {len(keys)} classes -> {out}")
    print("note: a class with 0 crops gets no folder, so the classifier will not know it.")
    if skipped:
        print(f"\nskipped (need >= {a.min_instances} boxes AND >= {a.min_images} distinct images):")
        print(f"  {'class':<26}{'images':>8}{'boxes':>8}")
        for n_img, n_box, k in skipped:
            print(f"  {k:<26}{n_img:>8}{n_box:>8}")


if __name__ == "__main__":
    main()