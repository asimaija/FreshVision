from pathlib import Path
from PIL import Image

DATASET_PATH = Path(
    r"D:\Projects\FreshVision\data\raw\LVIS_Fruits_And_Vegetables"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
NUM_CLASSES = 63
SPLITS = ["train", "val", "test"]


def list_stems(folder, extensions):
    if not folder.exists():
        print(f"[!] Folder does not exist: {folder}")
        return {}
    return {
        f.stem: f
        for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in extensions
    }


def validate_label_file(path):
    problems = []
    try:
        lines = path.read_text().strip().splitlines()
    except Exception as e:
        return [f"cannot read file: {e}"]

    if not lines:
        problems.append("empty label file")
        return problems

    for i, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) != 5:
            problems.append(f"line {i}: expected 5 values, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            problems.append(f"line {i}: non-numeric value")
            continue
        if not (0 <= cls < NUM_CLASSES):
            problems.append(f"line {i}: class id {cls} out of range [0,{NUM_CLASSES - 1}]")
        if not all(0.0 <= c <= 1.0 for c in coords):
            problems.append(f"line {i}: coordinates not normalized [0,1]")
    return problems


def validate_image_file(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return None
    except Exception as e:
        return str(e)


def validate_split(split):
    img_dir = DATASET_PATH / "images" / split
    lbl_dir = DATASET_PATH / "labels" / split

    images = list_stems(img_dir, IMAGE_EXTENSIONS)
    labels = list_stems(lbl_dir, {".txt"})

    img_stems = set(images)
    lbl_stems = set(labels)

    missing_labels = img_stems - lbl_stems
    orphan_labels = lbl_stems - img_stems

    label_problems = {}
    for stem, path in labels.items():
        probs = validate_label_file(path)
        if probs:
            label_problems[stem] = probs

    corrupt_images = {}
    for stem, path in images.items():
        err = validate_image_file(path)
        if err:
            corrupt_images[stem] = err

    return {
        "num_images": len(images),
        "num_labels": len(labels),
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
        "label_problems": label_problems,
        "corrupt_images": corrupt_images,
    }


def print_report(split, r):
    print("=" * 62)
    print(f"Split: {split}")
    print("=" * 62)
    print(f"Images: {r['num_images']:,}   Labels: {r['num_labels']:,}")

    if r["missing_labels"]:
        print(f"\n[!] {len(r['missing_labels'])} image(s) with NO label file:")
        for s in list(r["missing_labels"])[:10]:
            print(f"    - {s}")
        if len(r["missing_labels"]) > 10:
            print(f"    ... and {len(r['missing_labels']) - 10} more")

    if r["orphan_labels"]:
        print(f"\n[!] {len(r['orphan_labels'])} label file(s) with NO matching image:")
        for s in list(r["orphan_labels"])[:10]:
            print(f"    - {s}")

    if r["label_problems"]:
        print(f"\n[!] {len(r['label_problems'])} label file(s) with format problems:")
        for stem, probs in list(r["label_problems"].items())[:10]:
            print(f"    - {stem}: {probs[0]}" + (f" (+{len(probs)-1} more)" if len(probs) > 1 else ""))

    if r["corrupt_images"]:
        print(f"\n[!] {len(r['corrupt_images'])} corrupt/unreadable image(s):")
        for stem, err in list(r["corrupt_images"].items())[:10]:
            print(f"    - {stem}: {err}")

    if not any([r["missing_labels"], r["orphan_labels"], r["label_problems"], r["corrupt_images"]]):
        print("\n[OK] No issues found.")
    print()


def main():
    print(f"Dataset root: {DATASET_PATH}")
    print(f"Exists: {DATASET_PATH.exists()}\n")

    all_clean = True
    for split in SPLITS:
        r = validate_split(split)
        print_report(split, r)
        if any([r["missing_labels"], r["orphan_labels"], r["label_problems"], r["corrupt_images"]]):
            all_clean = False

    print("=" * 62)
    print("RESULT: dataset is clean, ready for training." if all_clean
          else "RESULT: issues found above - fix before training.")
    print("=" * 62)


if __name__ == "__main__":
    main()