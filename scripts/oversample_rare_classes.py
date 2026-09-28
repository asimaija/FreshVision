from pathlib import Path
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "LVIS_Fruits_And_Vegetables"

TRAIN_IMAGES = DATASET_PATH / "images" / "train"
TRAIN_LABELS = DATASET_PATH / "labels" / "train"

# Class IDs to oversample, and how many extra copies to add per image
# that contains them. Tune based on your nt_per_class imbalance.
OVERSAMPLE_CLASSES = {
    37: 3,   # lemon
    40: 3,   # mandarin
    39: 3,   # lime
    18: 3,   # cherry
    16: 5,   # cayenne
    22: 5,   # coconut
    19: 6,   # chickpea/garbanzo
}


def image_contains_class(label_path, class_id):
    if not label_path.exists():
        return False
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if parts and int(parts[0]) == class_id:
            return True
    return False


def oversample():
    all_labels = list(TRAIN_LABELS.glob("*.txt"))
    total_added = 0

    for class_id, multiplier in OVERSAMPLE_CLASSES.items():
        matching_labels = [
            lp for lp in all_labels
            if image_contains_class(lp, class_id) and not lp.stem.startswith("oversample_")
        ]

        print(f"Class {class_id}: found {len(matching_labels)} images, "
              f"adding {multiplier - 1} extra copies each")

        for label_path in matching_labels:
            stem = label_path.stem
            img_path = None
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                candidate = TRAIN_IMAGES / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                continue

            for i in range(1, multiplier):
                new_stem = f"oversample_{class_id}_{i}_{stem}"
                new_img_path = TRAIN_IMAGES / f"{new_stem}{img_path.suffix}"
                new_label_path = TRAIN_LABELS / f"{new_stem}.txt"

                if new_img_path.exists():
                    continue

                img = Image.open(img_path)
                flipped = ImageOps.mirror(img)
                flipped.save(new_img_path)

                lines = label_path.read_text().strip().splitlines()
                new_lines = []
                for line in lines:
                    parts = line.split()
                    cls, cx, cy, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    cx = 1.0 - cx
                    new_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                new_label_path.write_text("\n".join(new_lines) + "\n")

                total_added += 1

    print(f"\nDone. Added {total_added} oversampled images to train split.")


if __name__ == "__main__":
    oversample()