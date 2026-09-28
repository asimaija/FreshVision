import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "LVIS_Fruits_And_Vegetables"

TRAIN_IMAGES = DATASET_PATH / "images" / "train"
TRAIN_LABELS = DATASET_PATH / "labels" / "train"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def add_negatives(source_folder):
    source_folder = Path(source_folder)
    if not source_folder.exists():
        print(f"[!] Source folder not found: {source_folder}")
        return

    TRAIN_IMAGES.mkdir(parents=True, exist_ok=True)
    TRAIN_LABELS.mkdir(parents=True, exist_ok=True)

    images = [
        f for f in source_folder.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        print(f"[!] No images found in {source_folder}")
        return

    added = 0
    for img_path in images:
        dest_name = f"negative_{img_path.stem}{img_path.suffix}"
        dest_img_path = TRAIN_IMAGES / dest_name
        dest_label_path = TRAIN_LABELS / f"negative_{img_path.stem}.txt"

        if dest_img_path.exists():
            print(f"    Skipping (already exists): {dest_name}")
            continue

        shutil.copy2(img_path, dest_img_path)
        dest_label_path.write_text("")

        added += 1

    print(f"\nAdded {added} negative samples to train split.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python add_negative_samples.py <path_to_negative_images_folder>")
        sys.exit(1)

    add_negatives(sys.argv[1])