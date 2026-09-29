"""
train_classifier.py - train a crop classifier on data/crops (made by extract_crops.py).

The detector finds boxes; this model names them. Much more accurate than
zero-shot CLIP because it learns YOUR classes from YOUR data.

    python scripts/extract_crops.py --pad 0.10
    python scripts/train_classifier.py

Output: models/cls/<RUN_NAME>/weights/best.pt  (the app picks it up automatically)
"""

from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
CROPS = ROOT / "data" / "crops"
RUNS_DIR = ROOT / "models" / "cls"

MODEL_CANDIDATES = ["yolo26m-cls.pt", "yolo11m-cls.pt"]   # first one that downloads wins
RUN_NAME = "freshvision_cls_v1"
EPOCHS = 40
IMGSZ = 224
DEVICE = 0 if torch.cuda.is_available() else "cpu"
BATCH = 64 if DEVICE == 0 else 16


def load_model():
    last = None
    for name in MODEL_CANDIDATES:
        try:
            return YOLO(name), name
        except Exception as e:      # weights name not available in this ultralytics version
            last = e
    raise RuntimeError(f"No classification weights could be loaded: {last}")


def main():
    if not (CROPS / "train").exists():
        raise FileNotFoundError(f"{CROPS / 'train'} not found - run scripts/extract_crops.py first")

    model, name = load_model()
    print(f"Model: {name} | device: {DEVICE} | crops: {CROPS}")

    model.train(
        data=str(CROPS),            # folder with train/ and val/ class sub-folders
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        patience=10,
        workers=0,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        # augmentation that helps real-world photos
        hsv_h=0.02, hsv_s=0.6, hsv_v=0.4,
        degrees=15, scale=0.4, fliplr=0.5,
        erasing=0.3,
        label_smoothing=0.1,
        cos_lr=True,
    )
    print(f"\nBest weights: {RUNS_DIR / RUN_NAME / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()