"""
train_classifier.py - train a crop classifier on data/crops (made by extract_crops.py).

Matches the Colab run "freshvision_cls_l_v2" exactly, so results are reproducible
either place. On this machine (no GPU) it will fall back to CPU and be extremely
slow at these settings - see the note printed at startup.

    python scripts/extract_crops.py --pad 0.10
    python scripts/train_classifier.py

Output: models/cls/<RUN_NAME>/weights/best.pt
"""

from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
CROPS = ROOT / "data" / "crops"
RUNS_DIR = ROOT / "models" / "cls"

MODEL = "yolo26l-cls.pt"
RUN_NAME = "freshvision_cls_l_v2"
EPOCHS = 100
IMGSZ = 288
PATIENCE = 20

DEVICE = 0 if torch.cuda.is_available() else "cpu"
BATCH = 128 if DEVICE == 0 else 8   # 128 matches Colab; CPU falls back to a small batch


def main():
    if not (CROPS / "train").exists():
        raise FileNotFoundError(f"{CROPS / 'train'} not found - run scripts/extract_crops.py first")

    if DEVICE == "cpu":
        print("[!] No GPU detected. yolo26l-cls at imgsz=288 for 100 epochs will take a very")
        print("    long time on CPU. This config was designed for Colab's GPU - run it there.\n")

    print(f"Model: {MODEL} | device: {DEVICE} | batch: {BATCH} | crops: {CROPS}")

    model = YOLO(MODEL)
    model.train(
        data=str(CROPS),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        patience=PATIENCE,
        workers=0,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        optimizer="AdamW",
        lr0=0.0005,
        cos_lr=True,
        weight_decay=0.01,
        dropout=0.3,
        label_smoothing=0.1,
        auto_augment="randaugment",
        erasing=0.4,
        hsv_h=0.03, hsv_s=0.7, hsv_v=0.4,
        scale=0.6,
        fliplr=0.5,
    )
    print(f"\nBest weights: {RUNS_DIR / RUN_NAME / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()