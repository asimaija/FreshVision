from pathlib import Path

import torch
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "LVIS_Fruits_And_Vegetables"
DATA_YAML = PROJECT_ROOT / "configs" / "dataset.yaml"
RUNS_DIR = PROJECT_ROOT / "models" / "runs"

# ── Model config ─────────────────────────────────────────
MODEL_NAME = "yolo26m.pt"   # medium — a good balance of speed and accuracy
EPOCHS = 25
IMGSZ = 480
BATCH = 4
PATIENCE = 15
RUN_NAME = "freshvision_yolo26n_local"

# Auto-detect device: use GPU 0 if CUDA is available, else CPU.
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def ensure_data_yaml():
    """
    Use configs/dataset.yaml if it already exists and has real content.
    Only generate a placeholder if the file is missing or empty —
    never overwrites a real dataset.yaml you've already filled in.
    """
    if DATA_YAML.exists() and DATA_YAML.stat().st_size > 0:
        return DATA_YAML

    class_names = [f"class_{i}" for i in range(63)]

    content = (
        f"path: {DATASET_PATH.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"names:\n"
        + "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
        + "\n"
    )
    DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    DATA_YAML.write_text(content)
    print(f"[i] Wrote placeholder dataset.yaml to {DATA_YAML}")
    print("    -> Real class names not found - edit before serious training.")
    return DATA_YAML


def main():
    data_yaml = ensure_data_yaml()

    print("=" * 60)
    print("FreshVision - Local Training")
    print("=" * 60)
    print(f"Model      : {MODEL_NAME}")
    print(f"Data yaml  : {data_yaml}")
    print(f"Epochs     : {EPOCHS}  |  imgsz: {IMGSZ}  |  batch: {BATCH}")
    print(f"Device     : {DEVICE}  (CUDA available: {torch.cuda.is_available()})")
    print("=" * 60)

    if DEVICE == "cpu":
        print("\n[!] No GPU detected - training on CPU.")
        print("    Using yolo26n (nano) with reduced batch/imgsz to limit RAM usage.")
        print("    For serious training, use the Colab notebook in colab/ instead.\n")

    model = YOLO(MODEL_NAME)

    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        patience=PATIENCE,
        workers=0,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        save=True,
        val=True,
    )

    print("\nTraining complete.")
    print(f"Best weights: {RUNS_DIR / RUN_NAME / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()