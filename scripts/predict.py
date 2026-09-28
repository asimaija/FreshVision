import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────
PROJECT_ROOT = Path(r"D:\Projects\FreshVision")
WEIGHTS_PATH = PROJECT_ROOT / "models" / "runs" / "freshvision_yolo26s_v3" / "weights" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "predictions"

# ── Inference settings ───────────────────────────────────
CONF_THRESHOLD = 0.25   # minimum confidence to show a detection
IOU_THRESHOLD = 0.45    # NMS overlap threshold


def predict_image(image_path, save=True, show_labels=True):
    """Run detection on a single image and save the annotated result."""
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights not found at {WEIGHTS_PATH}")

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(WEIGHTS_PATH))

    results = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        save=save,
        project=str(OUTPUT_DIR),
        name="run",
        exist_ok=True,
        show_labels=show_labels,
        show_conf=True,
    )

    for r in results:
        print(f"\nImage: {image_path.name}")
        print(f"Detections: {len(r.boxes)}")
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]
            xyxy = box.xyxy[0].tolist()
            print(f"  - {cls_name:25s} conf={conf:.2f}  box={[round(x, 1) for x in xyxy]}")

    return results


def predict_folder(folder_path, save=True):
    """Run detection on every image in a folder."""
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights not found at {WEIGHTS_PATH}")

    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(WEIGHTS_PATH))

    results = model.predict(
        source=str(folder_path),
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        save=save,
        project=str(OUTPUT_DIR),
        name="run",
        exist_ok=True,
    )

    print(f"\nProcessed {len(results)} images from {folder_path}")
    print(f"Annotated images saved to: {OUTPUT_DIR / 'run'}")
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="FreshVision inference")
    parser.add_argument(
        "source", type=str,
        help="Path to a single image OR a folder of images"
    )
    args = parser.parse_args()

    source_path = Path(args.source)

    print("=" * 60)
    print("FreshVision - Inference")
    print("=" * 60)
    print(f"Weights : {WEIGHTS_PATH}")
    print(f"Source  : {source_path}")
    print(f"Conf    : {CONF_THRESHOLD}  |  IoU: {IOU_THRESHOLD}")
    print("=" * 60)

    if source_path.is_dir():
        predict_folder(source_path)
    else:
        predict_image(source_path)


if __name__ == "__main__":
    main()