from pathlib import Path

from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "configs" / "dataset.yaml"

# Point this at whichever run's weights you want to evaluate
WEIGHTS_PATH = PROJECT_ROOT / "models" / "runs" / "freshvision_yolo26m_v4" / "weights" / "best.pt"

# Which split to evaluate on: "val" or "test"
EVAL_SPLIT = "test"


def main():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights not found at {WEIGHTS_PATH}")
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"dataset.yaml not found at {DATA_YAML}")

    print("=" * 60)
    print("FreshVision - Evaluation")
    print("=" * 60)
    print(f"Weights : {WEIGHTS_PATH}")
    print(f"Data    : {DATA_YAML}")
    print(f"Split   : {EVAL_SPLIT}")
    print("=" * 60)

    model = YOLO(str(WEIGHTS_PATH))

    metrics = model.val(
        data=str(DATA_YAML),
        split=EVAL_SPLIT,
        plots=True,
    )

    print("\n=== Overall Metrics ===")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP75    : {metrics.box.map75:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall   : {metrics.box.mr:.4f}")

    # ── Per-class breakdown ──────────────────────────────
    print("\n=== Per-Class AP50-95 ===")
    class_names = model.names
    ap_per_class = metrics.box.maps  # array of AP50-95 per class

    # Sort weakest classes first — these need the most attention
    results = sorted(
        [(class_names[i], ap) for i, ap in enumerate(ap_per_class)],
        key=lambda x: x[1]
    )

    print(f"{'Class':<35}{'AP50-95':>10}")
    print("-" * 45)
    for name, ap in results:
        flag = "  <-- weak" if ap < 0.10 else ""
        print(f"{name:<35}{ap:>10.4f}{flag}")

    print(f"\nSaved plots (confusion matrix, PR curves) to: {metrics.save_dir}")


if __name__ == "__main__":
    main()