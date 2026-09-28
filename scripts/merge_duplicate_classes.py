from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "LVIS_Fruits_And_Vegetables"

# Map duplicate class IDs -> canonical class ID
MERGE_MAP = {
    57: 30,   # strawberry -> Strawberry
    59: 35,   # tomato -> Tomato
}

SPLITS = ["train", "val", "test"]


def merge_labels():
    total_files_changed = 0
    total_lines_changed = 0

    for split in SPLITS:
        label_dir = DATASET_PATH / "labels" / split
        if not label_dir.exists():
            print(f"[!] Skipping missing folder: {label_dir}")
            continue

        for label_file in label_dir.glob("*.txt"):
            lines = label_file.read_text().strip().splitlines()
            if not lines:
                continue

            changed = False
            new_lines = []
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if cls_id in MERGE_MAP:
                    parts[0] = str(MERGE_MAP[cls_id])
                    changed = True
                    total_lines_changed += 1
                new_lines.append(" ".join(parts))

            if changed:
                label_file.write_text("\n".join(new_lines) + "\n")
                total_files_changed += 1

    print(f"Done. Files changed: {total_files_changed}  |  Lines remapped: {total_lines_changed}")


if __name__ == "__main__":
    merge_labels()