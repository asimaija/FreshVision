from pathlib import Path

TRAIN_LABELS = Path(r"D:\Projects\FreshVision\data\raw\LVIS_Fruits_And_Vegetables\labels\train")

files = list(TRAIN_LABELS.glob("*.txt"))
print("Total label files:", len(files))

if files:
    sample = files[0]
    print("Sample file:", sample.name)
    print("Sample content:", repr(sample.read_text()[:300]))

count_37 = 0
for f in files:
    lines = f.read_text().strip().splitlines()
    for line in lines:
        parts = line.split()
        if parts and parts[0] == "37":
            count_37 += 1
            break

print("Files containing class 37 (lemon):", count_37)