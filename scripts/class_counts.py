from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "LVIS_Fruits_And_Vegetables"
DATA_YAML = PROJECT_ROOT / "configs" / "dataset.yaml"

names = yaml.safe_load(DATA_YAML.read_text())["names"]

orig_inst, total_inst = Counter(), Counter()
for f in (DATASET_PATH / "labels" / "train").rglob("*.txt"):
    is_copy = f.stem.startswith("oversample_")
    for line in f.read_text().splitlines():
        parts = line.split()
        if parts:
            cid = int(parts[0])
            total_inst[cid] += 1
            if not is_copy:
                orig_inst[cid] += 1

print(f"{'ID':>3}  {'Class':<28}{'Original':>10}{'With copies':>13}")
print("-" * 56)
for cid in sorted(names, key=lambda c: orig_inst[c]):
    print(f"{cid:>3}  {names[cid][:27]:<28}{orig_inst[cid]:>10}{total_inst[cid]:>13}")