import random, re, shutil
from pathlib import Path

OUT = Path("data/crops/train")
CAP = {"fruits360": 250, "kaggle36": 300}      # dataset -> max images per class
SOURCES = {                                    # dataset -> folder
    "fruits360": Path("downloads/fruits-360/Training"),
    "kaggle36":  Path("downloads/fruit_veg/train"),
}
# alag naam -> apni class (jo naam yahan nahi, wo auto-clean hoga)
RENAME = {"tomatoes": "tomato", "capsicum": "bell_pepper", "colocasia": "taro_root"}

def clean(name):
    n = re.sub(r"\d+$", "", name.lower())          # "tomato 1" -> "tomato"
    n = re.sub(r"[^a-z0-9]+", "_", n).strip("_")
    return RENAME.get(n, n)

random.seed(0)
for ds, root in SOURCES.items():
    groups = {}
    for d in root.iterdir():
        if d.is_dir():
            groups.setdefault(clean(d.name), []).extend(
                p for p in d.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for cls, files in groups.items():
        random.shuffle(files)
        dst = OUT / cls
        dst.mkdir(parents=True, exist_ok=True)
        for p in files[:CAP[ds]]:
            shutil.copy(p, dst / f"{ds}_{p.parent.name}_{p.name}")
        print(ds, cls, min(len(files), CAP[ds]))