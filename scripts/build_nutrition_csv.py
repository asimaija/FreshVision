"""
build_nutrition_csv.py - fetch per-100g nutrition for every FreshVision class
from USDA FoodData Central and save it as configs/nutrition.csv.

v2: fixes two problems found in the first run -
  1) "Foundation" foods store calories under Energy (Atwater) ids (2047/2048),
     not the plain Energy id (1008) - we now fall back through all three.
  2) Ambiguous single words (banana, carrot, cherry, date, fig, grape, onion,
     orange, papaya, peach, potato, pumpkin, tomato, ...) matched the wrong
     USDA entry - CLASS_QUERY_OVERRIDES below gives each one an unambiguous
     search phrase.

Setup:
    pip install requests pyyaml python-dotenv
    Put USDA_API_KEY in a .env file in the project root (see .env.example)

Run:
    python scripts/build_nutrition_csv.py
"""

import argparse
import csv
import os
import time
from pathlib import Path

import requests
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "configs" / "dataset.yaml"
OUT_CSV = ROOT / "configs" / "nutrition.csv"

USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

# Try these, in order, until one has a value - Foundation foods often only
# carry the Atwater-calculated energy ids, not the plain "Energy" id.
ENERGY_IDS = [1008, 2047, 2048]

NUTRIENT_IDS = {
    "protein_g": 1003,
    "fat_g": 1004,
    "carbs_g": 1005,
    "fiber_g": 1079,
    "sugar_g": 2000,
    "sodium_mg": 1093,
    "potassium_mg": 1092,
    "vitamin_c_mg": 1162,
}

PREFERRED_DATA_TYPES = ["Foundation", "SR Legacy", "Survey (FNDDS)"]

# Unambiguous search phrase for every class where the bare name matched the
# wrong food (a cookie, a pie, a different vegetable, plant leaves, etc.).
# If USDA's API keeps failing for a class even after retries (rare, network
# hiccup), fall back to these known-good values (source: USDA FDC, per 100g)
# instead of leaving the row empty.
HARDCODED_FALLBACK = {
    "banana": {
        "matched_food": "Bananas, raw (fallback)", "fdc_id": 173944, "data_type": "SR Legacy",
        "calories_kcal": 89, "protein_g": 1.09, "fat_g": 0.33, "carbs_g": 22.8,
        "fiber_g": 2.6, "sugar_g": 12.2, "sodium_mg": 1.0, "potassium_mg": 358, "vitamin_c_mg": 8.7,
    },
    "sweet potato": {
        "matched_food": "Sweet potato, raw (fallback)", "fdc_id": 168482, "data_type": "SR Legacy",
        "calories_kcal": 86, "protein_g": 1.57, "fat_g": 0.05, "carbs_g": 20.1,
        "fiber_g": 3.0, "sugar_g": 4.18, "sodium_mg": 55.0, "potassium_mg": 337, "vitamin_c_mg": 2.4,
    },
}

CLASS_QUERY_OVERRIDES = {
    "almond": "almonds, raw",
    "apricot": "apricots, raw",
    "asparagus": "asparagus, raw",
    "avocado": "avocados, raw, california",
    "banana": "bananas, raw",
    "bell pepper": "peppers, sweet, red, raw",
    "blackberry": "blackberries, raw",
    "blueberry": "blueberries, raw",
    "brussels sprouts": "brussels sprouts, raw",
    "carrot": "carrots, raw",
    "cauliflower": "cauliflower, raw",
    "cayenne": "spices, pepper, red or cayenne",
    "celery": "celery, raw",
    "cherry": "cherries, sweet, raw",
    "chickpea": "chickpeas, mature seeds, raw",
    "corn": "corn, sweet, yellow, raw",
    "cucumber": "cucumber, raw",
    "date": "dates, medjool, raw",
    "eggplant": "eggplant, raw",
    "fig": "figs, raw",
    "green bean": "beans, snap, green, raw",
    "grape": "grapes, red or green, raw",
    "kiwi fruit": "kiwifruit, green, raw",
    "lettuce": "lettuce, iceberg, raw",
    "mandarin": "tangerines, raw",
    "melon": "melons, honeydew, raw",
    "onion": "onions, raw",
    "orange": "oranges, raw",
    "papaya": "papayas, raw",
    "pea": "peas, green, raw",
    "peach": "peaches, raw",
    "pear": "pears, raw",
    "pickle": "pickles, cucumber, sour",
    "pineapple": "pineapple, raw",
    "potato": "potatoes, flesh and skin, raw",
    "prune": "prunes, dried",
    "pumpkin": "pumpkin, raw",
    "radish": "radishes, raw",
    "raspberry": "raspberries, raw",
    "strawberry": "strawberries, raw",
    "sweet potato": "sweet potato, raw",
    "tomato": "tomatoes, red, ripe, raw",
    "turnip": "turnips, raw",
    "watermelon": "watermelon, raw",
}


def key_of(name):
    return name.split("/")[0].strip().lower()


def load_class_keys():
    cfg = yaml.safe_load(DATA_YAML.read_text())
    return sorted({key_of(n) for n in cfg["names"].values()})


def search_food(query, api_key, retries=4, backoff=2.0):
    """USDA's API occasionally returns a transient error for a valid query -
    retry a few times with backoff before giving up."""
    params = {"query": query, "api_key": api_key, "pageSize": 10,
              "dataType": PREFERRED_DATA_TYPES}
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(USDA_SEARCH_URL, params=params, timeout=20)
            if r.status_code >= 400:
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code in (400, 429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep(backoff * attempt)
                    continue
                r.raise_for_status()
            foods = r.json().get("foods", [])
            if not foods:
                return None
            for dt in PREFERRED_DATA_TYPES:
                for f in foods:
                    if f.get("dataType") == dt:
                        return f
            return foods[0]
        except requests.RequestException as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(backoff * attempt)
    print(f"    (gave up after {retries} tries: {last_err})")
    return None


def extract_nutrients(food):
    row = {k: "" for k in NUTRIENT_IDS}
    row["calories_kcal"] = ""
    for n in food.get("foodNutrients", []):
        nid = n.get("nutrientId")
        if row["calories_kcal"] == "" and nid in ENERGY_IDS:
            row["calories_kcal"] = n.get("value", "")
        for key, wanted_id in NUTRIENT_IDS.items():
            if nid == wanted_id:
                row[key] = n.get("value", "")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("USDA_API_KEY"))
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit(
            "No USDA API key found. Put USDA_API_KEY=... in a .env file "
            "in the project root, or pass --api-key your_key_here"
        )

    keys = load_class_keys()
    print(f"Found {len(keys)} classes in {DATA_YAML}")

    rows, not_found = [], []
    for i, key in enumerate(keys, 1):
        query = CLASS_QUERY_OVERRIDES.get(key, f"{key}, raw")
        food = search_food(query, args.api_key)

        if food is None and key in HARDCODED_FALLBACK:
            fb = HARDCODED_FALLBACK[key]
            rows.append({"class": key, **fb})
            print(f"  [{i}/{len(keys)}] {key:<20} -> {fb['matched_food']} (fallback used)")
            time.sleep(args.sleep)
            continue

        if food is None:
            print(f"  [{i}/{len(keys)}] {key:<20} -> NOT FOUND (query: '{query}')")
            not_found.append(key)
            time.sleep(args.sleep)
            continue

        nutrients = extract_nutrients(food)
        rows.append({
            "class": key,
            "matched_food": food.get("description", ""),
            "fdc_id": food.get("fdcId", ""),
            "data_type": food.get("dataType", ""),
            **nutrients,
        })
        cal = nutrients["calories_kcal"]
        flag = "" if cal != "" else "  <-- still no calories, check by hand"
        print(f"  [{i}/{len(keys)}] {key:<20} -> {food.get('description', '')[:45]}{flag}")
        time.sleep(args.sleep)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["class", "matched_food", "fdc_id", "data_type", "calories_kcal"] + list(NUTRIENT_IDS)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {OUT_CSV}")
    if not_found:
        print(f"\n[!] {len(not_found)} class(es) still not found - add these to CLASS_QUERY_OVERRIDES:")
        for k in not_found:
            print(f"    {k}")


if __name__ == "__main__":
    main()