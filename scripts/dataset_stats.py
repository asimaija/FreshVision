from pathlib import Path

# Dataset path
DATASET_PATH = Path(
    r"D:\Projects\FreshVision\data\raw\LVIS_Fruits_And_Vegetables"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def count_images(folder):
    """Count image files inside a folder."""
    return sum(
        1
        for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


# Paths
train_path = DATASET_PATH / "images" / "train"
val_path = DATASET_PATH / "images" / "val"
test_path = DATASET_PATH / "images" / "test"


# Count images
train_count = count_images(train_path)
val_count = count_images(val_path)
test_count = count_images(test_path)

# Total
total = train_count + val_count + test_count


print("=" * 55)
print("FreshVision Dataset Statistics")
print("=" * 55)

print(f"Training images   : {train_count:,}")
print(f"Validation images : {val_count:,}")
print(f"Test images       : {test_count:,}")

print("-" * 55)
print(f"Total images      : {total:,}")
print("-" * 55)


# Percentages
if total > 0:
    train_percentage = (train_count / total) * 100
    val_percentage = (val_count / total) * 100
    test_percentage = (test_count / total) * 100

    print(f"Training split    : {train_percentage:.2f}%")
    print(f"Validation split  : {val_percentage:.2f}%")
    print(f"Test split        : {test_percentage:.2f}%")

print("=" * 55)