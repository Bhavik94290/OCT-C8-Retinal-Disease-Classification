from pathlib import Path
from collections import Counter

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

DATASET_ROOT = Path("dataset")

SPLITS = ["train", "val", "test"]


def count_images(folder):
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


print("=" * 70)
print("OCT-C8 DATASET CHECK")
print("=" * 70)

if not DATASET_ROOT.exists():
    print("ERROR: dataset/ folder does not exist.")
    raise SystemExit(1)

for split in SPLITS:

    split_path = DATASET_ROOT / split

    print(f"\n{'=' * 20} {split.upper()} {'=' * 20}")

    if not split_path.exists():
        print(f"WARNING: {split_path} not found.")
        continue

    total = 0

    class_dirs = sorted(
        [p for p in split_path.iterdir() if p.is_dir()]
    )

    if not class_dirs:
        print("No class folders found.")
        continue

    for class_dir in class_dirs:

        images = count_images(class_dir)

        print(
            f"{class_dir.name:10s} : {len(images):6d} images"
        )

        total += len(images)

    print("-" * 50)
    print(f"Total {split}: {total} images")


print("\n" + "=" * 70)
print("EXPECTED OCT-C8 CLASSES")
print("=" * 70)

expected_classes = [
    "AMD",
    "CNV",
    "CSR",
    "DME",
    "DR",
    "DRUSEN",
    "MH",
    "NORMAL"
]

for cls in expected_classes:
    print(f"  - {cls}")

print("\nDataset check completed.")
