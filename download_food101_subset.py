import random
import shutil
from pathlib import Path

from datasets import load_dataset

CLASSES = ["pizza", "sushi", "hamburger", "ice_cream", "ramen"]
OUTPUT_DIR = Path("data")
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


def main():
    print("Downloading Food-101 dataset (this may take a few minutes)...")
    ds = load_dataset("ethz/food101", split="train+validation", trust_remote_code=True)

    # Food-101 uses integer labels — build mapping
    label_names = ds.features["label"].names
    class_to_idx = {name: idx for idx, name in enumerate(label_names)}

    target_indices = {class_to_idx[c] for c in CLASSES}

    # Filter to our 5 classes
    ds_filtered = ds.filter(lambda x: x["label"] in target_indices)
    print(f"Filtered to {len(ds_filtered)} images across {len(CLASSES)} classes")

    # Group by class
    by_class: dict[str, list[int]] = {c: [] for c in CLASSES}
    for i, example in enumerate(ds_filtered):
        class_name = label_names[example["label"]]
        by_class[class_name].append(i)

    # Create output dirs
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in SPLITS:
        for cls in CLASSES:
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    # Split and save
    rng = random.Random(SEED)
    total_saved = 0

    for cls in CLASSES:
        indices = by_class[cls]
        rng.shuffle(indices)
        n = len(indices)

        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["val"])

        split_indices = {
            "train": indices[:n_train],
            "val": indices[n_train : n_train + n_val],
            "test": indices[n_train + n_val :],
        }

        for split_name, idxs in split_indices.items():
            for j, idx in enumerate(idxs):
                img = ds_filtered[idx]["image"]
                out_path = OUTPUT_DIR / split_name / cls / f"{cls}_{j:04d}.jpg"
                img.save(out_path, "JPEG")

            print(f"  {cls}/{split_name}: {len(idxs)} images")
            total_saved += len(idxs)

    print(f"\nDone! {total_saved} images saved to {OUTPUT_DIR}/")

    # Summary
    for split in SPLITS:
        count = sum(1 for _ in (OUTPUT_DIR / split).rglob("*.jpg"))
        print(f"  {split}: {count} images")


if __name__ == "__main__":
    main()
