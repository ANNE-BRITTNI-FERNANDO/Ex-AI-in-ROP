"""
STEP A: ORGANIZE CZECH ROP DATASET (6004 images)
- Parse binary labels from filenames (DG0=Normal, DG1-13=ROP)
- Patient-level train/val/test split (no data leakage)
- Output: organized folder structure + metadata JSON
"""

import os
import re
import json
import shutil
from pathlib import Path
from collections import defaultdict
import random

random.seed(42)

IMAGES_DIR = Path("data/raw_data/classification_kaggle/images_stack")
OUTPUT_DIR = Path("data/czech_rop_organized")
METADATA_PATH = Path("results/czech_dataset_organized.json")

print("=" * 70)
print("STEP A: ORGANIZING CZECH ROP DATASET")
print("=" * 70)

# Parse all images and extract labels from filename
# Format: 001_F_GA41_BW2905_PA44_DG11_PF0_D1_S01_1.jpg
# DG0 = Normal (diagnosis grade 0), DG1+ = ROP

image_files = list(IMAGES_DIR.glob("*.jpg"))
print(f"\nFound {len(image_files)} images")

patient_data = defaultdict(list)  # patient_id -> list of (filepath, label)
label_counts = {"Normal": 0, "ROP": 0}
skipped = 0

for img_path in image_files:
    fname = img_path.stem  # e.g. 001_F_GA41_BW2905_PA44_DG11_PF0_D1_S01_1
    parts = fname.split("_")
    
    # Extract patient ID (first part) and diagnosis (DGxx part)
    patient_id = parts[0]  # e.g. "001"
    
    dg_part = None
    for p in parts:
        if p.startswith("DG"):
            dg_part = p
            break
    
    if dg_part is None:
        skipped += 1
        continue
    
    try:
        dg_code = int(dg_part[2:])  # e.g. DG11 -> 11
    except ValueError:
        skipped += 1
        continue
    
    label = "Normal" if dg_code == 0 else "ROP"
    label_counts[label] += 1
    patient_data[patient_id].append((img_path, label))

print(f"Parsed: {label_counts['Normal']} Normal, {label_counts['ROP']} ROP")
print(f"Unique patients: {len(patient_data)}")
if skipped:
    print(f"Skipped (unparseable): {skipped}")

# Patient-level split to prevent data leakage
# Each patient's ALL images go to the same split
patients = list(patient_data.keys())
random.shuffle(patients)

n = len(patients)
n_train = int(n * 0.70)
n_val = int(n * 0.15)

train_patients = set(patients[:n_train])
val_patients = set(patients[n_train:n_train + n_val])
test_patients = set(patients[n_train + n_val:])

print(f"\nPatient split: {len(train_patients)} train / {len(val_patients)} val / {len(test_patients)} test")

# Create output directories
for split in ["train", "val", "test"]:
    for cls in ["Normal", "ROP"]:
        (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

# Copy images to split folders
split_counts = {"train": {"Normal": 0, "ROP": 0},
                "val": {"Normal": 0, "ROP": 0},
                "test": {"Normal": 0, "ROP": 0}}

for patient_id, images in patient_data.items():
    if patient_id in train_patients:
        split = "train"
    elif patient_id in val_patients:
        split = "val"
    else:
        split = "test"
    
    for img_path, label in images:
        dest = OUTPUT_DIR / split / label / img_path.name
        shutil.copy2(img_path, dest)
        split_counts[split][label] += 1

print("\nDataset organized:")
total = 0
for split in ["train", "val", "test"]:
    n_norm = split_counts[split]["Normal"]
    n_rop = split_counts[split]["ROP"]
    n_total = n_norm + n_rop
    total += n_total
    print(f"  {split:5s}: {n_total:4d} images  (Normal={n_norm}, ROP={n_rop})")

print(f"  Total: {total} images")

# Save metadata
metadata = {
    "total_images": total,
    "total_patients": len(patient_data),
    "label_distribution": label_counts,
    "split_strategy": "patient-level (no data leakage)",
    "train_patients": len(train_patients),
    "val_patients": len(val_patients),
    "test_patients": len(test_patients),
    "split_counts": split_counts,
    "output_directory": str(OUTPUT_DIR),
    "random_seed": 42
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nMetadata saved to: {METADATA_PATH}")
print("\nDone! Dataset ready for training.")
print("=" * 70)
