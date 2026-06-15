"""
STEP 2B: IMPROVED CZECH ROP PARSING
Maps all 6004 images using patient ID from filename
"""

import pandas as pd
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
from collections import defaultdict

print("="*80)
print("STEP 2B: IMPROVED CZECH ROP PARSING (Using Patient ID)")
print("="*80)

# Paths
excel_path = Path(r"C:\Users\ASUS\FYP - MidPoint\data\raw_data\classification_kaggle\images_stack\infant_retinal_database_info.xlsx")
images_dir = Path(r"C:\Users\ASUS\FYP - MidPoint\data\raw_data\classification_kaggle\images_stack")
output_dir = Path(r"C:\Users\ASUS\FYP - MidPoint\data\czech_rop_organized")

# Read metadata
df = pd.read_excel(excel_path)
print(f"✓ Loaded {len(df)} patient entries")

# Create patient ID to label mapping
patient_to_label = {}
for idx, row in df.iterrows():
    patient_id = str(row['ID']).zfill(3)  # e.g., "001"
    diagnosis_code = row['DIAGNOSIS CODE (DG)']
    label = 'Normal' if diagnosis_code == 0 else 'ROP'
    patient_to_label[patient_id] = label

print(f"✓ Created mapping for {len(patient_to_label)} patients")

# Map all images
print("\n⏳ Mapping all 6004 images...")
image_files = list(images_dir.glob("*.jpg"))
image_label_map = {}
unmapped_count = 0

for img_file in image_files:
    # Extract patient ID from filename: "001_F_GA41_..." → "001"
    patient_id = img_file.name.split('_')[0]
    
    if patient_id in patient_to_label:
        image_label_map[img_file.name] = patient_to_label[patient_id]
    else:
        unmapped_count += 1

print(f"✓ Mapped {len(image_label_map)} images")
print(f"⚠ Unmapped: {unmapped_count} images (patient ID not in Excel)")

# Patient-wise split (CRITICAL: prevent data leakage)
print("\n⏳ Creating PATIENT-WISE train/val/test splits...")

patient_ids = list(patient_to_label.keys())
patient_labels = [patient_to_label[pid] for pid in patient_ids]

# Split patients (not images)
train_patients, temp_patients, train_plabels, temp_plabels = train_test_split(
    patient_ids, patient_labels, test_size=0.3, stratify=patient_labels, random_state=42
)
val_patients, test_patients, val_plabels, test_plabels = train_test_split(
    temp_patients, temp_plabels, test_size=0.5, stratify=temp_plabels, random_state=42
)

train_patients_set = set(train_patients)
val_patients_set = set(val_patients)
test_patients_set = set(test_patients)

# Assign images based on patient split
train_imgs, val_imgs, test_imgs = [], [], []
train_labels, val_labels, test_labels = [], [], []

for img_name, label in image_label_map.items():
    patient_id = img_name.split('_')[0]
    
    if patient_id in train_patients_set:
        train_imgs.append(img_name)
        train_labels.append(label)
    elif patient_id in val_patients_set:
        val_imgs.append(img_name)
        val_labels.append(label)
    elif patient_id in test_patients_set:
        test_imgs.append(img_name)
        test_labels.append(label)

print(f"\nPatient-wise split:")
print(f"  Train: {len(train_patients)} patients → {len(train_imgs)} images ({train_labels.count('Normal')} Normal, {train_labels.count('ROP')} ROP)")
print(f"  Val:   {len(val_patients)} patients → {len(val_imgs)} images ({val_labels.count('Normal')} Normal, {val_labels.count('ROP')} ROP)")
print(f"  Test:  {len(test_patients)} patients → {len(test_imgs)} images ({test_labels.count('Normal')} Normal, {test_labels.count('ROP')} ROP)")

# Clean output directory
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True)

# Organize files
print("\n⏳ Organizing images...")
for split_name, split_imgs, split_labels in [
    ('train', train_imgs, train_labels),
    ('val', val_imgs, val_labels),
    ('test', test_imgs, test_labels)
]:
    for label in ['Normal', 'ROP']:
        label_dir = output_dir / split_name / label
        label_dir.mkdir(parents=True, exist_ok=True)
    
    for img_name, label in zip(split_imgs, split_labels):
        src = images_dir / img_name
        dst = output_dir / split_name / label / img_name
        shutil.copy2(src, dst)

print("✓ Images organized!")

# Save metadata
metadata = {
    'total_images': len(image_label_map),
    'total_patients': len(patient_to_label),
    'train_patients': len(train_patients),
    'train_images': len(train_imgs),
    'val_patients': len(val_patients),
    'val_images': len(val_imgs),
    'test_patients': len(test_patients),
    'test_images': len(test_imgs),
    'normal_count': sum(1 for l in image_label_map.values() if l == 'Normal'),
    'rop_count': sum(1 for l in image_label_map.values() if l == 'ROP'),
    'unmapped_images': unmapped_count,
    'split_strategy': 'patient-wise (prevents data leakage)',
    'output_directory': str(output_dir)
}

metadata_path = Path(r"C:\Users\ASUS\FYP - MidPoint\results\czech_rop_metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ Metadata saved to: {metadata_path}")

print("\n" + "="*80)
print("CZECH ROP DATASET READY FOR PRETRAINING")
print("="*80)
print(f"Total images: {len(image_label_map)}")
print(f"Training images: {len(train_imgs)} ({len(train_patients)} patients)")
print(f"Validation images: {len(val_imgs)} ({len(val_patients)} patients)")
print(f"Test images: {len(test_imgs)} ({len(test_patients)} patients)")
print(f"\nOutput: {output_dir}")
print("="*80)
