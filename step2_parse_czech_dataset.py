"""
STEP 2: PARSE CZECH ROP DATASET METADATA
Reads Excel file, extracts diagnosis codes, creates binary classification labels
Organizes 6004 images into train/val/test splits
"""

import pandas as pd
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

print("="*80)
print("STEP 2: PARSING CZECH ROP DATASET")
print("="*80)

# Paths
excel_path = Path(r"C:\Users\ASUS\FYP - MidPoint\data\raw_data\classification_kaggle\images_stack\infant_retinal_database_info.xlsx")
images_dir = Path(r"C:\Users\ASUS\FYP - MidPoint\data\raw_data\classification_kaggle\images_stack")
output_dir = Path(r"C:\Users\ASUS\FYP - MidPoint\data\czech_rop_organized")

# Read Excel metadata
print(f"\nReading metadata from: {excel_path.name}")
df = pd.read_excel(excel_path)

print(f"✓ Loaded {len(df)} entries")
print(f"\nColumns: {list(df.columns)}")

# Check for diagnosis column (case-insensitive)
diagnosis_col = None
for col in df.columns:
    if 'diagnosis' in col.lower() and 'code' in col.lower():
        diagnosis_col = col
        break

if not diagnosis_col:
    # Try just 'diagnosis'
    for col in df.columns:
        if 'diagnosis' in col.lower():
            diagnosis_col = col
            break

if diagnosis_col:
    print(f"\n✓ Found diagnosis column: '{diagnosis_col}'")
    print(f"Unique diagnosis codes: {sorted(df[diagnosis_col].unique())}")
    
    # Create binary labels (0 = Normal/Physiological, 1-13 = ROP)
    df['binary_label'] = df[diagnosis_col].apply(lambda x: 'Normal' if x == 0 else 'ROP')
    
    print(f"\nBinary classification distribution:")
    print(df['binary_label'].value_counts())
else:
    print("\n⚠ No diagnosis column found. Available columns:")
    print(df.columns.tolist())
    print("\nPlease specify the correct column name.")
    exit(1)

# Match images to metadata
# Extract patient/image ID from filename (format: 001_F_GA41_BW2905_PA44_DG11_PF0_D1_S01_1.jpg)
print("\n⏳ Matching images to metadata...")

image_files = list(images_dir.glob("*.jpg"))
print(f"✓ Found {len(image_files)} images")

# Check if Excel has filename column
filename_col = None
for col in df.columns:
    if 'file' in col.lower() or 'image' in col.lower() or 'name' in col.lower():
        filename_col = col
        break

if filename_col:
    print(f"✓ Using filename column: '{filename_col}'")
    
    # Create mapping
    image_label_map = {}
    for idx, row in df.iterrows():
        filename = str(row[filename_col])
        label = row['binary_label']
        
        # Find matching image file
        matching_files = [f for f in image_files if filename in f.name]
        if matching_files:
            image_label_map[matching_files[0].name] = label
    
    print(f"✓ Matched {len(image_label_map)} images to labels")
else:
    print("\n⚠ No filename column found. Using patient ID matching...")
    
    # Extract patient ID from first part of filename
    image_label_map = {}
    patient_col = df.columns[0]  # Assume first column is patient ID
    
    for img_file in image_files:
        patient_id = img_file.name.split('_')[0]  # e.g., "001" from "001_F_GA41_..."
        
        # Find row for this patient
        patient_rows = df[df[patient_col].astype(str).str.contains(patient_id, na=False)]
        if not patient_rows.empty:
            label = patient_rows.iloc[0]['binary_label']
            image_label_map[img_file.name] = label
    
    print(f"✓ Matched {len(image_label_map)} images using patient ID")

# Split dataset: 70% train, 15% val, 15% test
print("\n⏳ Creating train/val/test splits...")

labeled_images = list(image_label_map.keys())
labels = [image_label_map[img] for img in labeled_images]

# Stratified split
train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
    labeled_images, labels, test_size=0.3, stratify=labels, random_state=42
)
val_imgs, test_imgs, val_labels, test_labels = train_test_split(
    temp_imgs, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

print(f"\nSplit sizes:")
print(f"  Train: {len(train_imgs)} ({len([l for l in train_labels if l=='Normal'])} Normal, {len([l for l in train_labels if l=='ROP'])} ROP)")
print(f"  Val:   {len(val_imgs)} ({len([l for l in val_labels if l=='Normal'])} Normal, {len([l for l in val_labels if l=='ROP'])} ROP)")
print(f"  Test:  {len(test_imgs)} ({len([l for l in test_labels if l=='Normal'])} Normal, {len([l for l in test_labels if l=='ROP'])} ROP)")

# Organize files
print("\n⏳ Organizing images into folders...")

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
        
        if src.exists():
            shutil.copy2(src, dst)

print("✓ Images organized!")

# Save metadata
metadata = {
    'total_images': len(labeled_images),
    'train_size': len(train_imgs),
    'val_size': len(val_imgs),
    'test_size': len(test_imgs),
    'normal_count': len([l for l in labels if l == 'Normal']),
    'rop_count': len([l for l in labels if l == 'ROP']),
    'output_directory': str(output_dir),
    'diagnosis_column_used': diagnosis_col
}

metadata_path = Path(r"C:\Users\ASUS\FYP - MidPoint\results\czech_rop_metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ Metadata saved to: {metadata_path}")

print("\n" + "="*80)
print("CZECH ROP DATASET READY")
print("="*80)
print(f"Output directory: {output_dir}")
print(f"\nNext step: Train ResNet50 on {len(train_imgs)} training images")
print("="*80)
