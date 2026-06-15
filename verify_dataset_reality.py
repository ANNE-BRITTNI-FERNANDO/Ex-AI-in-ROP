"""
DATASET REALITY CHECK
Determines the ACTUAL relationship between classification and segmentation images
"""

import os
from pathlib import Path
import hashlib
from collections import defaultdict
from PIL import Image

def compute_image_hash(img_path):
    """Compute MD5 hash of image file to identify duplicates"""
    try:
        with open(img_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def get_all_images(directory):
    """Get all image paths recursively"""
    images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join(root, file))
    return images

base_path = Path(r"C:\Users\ASUS\FYP - MidPoint\data\raw_data")

print("="*80)
print("DATASET REALITY CHECK - UNDERSTANDING YOUR ACTUAL DATA")
print("="*80)

# Get all classification images
classification_dir = base_path / 'classification'
classification_images = get_all_images(classification_dir)

# Get all segmentation images (images only, not masks)
segmentation_images = {
    'Ridge': get_all_images(base_path / 'segmentation' / 'ridge' / 'images'),
    'OpticDisc': get_all_images(base_path / 'segmentation' / 'optic_disc' / 'images'),
    'Vessels': get_all_images(base_path / 'segmentation' / 'vessels' / 'images'),
}

print(f"\n📊 RAW IMAGE COUNTS:")
print(f"Classification images: {len(classification_images)}")
print(f"Ridge images: {len(segmentation_images['Ridge'])}")
print(f"Optic Disc images: {len(segmentation_images['OpticDisc'])}")
print(f"Vessels images: {len(segmentation_images['Vessels'])}")
print(f"Total segmentation images: {sum(len(v) for v in segmentation_images.values())}")

# Build hash index
print("\n⏳ Computing image hashes (this may take 1-2 minutes)...")
hash_to_paths = defaultdict(list)

# Hash classification images
for img_path in classification_images:
    img_hash = compute_image_hash(img_path)
    if img_hash:
        hash_to_paths[img_hash].append(('Classification', img_path))

# Hash segmentation images
for lesion_type, img_list in segmentation_images.items():
    for img_path in img_list:
        img_hash = compute_image_hash(img_path)
        if img_hash:
            hash_to_paths[img_hash].append((f'Segmentation_{lesion_type}', img_path))

print("✓ Hash computation complete!")

# Analyze overlaps
unique_images = len(hash_to_paths)
duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}

print("\n" + "="*80)
print("🔍 OVERLAP ANALYSIS")
print("="*80)

print(f"\nTotal UNIQUE images across all folders: {unique_images}")
print(f"Total file count: {len(classification_images) + sum(len(v) for v in segmentation_images.values())}")
print(f"Duplicate images (same content, different locations): {len(duplicates)}")

# Classification ↔ Segmentation overlap
classification_segmentation_overlap = 0
classification_hashes = set()
segmentation_hashes = set()

for img_hash, paths in hash_to_paths.items():
    categories = [p[0] for p in paths]
    
    has_classification = any('Classification' in c for c in categories)
    has_segmentation = any('Segmentation' in c for c in categories)
    
    if has_classification:
        classification_hashes.add(img_hash)
    if has_segmentation:
        segmentation_hashes.add(img_hash)
    
    if has_classification and has_segmentation:
        classification_segmentation_overlap += 1

print(f"\n🎯 CRITICAL FINDINGS:")
print(f"Classification-only images: {len(classification_hashes - segmentation_hashes)}")
print(f"Segmentation-only images: {len(segmentation_hashes - classification_hashes)}")
print(f"Images in BOTH classification AND segmentation: {classification_segmentation_overlap}")

# Segmentation internal overlaps
print(f"\n🔬 SEGMENTATION FOLDER RELATIONSHIPS:")
ridge_hashes = set(compute_image_hash(img) for img in segmentation_images['Ridge'])
od_hashes = set(compute_image_hash(img) for img in segmentation_images['OpticDisc'])
vessels_hashes = set(compute_image_hash(img) for img in segmentation_images['Vessels'])

ridge_hashes.discard(None)
od_hashes.discard(None)
vessels_hashes.discard(None)

ridge_od_overlap = len(ridge_hashes.intersection(od_hashes))
ridge_vessels_overlap = len(ridge_hashes.intersection(vessels_hashes))
od_vessels_overlap = len(od_hashes.intersection(vessels_hashes))
all_three_overlap = len(ridge_hashes.intersection(od_hashes).intersection(vessels_hashes))

print(f"Ridge ↔ Optic Disc overlap: {ridge_od_overlap} images")
print(f"Ridge ↔ Vessels overlap: {ridge_vessels_overlap} images")
print(f"Optic Disc ↔ Vessels overlap: {od_vessels_overlap} images")
print(f"All 3 lesion types (same image): {all_three_overlap} images")

# Show examples of duplicates
if duplicates:
    print(f"\n📂 EXAMPLE DUPLICATE IMAGES (first 5):")
    for i, (img_hash, paths) in enumerate(list(duplicates.items())[:5]):
        print(f"\nDuplicate #{i+1}:")
        for category, path in paths:
            print(f"  {category}: {Path(path).name}")

# Check if Ridge/1.png and OpticDisc/1.png are the same
print("\n" + "="*80)
print("🧪 FILENAME TEST: Are images with same number the same image?")
print("="*80)

ridge_img_1 = next((img for img in segmentation_images['Ridge'] if '1.png' in img or '1.jpg' in img), None)
od_img_1 = next((img for img in segmentation_images['OpticDisc'] if '1.png' in img or '1.jpg' in img), None)

if ridge_img_1 and od_img_1:
    hash_ridge_1 = compute_image_hash(ridge_img_1)
    hash_od_1 = compute_image_hash(od_img_1)
    
    print(f"Ridge/1.png hash: {hash_ridge_1[:16]}...")
    print(f"OpticDisc/1.png hash: {hash_od_1[:16]}...")
    
    if hash_ridge_1 == hash_od_1:
        print("✓ SAME IMAGE - Images with same filename are duplicates")
    else:
        print("✗ DIFFERENT IMAGES - Images with same filename are NOT the same")

# Final verdict
print("\n" + "="*80)
print("📋 DATASET STRUCTURE VERDICT")
print("="*80)

if classification_segmentation_overlap > 0:
    print(f"""
✓ You have {classification_segmentation_overlap} images with BOTH:
  - Classification labels (Normal/ROP)
  - Segmentation masks (Ridge/OD/Vessels)
  
These {classification_segmentation_overlap} images are USABLE for Grad-CAM validation (IoU/Dice)!

Strategy:
  1. Train model on all {len(classification_hashes)} classification images
  2. Generate Grad-CAM on {classification_segmentation_overlap} overlapping images
  3. Compute IoU/Dice against expert masks
""")
else:
    print(f"""
✗ NO OVERLAP between classification and segmentation datasets!

This means:
  - Classification: {len(classification_hashes)} images for training ROP classifier
  - Segmentation: {len(segmentation_hashes)} images with masks (NO classification labels)
  
Problem: Cannot compute IoU/Dice without classification labels!

Solution Options:
  1. Manually label segmentation images (Normal/ROP)
  2. Use model predictions as pseudo-labels
  3. Get expert to provide classification labels for segmentation images
""")

if all_three_overlap > 0:
    print(f"""
✓ {all_three_overlap} images have masks for ALL 3 lesion types (Ridge + OD + Vessels)
  - Enables comprehensive multi-lesion Grad-CAM analysis
""")

print("="*80)
