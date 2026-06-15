"""
Dataset Overlap Analysis for Quantitative XAI in ROP
Identifies images that have BOTH classification labels AND segmentation masks
This is CRITICAL for computing IoU/Dice between Grad-CAM and expert annotations
"""

import os
from pathlib import Path
import json

def get_image_ids(directory):
    """Extract image IDs (filenames without extension)"""
    images = []
    for file in os.listdir(directory):
        if file.endswith('.png') or file.endswith('.jpg'):
            images.append(file.replace('.png', '').replace('.jpg', ''))
    return set(images)

# Base paths
base_path = Path(r"C:\Users\ASUS\FYP - MidPoint\data\raw_data")

# Classification images
classification_paths = {
    'Neo_Normal': base_path / 'classification' / 'Normal' / 'Neo_Normal',
    'Neo_ROP': base_path / 'classification' / 'ROP' / 'Neo_ROP',
    'RetCam_Normal': base_path / 'classification' / 'Normal' / 'RetCam_Normal',
    'RetCam_ROP': base_path / 'classification' / 'ROP' / 'RetCam_ROP',
}

# Segmentation images (with masks)
segmentation_paths = {
    'Neo_Ridge': base_path / 'segmentation' / 'ridge' / 'images' / 'Neo_Ridge_images',
    'RetCam_Ridge': base_path / 'segmentation' / 'ridge' / 'images' / 'RetCam_Ridge_images',
    'Neo_OpticDisc': base_path / 'segmentation' / 'optic_disc' / 'images' / 'Neo_OpticDisc_images',
    'Retcam_OpticDisc': base_path / 'segmentation' / 'optic_disc' / 'images' / 'Retcam_OpticDisc_images',
    'Neo_Vessels': base_path / 'segmentation' / 'vessels' / 'images' / 'Neo_Vessels_images',
    'RetCam_Vessels': base_path / 'segmentation' / 'vessels' / 'images' / 'RetCam_Vessels_images',
}

print("="*80)
print("HVDROPDB DATASET OVERLAP ANALYSIS")
print("="*80)
print("\nPurpose: Identify images with BOTH classification labels AND segmentation masks")
print("Why: These are the ONLY images where we can compute IoU/Dice for Grad-CAM validation\n")

# Get all classification image IDs
classification_ids = {}
for name, path in classification_paths.items():
    if path.exists():
        classification_ids[name] = get_image_ids(str(path))
        print(f"{name}: {len(classification_ids[name])} images")
    else:
        classification_ids[name] = set()
        print(f"{name}: Path not found!")

print()

# Get all segmentation image IDs
segmentation_ids = {}
for name, path in segmentation_paths.items():
    if path.exists():
        segmentation_ids[name] = get_image_ids(str(path))
        print(f"{name}: {len(segmentation_ids[name])} images")
    else:
        segmentation_ids[name] = set()
        print(f"{name}: Path not found!")

# Combine all classification and segmentation IDs
all_classification = set()
for ids in classification_ids.values():
    all_classification.update(ids)

all_segmentation = set()
for ids in segmentation_ids.values():
    all_segmentation.update(ids)

# Find overlap
overlap = all_classification.intersection(all_segmentation)

print("\n" + "="*80)
print("CRITICAL FINDINGS")
print("="*80)
print(f"\nTotal Classification Images: {len(all_classification)}")
print(f"Total Segmentation Images: {len(all_segmentation)}")
print(f"Images with BOTH labels and masks: {len(overlap)}")
print(f"Overlap Percentage: {len(overlap)/len(all_classification)*100:.1f}% of classification data")

# Device-specific overlap
neo_classification = classification_ids.get('Neo_Normal', set()).union(classification_ids.get('Neo_ROP', set()))
neo_segmentation = segmentation_ids.get('Neo_Ridge', set()).union(
    segmentation_ids.get('Neo_OpticDisc', set())).union(segmentation_ids.get('Neo_Vessels', set()))
neo_overlap = neo_classification.intersection(neo_segmentation)

retcam_classification = classification_ids.get('RetCam_Normal', set()).union(classification_ids.get('RetCam_ROP', set()))
retcam_segmentation = segmentation_ids.get('RetCam_Ridge', set()).union(
    segmentation_ids.get('Retcam_OpticDisc', set())).union(segmentation_ids.get('RetCam_Vessels', set()))
retcam_overlap = retcam_classification.intersection(retcam_segmentation)

print(f"\nNeonatal Device Overlap: {len(neo_overlap)} images")
print(f"RetCam Device Overlap: {len(retcam_overlap)} images")

# Analyze overlap by lesion type
print("\n" + "="*80)
print("OVERLAP BY LESION TYPE (Images usable for Grad-CAM validation)")
print("="*80)

lesion_overlaps = {}
for lesion_type, seg_ids in segmentation_ids.items():
    device = 'Neo' if 'Neo' in lesion_type else 'RetCam'
    lesion = lesion_type.replace('Neo_', '').replace('RetCam_', '').replace('Retcam_', '')
    
    # Find which classification images overlap with this lesion
    if device == 'Neo':
        class_ids = neo_classification
    else:
        class_ids = retcam_classification
    
    overlap_ids = class_ids.intersection(seg_ids)
    lesion_overlaps[lesion_type] = overlap_ids
    
    print(f"\n{lesion_type}:")
    print(f"  Total segmentation images: {len(seg_ids)}")
    print(f"  Have classification labels: {len(overlap_ids)}")
    print(f"  Usable for IoU validation: {len(overlap_ids)}")

# Save detailed overlap report
report = {
    'summary': {
        'total_classification': len(all_classification),
        'total_segmentation': len(all_segmentation),
        'total_overlap': len(overlap),
        'overlap_percentage': round(len(overlap)/len(all_classification)*100, 2),
        'neo_overlap': len(neo_overlap),
        'retcam_overlap': len(retcam_overlap)
    },
    'lesion_overlaps': {
        lesion: list(ids) for lesion, ids in lesion_overlaps.items()
    },
    'overlap_image_ids': sorted(list(overlap))
}

output_path = Path(r"C:\Users\ASUS\FYP - MidPoint\results\dataset_overlap_analysis.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n\n✓ Detailed overlap report saved to: {output_path}")

# Critical recommendation
print("\n" + "="*80)
print("RECOMMENDATION FOR YOUR FYP")
print("="*80)
print("""
Based on this analysis:

1. You have {overlap} images with BOTH classification labels AND segmentation masks
2. These are your "gold standard" images for quantitative XAI validation
3. Split strategy should be:
   - Use overlapping images for Grad-CAM validation (IoU/Dice computation)
   - Use all classification images for model training
   
4. Implementation Pipeline:
   Step 1: Train classification model on ALL {total_class} classification images
   Step 2: Apply augmentation to reach 2000 images
   Step 3: Generate Grad-CAM heatmaps for {overlap} overlapping images
   Step 4: Compute IoU/Dice between Grad-CAM and expert masks
   Step 5: Statistical analysis by lesion type (Ridge/OD/Vessels)
   
5. This aligns with your literature review methodology perfectly!
""".format(
    overlap=len(overlap),
    total_class=len(all_classification)
))
