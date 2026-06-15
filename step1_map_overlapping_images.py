"""
STEP 1: MAP 97 OVERLAPPING IMAGES TO THEIR AVAILABLE MASKS
Creates JSON mapping: image_hash -> {classification_path, mask_paths, lesion_types}
"""

import os
import json
import hashlib
from pathlib import Path
from collections import defaultdict

def compute_hash(img_path):
    """Compute MD5 hash of image"""
    try:
        with open(img_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def get_all_images(directory):
    """Recursively get all image paths"""
    images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join(root, file))
    return images

base = Path(r"C:\Users\ASUS\FYP - MidPoint\data\raw_data")

print("="*80)
print("STEP 1: MAPPING OVERLAPPING IMAGES TO MASKS")
print("="*80)

# Build hash index
hash_to_info = defaultdict(lambda: {
    'classification_path': None,
    'classification_label': None,
    'segmentation_paths': {},
    'mask_paths': {}
})

# Index classification images
classification_folders = {
    'Normal': ['Neo_Normal', 'RetCam_Normal'],
    'ROP': ['Neo_ROP', 'RetCam_ROP']
}

for label, folders in classification_folders.items():
    for folder in folders:
        folder_path = base / 'classification' / label / folder
        if folder_path.exists():
            for img_path in get_all_images(str(folder_path)):
                img_hash = compute_hash(img_path)
                if img_hash:
                    hash_to_info[img_hash]['classification_path'] = img_path
                    hash_to_info[img_hash]['classification_label'] = label

# Index segmentation images and masks
segmentation_types = ['ridge', 'optic_disc', 'vessels']
device_prefixes = [('Neo', 'Neo_'), ('RetCam', 'RetCam_'), ('Retcam', 'Retcam_')]

for seg_type in segmentation_types:
    for device, prefix in device_prefixes:
        # Image folders
        if seg_type == 'optic_disc':
            img_folder_names = [f'{prefix}OpticDisc_images']
            mask_folder_names = [f'{prefix}OpticDisc_masks']
        elif seg_type == 'ridge':
            img_folder_names = [f'{prefix}Ridge_images']
            mask_folder_names = [f'{prefix}Ridge_masks']
        else:  # vessels
            img_folder_names = [f'{prefix}Vessels_images']
            mask_folder_names = [f'{prefix}Vessels_masks']
        
        for img_folder_name in img_folder_names:
            img_folder = base / 'segmentation' / seg_type / 'images' / img_folder_name
            
            if img_folder.exists():
                for img_path in get_all_images(str(img_folder)):
                    img_hash = compute_hash(img_path)
                    if img_hash:
                        hash_to_info[img_hash]['segmentation_paths'][seg_type] = img_path
                        
                        # Find corresponding mask (same filename)
                        img_name = os.path.basename(img_path)
                        for mask_folder_name in mask_folder_names:
                            mask_folder = base / 'segmentation' / seg_type / 'masks' / mask_folder_name
                            mask_path = mask_folder / img_name
                            if mask_path.exists():
                                hash_to_info[img_hash]['mask_paths'][seg_type] = str(mask_path)
                                break

# Filter to only overlapping images (have both classification and segmentation)
overlapping = {}
for img_hash, info in hash_to_info.items():
    if info['classification_path'] and info['segmentation_paths']:
        overlapping[img_hash] = info

print(f"\n✓ Found {len(overlapping)} images with BOTH classification labels AND segmentation masks\n")

# Analyze mask availability
mask_counts = defaultdict(int)
for img_hash, info in overlapping.items():
    lesion_types = sorted(info['mask_paths'].keys())
    mask_combo = '+'.join(lesion_types) if lesion_types else 'none'
    mask_counts[mask_combo] += 1

print("Mask Availability Breakdown:")
for combo, count in sorted(mask_counts.items(), key=lambda x: -x[1]):
    print(f"  {combo}: {count} images")

# Create structured output
output = {
    'total_overlapping_images': len(overlapping),
    'images': []
}

for img_hash, info in overlapping.items():
    output['images'].append({
        'image_hash': img_hash,
        'classification_path': info['classification_path'],
        'classification_label': info['classification_label'],
        'available_lesions': sorted(info['segmentation_paths'].keys()),
        'segmentation_image_paths': info['segmentation_paths'],
        'mask_paths': info['mask_paths']
    })

# Save mapping
output_path = Path(r"C:\Users\ASUS\FYP - MidPoint\results\overlapping_images_map.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Mapping saved to: {output_path}")

# Summary statistics
total_ridge = sum(1 for img in output['images'] if 'ridge' in img['available_lesions'])
total_od = sum(1 for img in output['images'] if 'optic_disc' in img['available_lesions'])
total_vessels = sum(1 for img in output['images'] if 'vessels' in img['available_lesions'])

normal_count = sum(1 for img in output['images'] if img['classification_label'] == 'Normal')
rop_count = sum(1 for img in output['images'] if img['classification_label'] == 'ROP')

print("\n" + "="*80)
print("SUMMARY FOR GRAD-CAM VALIDATION")
print("="*80)
print(f"Total usable images: {len(overlapping)}")
print(f"\nBy classification label:")
print(f"  Normal: {normal_count}")
print(f"  ROP: {rop_count}")
print(f"\nBy lesion type:")
print(f"  Ridge masks: {total_ridge}")
print(f"  Optic Disc masks: {total_od}")
print(f"  Vessel masks: {total_vessels}")
print(f"\nThese {len(overlapping)} images will be used for:")
print(f"  - Grad-CAM heatmap generation")
print(f"  - IoU/Dice computation against expert masks")
print(f"  - Statistical analysis by lesion type")
print("="*80)
