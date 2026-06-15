import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import json

# Set random seed for reproducibility
random.seed(42)

# Paths
raw_data_path = Path(r"C:\Users\ASUS\FYP - MidPoint\data\raw_data\classification")
split_data_path = Path(r"C:\Users\ASUS\FYP - MidPoint\data\classification")

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

print("="*70)
print("CLASSIFICATION DATASET SPLITTING")
print("="*70)
print(f"\nSplit Ratios: Train={TRAIN_RATIO*100}%, Val={VAL_RATIO*100}%, Test={TEST_RATIO*100}%")
print(f"Strategy: Stratified Random Split (maintains class balance)")
print(f"Random Seed: 42 (for reproducibility)")

# Create output directories
for split in ['train', 'val', 'test']:
    for class_name in ['Normal', 'ROP']:
        output_dir = split_data_path / split / class_name
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created: {output_dir}")

# Collect all images by class
print("\n" + "="*70)
print("COLLECTING IMAGES")
print("="*70)

class_images = defaultdict(list)

for class_name in ['Normal', 'ROP']:
    class_dir = raw_data_path / class_name
    
    # Collect from both Neo and RetCam subdirectories
    for subdir in class_dir.iterdir():
        if subdir.is_dir():
            for img_file in subdir.glob('*.png'):
                # Store tuple of (source_path, device_type)
                device_type = 'Neo' if 'Neo' in subdir.name else 'RetCam'
                class_images[class_name].append((img_file, device_type))
    
    print(f"{class_name}: {len(class_images[class_name])} images collected")

# Calculate split sizes for each class
print("\n" + "="*70)
print("CALCULATING SPLIT SIZES")
print("="*70)

split_info = {}

for class_name in ['Normal', 'ROP']:
    total = len(class_images[class_name])
    
    train_size = int(total * TRAIN_RATIO)
    val_size = int(total * VAL_RATIO)
    test_size = total - train_size - val_size  # Remainder goes to test
    
    split_info[class_name] = {
        'total': total,
        'train': train_size,
        'val': val_size,
        'test': test_size
    }
    
    print(f"\n{class_name}:")
    print(f"  Total: {total}")
    print(f"  Train: {train_size} ({train_size/total*100:.1f}%)")
    print(f"  Val:   {val_size} ({val_size/total*100:.1f}%)")
    print(f"  Test:  {test_size} ({test_size/total*100:.1f}%)")

# Shuffle and split for each class
print("\n" + "="*70)
print("SPLITTING AND COPYING FILES")
print("="*70)

split_details = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'Neo': 0, 'RetCam': 0, 'files': []}))
file_counter = defaultdict(lambda: 1)  # For renaming files sequentially

for class_name in ['Normal', 'ROP']:
    # Shuffle the list
    images = class_images[class_name].copy()
    random.shuffle(images)
    
    # Split indices
    train_end = split_info[class_name]['train']
    val_end = train_end + split_info[class_name]['val']
    
    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }
    
    # Copy files to respective directories
    for split_name, split_images in splits.items():
        print(f"\nCopying {class_name} to {split_name}...")
        
        for src_path, device_type in split_images:
            # Create new filename: class_device_number.png
            new_filename = f"{class_name}_{device_type}_{file_counter[(class_name, split_name, device_type)]:03d}.png"
            file_counter[(class_name, split_name, device_type)] += 1
            
            # Destination path
            dst_path = split_data_path / split_name / class_name / new_filename
            
            # Copy file
            shutil.copy2(src_path, dst_path)
            
            # Track statistics
            split_details[split_name][class_name]['count'] += 1
            split_details[split_name][class_name][device_type] += 1
            split_details[split_name][class_name]['files'].append(new_filename)
        
        print(f"  Copied {len(split_images)} images")

# Display summary
print("\n" + "="*70)
print("SPLIT SUMMARY")
print("="*70)

total_images = 0
for split_name in ['train', 'val', 'test']:
    print(f"\n{split_name.upper()}:")
    split_total = 0
    for class_name in ['Normal', 'ROP']:
        count = split_details[split_name][class_name]['count']
        neo_count = split_details[split_name][class_name]['Neo']
        retcam_count = split_details[split_name][class_name]['RetCam']
        split_total += count
        print(f"  {class_name:8s}: {count:3d} images (Neo: {neo_count:2d}, RetCam: {retcam_count:2d})")
    print(f"  {'TOTAL':8s}: {split_total:3d} images")
    total_images += split_total

print(f"\n{'GRAND TOTAL':8s}: {total_images} images")

# Verify against original count
original_total = sum(len(class_images[c]) for c in ['Normal', 'ROP'])
print(f"\nVerification: Original={original_total}, Split={total_images}, Match={'✓' if original_total == total_images else '✗'}")

# Save split report
report = {
    'split_date': '2026-01-18',
    'split_strategy': 'Stratified Random Split',
    'random_seed': 42,
    'ratios': {
        'train': TRAIN_RATIO,
        'val': VAL_RATIO,
        'test': TEST_RATIO
    },
    'summary': {
        split_name: {
            class_name: {
                'total': split_details[split_name][class_name]['count'],
                'Neo': split_details[split_name][class_name]['Neo'],
                'RetCam': split_details[split_name][class_name]['RetCam']
            }
            for class_name in ['Normal', 'ROP']
        }
        for split_name in ['train', 'val', 'test']
    },
    'total_images': total_images,
    'original_total': original_total
}

report_path = Path(r"C:\Users\ASUS\FYP - MidPoint\results\dataset_split_report.json")
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2)

print(f"\nReport saved to: {report_path}")

# Create a readable summary file
summary_path = Path(r"C:\Users\ASUS\FYP - MidPoint\results\DATASET_SPLIT_SUMMARY.md")
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("# Dataset Split Summary\n\n")
    f.write("**Date:** January 18, 2026  \n")
    f.write("**Strategy:** Stratified Random Split  \n")
    f.write("**Random Seed:** 42 (reproducible)  \n\n")
    
    f.write("## Split Ratios\n\n")
    f.write("- **Training:** 70%\n")
    f.write("- **Validation:** 15%\n")
    f.write("- **Test:** 15%\n\n")
    
    f.write("## Classification Split Results\n\n")
    f.write("| Split | Normal | ROP | Total |\n")
    f.write("|-------|--------|-----|-------|\n")
    
    for split_name in ['train', 'val', 'test']:
        normal_count = split_details[split_name]['Normal']['count']
        rop_count = split_details[split_name]['ROP']['count']
        total = normal_count + rop_count
        f.write(f"| {split_name.capitalize()} | {normal_count} | {rop_count} | {total} |\n")
    
    f.write(f"\n**Total:** {total_images} images\n\n")
    
    f.write("## Device Distribution\n\n")
    for split_name in ['train', 'val', 'test']:
        f.write(f"### {split_name.capitalize()}\n\n")
        for class_name in ['Normal', 'ROP']:
            neo = split_details[split_name][class_name]['Neo']
            retcam = split_details[split_name][class_name]['RetCam']
            f.write(f"- **{class_name}:** Neo={neo}, RetCam={retcam}\n")
        f.write("\n")
    
    f.write("## Segmentation Data\n\n")
    f.write("Segmentation data remains **unsplit** in the original location:\n")
    f.write("- `data/raw_data/segmentation/`\n")
    f.write("- **Total:** 600 files (300 images + 300 masks)\n")
    f.write("- **Usage:** Grad-CAM evaluation only (not for training)\n\n")
    
    f.write("## Directory Structure\n\n")
    f.write("```\n")
    f.write("data/\n")
    f.write("├── classification/\n")
    f.write("│   ├── train/\n")
    f.write("│   │   ├── Normal/\n")
    f.write("│   │   └── ROP/\n")
    f.write("│   ├── val/\n")
    f.write("│   │   ├── Normal/\n")
    f.write("│   │   └── ROP/\n")
    f.write("│   └── test/\n")
    f.write("│       ├── Normal/\n")
    f.write("│       └── ROP/\n")
    f.write("└── raw_data/\n")
    f.write("    ├── classification/ (original, kept for reference)\n")
    f.write("    └── segmentation/ (not split, used for Grad-CAM)\n")
    f.write("```\n\n")
    
    f.write("## Notes\n\n")
    f.write("- ✅ Stratified split maintains class balance\n")
    f.write("- ✅ Random seed (42) ensures reproducibility\n")
    f.write("- ✅ Both Neo and RetCam images distributed across splits\n")
    f.write("- ✅ Files renamed for consistency: `ClassName_Device_XXX.png`\n")
    f.write("- ✅ Original raw data preserved in `raw_data/`\n")
    f.write("- ℹ️ Segmentation data NOT split (used for evaluation only)\n")

print(f"Summary saved to: {summary_path}")

print("\n" + "="*70)
print("✅ DATASET SPLIT COMPLETE!")
print("="*70)
print(f"\nClassification data split into:")
print(f"  📁 {split_data_path / 'train'}")
print(f"  📁 {split_data_path / 'val'}")
print(f"  📁 {split_data_path / 'test'}")
print(f"\nSegmentation data (not split):")
print(f"  📁 {Path(r'C:\Users\ASUS\FYP - MidPoint\data\raw_data\segmentation')}")
print(f"\nReady for training! 🚀")
