"""
Verification Script: List all MD5 hash matches between classification and segmentation images
This allows manual verification that matched images are truly identical
"""

import json
import hashlib
from pathlib import Path
import pandas as pd

def compute_md5(filepath):
    """Compute MD5 hash of an image file"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Load the overlapping images map
with open('results/overlapping_images_map.json', 'r') as f:
    data = json.load(f)

print("="*100)
print("VERIFICATION: MD5 Hash Matches Between Classification and Segmentation Images")
print("="*100)
print(f"\nTotal overlapping images: {data['total_overlapping_images']}")
print("\nYou can manually verify these are identical by opening both files side-by-side\n")

# Prepare data for detailed listing
verification_data = []

for img in data['images']:
    classification_path = img['classification_path']
    classification_label = img['classification_label']
    
    # Compute hash for classification image
    class_hash = compute_md5(classification_path)
    
    # Get all available lesion masks for this image
    for lesion_type in img['available_lesions']:
        mask_path = img['mask_paths'][lesion_type]
        
        # Get the segmentation image path from JSON (already has correct path)
        seg_image_path = img['segmentation_image_paths'][lesion_type]
        
        # Compute hash for segmentation image
        seg_hash = compute_md5(seg_image_path)
        
        # Check if hashes match
        match_status = "✓ MATCH" if class_hash == seg_hash else "✗ MISMATCH"
        
        verification_data.append({
            'Classification Image': Path(classification_path).name,
            'Classification Path': classification_path,
            'Classification Hash': class_hash,
            'Segmentation Image': Path(seg_image_path).name,
            'Segmentation Path': seg_image_path,
            'Segmentation Hash': seg_hash,
            'Lesion Type': lesion_type,
            'Label': classification_label,
            'Status': match_status
        })

# Convert to DataFrame for better display
df = pd.DataFrame(verification_data)

# Save detailed CSV for manual inspection
output_csv = 'results/md5_hash_verification.csv'
df.to_csv(output_csv, index=False)
print(f"✓ Detailed verification saved to: {output_csv}")

# Display summary
print("\n" + "="*100)
print("SUMMARY BY LESION TYPE")
print("="*100)
for lesion in ['optic_disc', 'vessels', 'ridge']:
    lesion_df = df[df['Lesion Type'] == lesion]
    matches = (lesion_df['Status'] == '✓ MATCH').sum()
    total = len(lesion_df)
    print(f"{lesion.upper()}: {matches}/{total} matches")

# Check for any mismatches
mismatches = df[df['Status'] == '✗ MISMATCH']
if len(mismatches) > 0:
    print(f"\n⚠️ WARNING: Found {len(mismatches)} mismatches!")
    print(mismatches[['Classification Image', 'Segmentation Image', 'Lesion Type']])
else:
    print(f"\n✓ ALL {len(df)} image pairs have matching MD5 hashes")

# Display first 10 examples for manual verification
print("\n" + "="*100)
print("FIRST 10 EXAMPLES (for manual spot-checking)")
print("="*100)
print("\nYou can open these pairs side-by-side to visually confirm they're identical:\n")

for i, row in df.head(10).iterrows():
    print(f"\n{i+1}. {row['Lesion Type'].upper()} - {row['Label']}")
    print(f"   Classification: {row['Classification Path']}")
    print(f"   Segmentation:   {row['Segmentation Path']}")
    print(f"   Hash: {row['Classification Hash'][:16]}... (first 16 chars)")
    print(f"   Status: {row['Status']}")

print("\n" + "="*100)
print("\nWHY NOT ALL 300 SEGMENTATION IMAGES?")
print("="*100)
print("""
The segmentation dataset has 300 images across 3 lesion types:
- 100 Optic Disc images
- 100 Vessels images  
- 100 Ridge images

BUT: Ridge/1.png ≠ Optic_Disc/1.png ≠ Vessels/1.png
(Same filename, DIFFERENT images)

So the segmentation dataset actually has ~100 UNIQUE images, each annotated 
for different lesion types, NOT 300 unique images.

The classification dataset has 200 images (100 Normal + 100 ROP).

MD5 matching found only 97 of those 200 classification images exist in 
the ~100 unique segmentation images.

This is why we have 97 overlapping images, not 300.
""")

print("\n" + "="*100)
print("VERIFICATION INSTRUCTIONS")
print("="*100)
print("""
To manually verify a few examples:

1. Open results/md5_hash_verification.csv in Excel
2. Pick any row (e.g., row 5)
3. Open both the Classification Path and Segmentation Path images
4. Visually compare them side-by-side
5. They should be PIXEL-PERFECT IDENTICAL (same image, just different folders)

If you find ANY mismatch, the MD5 hashes would be different and 
Status would show "✗ MISMATCH".
""")
