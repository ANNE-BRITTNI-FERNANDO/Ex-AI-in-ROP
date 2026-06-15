import os
from PIL import Image
from pathlib import Path
import json
from collections import defaultdict

# Base path
base_path = r"C:\Users\ASUS\FYP - MidPoint\data\raw_data"

# Dictionary to store results
results = {
    "summary": {},
    "detailed_breakdown": {},
    "resolution_analysis": defaultdict(list),
    "issues": []
}

# Get all subdirectories
for root, dirs, files in os.walk(base_path):
    # Skip if no files
    if not files:
        continue
    
    # Get relative path for reporting
    rel_path = os.path.relpath(root, base_path)
    
    # Count files
    file_count = len(files)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    print(f"\n{'='*60}")
    print(f"Directory: {rel_path}")
    print(f"Total files: {file_count}")
    print(f"Image files: {len(image_files)}")
    
    # Store in results
    if rel_path not in results["detailed_breakdown"]:
        results["detailed_breakdown"][rel_path] = {
            "total_files": file_count,
            "image_files": len(image_files),
            "resolutions": []
        }
    
    # Check resolutions for image files
    resolutions_set = set()
    for img_file in image_files[:5]:  # Check first 5 images as sample
        img_path = os.path.join(root, img_file)
        try:
            # Use PIL to get image dimensions
            with Image.open(img_path) as img:
                width, height = img.size
                resolution = f"{width}x{height}"
                resolutions_set.add(resolution)
                results["detailed_breakdown"][rel_path]["resolutions"].append({
                    "file": img_file,
                    "resolution": resolution
                })
                print(f"  {img_file}: {resolution}")
        except Exception as e:
            print(f"  {img_file}: ERROR - {str(e)}")
            results["issues"].append(f"{os.path.join(rel_path, img_file)}: {str(e)}")
    
    # Store resolution summary for this directory
    if len(resolutions_set) > 1:
        results["resolution_analysis"][rel_path] = list(resolutions_set)
        results["issues"].append(f"MIXED RESOLUTIONS in {rel_path}: {', '.join(resolutions_set)}")
        print(f"  ⚠️  WARNING: Mixed resolutions detected: {', '.join(resolutions_set)}")
    else:
        results["resolution_analysis"][rel_path] = list(resolutions_set)
        print(f"  ✓ Uniform resolution: {list(resolutions_set)[0] if resolutions_set else 'No images'}")

# Count summary by category
print(f"\n\n{'='*60}")
print("SUMMARY BY CATEGORY")
print(f"{'='*60}")

categories = defaultdict(lambda: {"count": 0, "resolutions": set()})

for path, data in results["detailed_breakdown"].items():
    category = path.split(os.sep)[0]  # Get top-level category (classification/segmentation)
    subcategory = path.split(os.sep)[1] if len(path.split(os.sep)) > 1 else "root"
    
    key = f"{category}/{subcategory}"
    categories[key]["count"] += data["image_files"]
    
    for res_info in data["resolutions"]:
        categories[key]["resolutions"].add(res_info["resolution"])

for category, data in sorted(categories.items()):
    print(f"\n{category}: {data['count']} images")
    if data['resolutions']:
        print(f"  Resolutions: {', '.join(data['resolutions'])}")

# Save results to file
output_file = r"C:\Users\ASUS\FYP - MidPoint\results\dataset_verification_report.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Convert sets to lists for JSON serialization
results["resolution_analysis"] = {k: list(v) for k, v in results["resolution_analysis"].items()}

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n\n{'='*60}")
print(f"Report saved to: {output_file}")
print(f"{'='*60}")

# Final summary
print(f"\n\nFINAL SUMMARY:")
print(f"Total directories analyzed: {len(results['detailed_breakdown'])}")
print(f"Total image files: {sum(d['image_files'] for d in results['detailed_breakdown'].values())}")
print(f"Issues found: {len(results['issues'])}")

if results['issues']:
    print(f"\n⚠️  ISSUES DETECTED:")
    for issue in results['issues']:
        print(f"  - {issue}")
else:
    print(f"\n✓ No issues detected!")
