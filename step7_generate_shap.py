"""
Step 7: Generate SHAP Explanations for ROP Detection
Using GradientSHAP (optimized for CNNs) on 97 overlapping images

Based on literature:
- Zhang et al. (2025): SHAP on ResNet-50 for diabetic retinopathy
- BenchXAI (2025): GradientSHAP for biomedical images
- Expected IoU: 0.15-0.25
- Expected computation: 10-100× slower than Grad-CAM
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import json
import shap
from pathlib import Path
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
MODEL_PATH = 'models/augmented_best_model.pth'
OVERLAP_MAP_PATH = 'results/overlapping_images_map.json'
OUTPUT_DIR = Path('results/shap_visualizations')
HEATMAP_DIR = OUTPUT_DIR / 'heatmaps'
RESULTS_FILE = OUTPUT_DIR / 'shap_results.json'

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

def load_model():
    """Load trained ResNet50 model"""
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # Binary classification
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from {MODEL_PATH}")
    return model

def preprocess_image(image_path):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')
    img_original = np.array(img)
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor, img_original

def normalize_to_0_1(array):
    """Normalize array to [0, 1] range"""
    min_val = array.min()
    max_val = array.max()
    if max_val - min_val == 0:
        return np.zeros_like(array)
    return (array - min_val) / (max_val - min_val)

def generate_shap_explanation(model, img_tensor, target_class):
    """
    Generate SHAP explanation using GradientExplainer
    
    GradientExplainer (GradientSHAP) is optimized for deep learning models:
    - Uses integrated gradients with reference baseline
    - Specifically designed for CNNs
    - More efficient than KernelSHAP for image data
    """
    # Create background dataset (using mean image as baseline)
    # In practice, using 50-100 background samples is recommended
    # For efficiency, we'll use a single baseline (zero image)
    background = torch.zeros(1, 3, 224, 224).to(device)
    
    # Initialize GradientExplainer
    explainer = shap.GradientExplainer(model, background)
    
    # Generate SHAP values
    # This computes attribution for each input pixel
    shap_values = explainer.shap_values(img_tensor)
    
    # shap_values shape: [batch, num_classes, channels, height, width]
    # We want the target class (ROP or Normal)
    if isinstance(shap_values, list):
        shap_array = shap_values[target_class][0]  # [C, H, W]
    else:
        shap_array = shap_values[0, target_class]  # [C, H, W]
    
    # Aggregate across RGB channels (take absolute value for magnitude)
    # Sum absolute SHAP values across channels
    shap_map = np.abs(shap_array).sum(axis=0)  # [H, W]
    
    # Normalize to [0, 1]
    shap_map_norm = normalize_to_0_1(shap_map)
    
    return shap_map_norm

def resize_to_original(heatmap, original_shape):
    """Resize heatmap to match original image size"""
    h, w = original_shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    return heatmap_resized

def create_visualization(img_original, shap_map, mask, output_path):
    """Create 4-panel visualization: Original | SHAP Heatmap | Overlay | Expert Mask"""
    h, w = img_original.shape[:2]
    
    # Resize SHAP map to match original image
    shap_resized = resize_to_original(shap_map, img_original.shape)
    
    # Create heatmap visualization
    shap_colored = cv2.applyColorMap((shap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    shap_colored = cv2.cvtColor(shap_colored, cv2.COLOR_BGR2RGB)
    
    # Create overlay
    overlay = cv2.addWeighted(img_original, 0.6, shap_colored, 0.4, 0)
    
    # Prepare mask visualization
    if len(mask.shape) == 2:
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    else:
        mask_vis = mask
    
    # Resize all to same height for concatenation
    target_h = 300
    target_w = int(w * (target_h / h))
    
    img_resized = cv2.resize(img_original, (target_w, target_h))
    shap_resized_vis = cv2.resize(shap_colored, (target_w, target_h))
    overlay_resized = cv2.resize(overlay, (target_w, target_h))
    mask_resized = cv2.resize(mask_vis, (target_w, target_h))
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_resized, 'Original', (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(shap_resized_vis, 'SHAP Heatmap', (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(overlay_resized, 'SHAP Overlay', (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(mask_resized, 'Expert Mask', (10, 30), font, 0.8, (255, 255, 255), 2)
    
    # Concatenate horizontally
    combined = np.hstack([img_resized, shap_resized_vis, overlay_resized, mask_resized])
    
    # Save
    cv2.imwrite(str(output_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

def compute_iou_dice(shap_map, mask, threshold=0.5):
    """
    Compute IoU and Dice coefficient between SHAP map and expert mask
    
    Args:
        shap_map: SHAP attribution map (continuous [0, 1])
        mask: Expert binary mask (0 or 255)
        threshold: Threshold to binarize SHAP map
    """
    # Ensure mask is binary
    mask_binary = (mask > 127).astype(np.float32)
    
    # Binarize SHAP map
    shap_binary = (shap_map > threshold).astype(np.float32)
    
    # Compute intersection and union
    intersection = np.logical_and(shap_binary, mask_binary).sum()
    union = np.logical_or(shap_binary, mask_binary).sum()
    
    # Compute metrics
    iou = intersection / union if union > 0 else 0.0
    dice = (2 * intersection) / (shap_binary.sum() + mask_binary.sum()) if (shap_binary.sum() + mask_binary.sum()) > 0 else 0.0
    
    return float(iou), float(dice), float(threshold)

def find_optimal_threshold(shap_map, mask):
    """Find threshold that maximizes Dice coefficient"""
    best_dice = 0.0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 0.9, 0.1):
        iou, dice, _ = compute_iou_dice(shap_map, mask, threshold)
        if dice > best_dice:
            best_dice = dice
            best_threshold = threshold
    
    return best_threshold

def main():
    print("=" * 80)
    print("STEP 7: SHAP EXPLANATION GENERATION")
    print("=" * 80)
    
    # Load model
    model = load_model()
    
    # Load overlapping images map
    with open(OVERLAP_MAP_PATH, 'r') as f:
        overlap_data = json.load(f)
    
    overlapping_images = overlap_data['images']
    total_images = len(overlapping_images)
    
    print(f"\n✓ Found {total_images} overlapping images")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print(f"\nGenerating SHAP explanations...")
    print("⚠️  Note: SHAP is computationally intensive (10-100× slower than Grad-CAM)")
    print("    Expected time: 5-10 minutes per image\n")
    
    # Results storage
    results = {
        'total_images': 0,
        'computation_time_total': 0.0,
        'computation_time_per_image_avg': 0.0,
        'lesion_types': {},
        'images': []
    }
    
    # Process each image
    lesion_times = {'optic_disc': [], 'vessels': [], 'ridge': []}
    
    for img_data in tqdm(overlapping_images, desc="Processing images"):
        class_label = img_data['classification_label']
        image_path = Path(img_data['classification_path'])
        
        if not image_path.exists():
            print(f"\n⚠ Warning: Image not found: {image_path}")
            continue
        
        img_name = image_path.name
        
        try:
            # Load and preprocess image once for all lesion types
            img_tensor, img_original = preprocess_image(str(image_path))
            
            # Get model prediction once
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
            
            # Generate SHAP explanation once for predicted class
            start_shap = time.time()
            shap_map = generate_shap_explanation(model, img_tensor, predicted_class)
            shap_time = time.time() - start_shap
            
            # Save heatmap once
            heatmap_file = HEATMAP_DIR / f"{img_name.replace('.png', '')}.npy"
            np.save(heatmap_file, shap_map)
            
            # Process each available lesion type for this image
            for lesion_type in img_data['available_lesions']:
                mask_path = img_data['mask_paths'][lesion_type]
                
                # Start timing for this lesion
                start_time = time.time()
                
                # Load expert mask
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"\nWarning: Could not load mask for {img_name} - {lesion_type}")
                    continue
                
                # Resize SHAP map to match mask
                shap_map_resized = resize_to_original(shap_map, mask.shape)
                
                # Find optimal threshold
                optimal_threshold = find_optimal_threshold(shap_map_resized, mask)
                
                # Compute IoU and Dice with optimal threshold
                iou, dice, threshold = compute_iou_dice(shap_map_resized, mask, optimal_threshold)
                
                # End timing
                end_time = time.time()
                computation_time = end_time - start_time + (shap_time / len(img_data['available_lesions']))
                lesion_times[lesion_type].append(computation_time)
                
                # Create visualization
                output_subdir = OUTPUT_DIR / lesion_type
                output_subdir.mkdir(exist_ok=True)
                viz_path = output_subdir / f"{img_name.replace('.png', '')}_{lesion_type}.png"
                create_visualization(img_original, shap_map, mask, viz_path)
                
                # Store results
                image_result = {
                    'image_name': img_name,
                    'lesion_type': lesion_type,
                    'class_label': class_label,
                    'predicted_class': 'ROP' if predicted_class == 1 else 'Normal',
                    'confidence': float(confidence),
                    'iou': iou,
                    'dice': dice,
                    'optimal_threshold': optimal_threshold,
                    'computation_time_seconds': computation_time,
                    'heatmap_path': str(heatmap_file),
                    'visualization_path': str(viz_path)
                }
                
                results['images'].append(image_result)
            
        except Exception as e:
            print(f"\nError processing {img_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compute summary statistics
    results['total_images'] = len(results['images'])
    results['computation_time_total'] = sum([img['computation_time_seconds'] for img in results['images']])
    results['computation_time_per_image_avg'] = results['computation_time_total'] / results['total_images'] if results['total_images'] > 0 else 0.0
    
    # Lesion-specific statistics
    for lesion in ['optic_disc', 'vessels', 'ridge']:
        lesion_results = [img for img in results['images'] if img['lesion_type'] == lesion]
        if lesion_results:
            results['lesion_types'][lesion] = {
                'count': len(lesion_results),
                'mean_iou': float(np.mean([img['iou'] for img in lesion_results])),
                'std_iou': float(np.std([img['iou'] for img in lesion_results])),
                'mean_dice': float(np.mean([img['dice'] for img in lesion_results])),
                'std_dice': float(np.std([img['dice'] for img in lesion_results])),
                'mean_computation_time': float(np.mean(lesion_times[lesion])),
                'std_computation_time': float(np.std(lesion_times[lesion]))
            }
    
    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SHAP GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\n✓ Processed: {results['total_images']} images")
    print(f"✓ Total computation time: {results['computation_time_total']:.1f} seconds ({results['computation_time_total']/60:.1f} minutes)")
    print(f"✓ Average time per image: {results['computation_time_per_image_avg']:.1f} seconds")
    
    print("\nLesion-specific results:")
    for lesion, stats in results['lesion_types'].items():
        print(f"\n  {lesion.upper()}:")
        print(f"    Count: {stats['count']}")
        print(f"    Mean IoU: {stats['mean_iou']:.4f} ± {stats['std_iou']:.4f}")
        print(f"    Mean Dice: {stats['mean_dice']:.4f} ± {stats['std_dice']:.4f}")
        print(f"    Avg time: {stats['mean_computation_time']:.1f}s")
    
    print(f"\n✓ Results saved to: {RESULTS_FILE}")
    print(f"✓ Visualizations saved to: {OUTPUT_DIR}")
    print("\nNext: Run step8_generate_lime.py for LIME explanations")

if __name__ == "__main__":
    main()
