"""
Step 8: Generate LIME Explanations for ROP Detection
Using superpixel-based LIME on 97 overlapping images

Based on literature:
- Li et al. (2025): LIME on ResViT for diabetic retinopathy fundus images
- Arun et al. (2025): LIME with superpixels for medical imaging
- Expected IoU: Variable (0.10-0.20 for attention regions)
- Expected computation: Minutes per image due to perturbation sampling
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import json
from lime import lime_image
from skimage.segmentation import mark_boundaries
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
OUTPUT_DIR = Path('results/lime_visualizations')
HEATMAP_DIR = OUTPUT_DIR / 'heatmaps'
RESULTS_FILE = OUTPUT_DIR / 'lime_results.json'

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

def batch_predict(model, images):
    """
    Prediction function for LIME
    
    Args:
        model: PyTorch model
        images: numpy array of images [batch_size, H, W, C] in [0, 255] range
    
    Returns:
        probabilities: numpy array [batch_size, num_classes]
    """
    batch = []
    for img in images:
        # Convert to PIL and apply transforms
        pil_img = Image.fromarray(img.astype('uint8'))
        tensor = transform(pil_img)
        batch.append(tensor)
    
    batch_tensor = torch.stack(batch).to(device)
    
    with torch.no_grad():
        outputs = model(batch_tensor)
        probabilities = torch.softmax(outputs, dim=1)
    
    return probabilities.cpu().numpy()

def generate_lime_explanation(model, img_array, target_class, num_samples=1000):
    """
    Generate LIME explanation using superpixel segmentation
    
    Args:
        model: PyTorch model
        img_array: numpy array [H, W, C] in [0, 255] range
        target_class: target class for explanation (0 or 1)
        num_samples: number of perturbed samples (higher = more accurate but slower)
    
    Returns:
        lime_map: attribution heatmap [H, W]
    """
    # Initialize LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Generate explanation
    # num_samples: more samples = more accurate but slower (default 1000)
    # num_features: number of superpixels to highlight (default 5)
    # hide_color: color for hidden regions (0 = black)
    explanation = explainer.explain_instance(
        img_array,
        lambda x: batch_predict(model, x),
        top_labels=2,
        hide_color=0,
        num_samples=num_samples,
        num_features=10,  # Top 10 superpixels
        random_seed=42
    )
    
    # Get explanation for target class
    # explanation.local_exp contains superpixel importance scores
    temp, mask = explanation.get_image_and_mask(
        target_class,
        positive_only=True,
        num_features=10,
        hide_rest=False
    )
    
    # Create heatmap from superpixel importance
    # mask is binary (1 for important superpixels, 0 for others)
    # We'll create a smoother heatmap based on superpixel weights
    
    # Get segments (superpixels)
    segments = explanation.segments
    
    # Get weights for each segment
    weights = dict(explanation.local_exp[target_class])
    
    # Create heatmap
    lime_map = np.zeros((img_array.shape[0], img_array.shape[1]))
    
    for segment_id, weight in weights.items():
        if weight > 0:  # Only positive contributions
            lime_map[segments == segment_id] = weight
    
    # Normalize to [0, 1]
    if lime_map.max() > 0:
        lime_map = lime_map / lime_map.max()
    
    return lime_map

def resize_to_original(heatmap, original_shape):
    """Resize heatmap to match original image size"""
    h, w = original_shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    return heatmap_resized

def create_visualization(img_original, lime_map, mask, output_path):
    """Create 4-panel visualization: Original | LIME Heatmap | Overlay | Expert Mask"""
    h, w = img_original.shape[:2]
    
    # Resize LIME map to match original image
    lime_resized = resize_to_original(lime_map, img_original.shape)
    
    # Create heatmap visualization
    lime_colored = cv2.applyColorMap((lime_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    lime_colored = cv2.cvtColor(lime_colored, cv2.COLOR_BGR2RGB)
    
    # Create overlay
    overlay = cv2.addWeighted(img_original, 0.6, lime_colored, 0.4, 0)
    
    # Prepare mask visualization
    if len(mask.shape) == 2:
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    else:
        mask_vis = mask
    
    # Resize all to same height for concatenation
    target_h = 300
    target_w = int(w * (target_h / h))
    
    img_resized = cv2.resize(img_original, (target_w, target_h))
    lime_resized_vis = cv2.resize(lime_colored, (target_w, target_h))
    overlay_resized = cv2.resize(overlay, (target_w, target_h))
    mask_resized = cv2.resize(mask_vis, (target_w, target_h))
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_resized, 'Original', (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(lime_resized_vis, 'LIME Heatmap', (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(overlay_resized, 'LIME Overlay', (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(mask_resized, 'Expert Mask', (10, 30), font, 0.8, (255, 255, 255), 2)
    
    # Concatenate horizontally
    combined = np.hstack([img_resized, lime_resized_vis, overlay_resized, mask_resized])
    
    # Save
    cv2.imwrite(str(output_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

def compute_iou_dice(lime_map, mask, threshold=0.5):
    """
    Compute IoU and Dice coefficient between LIME map and expert mask
    
    Args:
        lime_map: LIME attribution map (continuous [0, 1])
        mask: Expert binary mask (0 or 255)
        threshold: Threshold to binarize LIME map
    """
    # Ensure mask is binary
    mask_binary = (mask > 127).astype(np.float32)
    
    # Binarize LIME map
    lime_binary = (lime_map > threshold).astype(np.float32)
    
    # Compute intersection and union
    intersection = np.logical_and(lime_binary, mask_binary).sum()
    union = np.logical_or(lime_binary, mask_binary).sum()
    
    # Compute metrics
    iou = intersection / union if union > 0 else 0.0
    dice = (2 * intersection) / (lime_binary.sum() + mask_binary.sum()) if (lime_binary.sum() + mask_binary.sum()) > 0 else 0.0
    
    return float(iou), float(dice), float(threshold)

def find_optimal_threshold(lime_map, mask):
    """Find threshold that maximizes Dice coefficient"""
    best_dice = 0.0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 0.9, 0.1):
        iou, dice, _ = compute_iou_dice(lime_map, mask, threshold)
        if dice > best_dice:
            best_dice = dice
            best_threshold = threshold
    
    return best_threshold

def main():
    print("=" * 80)
    print("STEP 8: LIME EXPLANATION GENERATION")
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
    print(f"\nGenerating LIME explanations...")
    print("⚠️  Note: LIME uses perturbation sampling (computationally intensive)")
    print("    Expected time: 1-3 minutes per image\n")
    
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
            # Load image once (LIME expects numpy array in [0, 255] range)
            img_pil = Image.open(str(image_path)).convert('RGB')
            img_pil_resized = img_pil.resize((224, 224))
            img_array = np.array(img_pil_resized)  # [224, 224, 3] in [0, 255]
            img_original = np.array(img_pil)
            
            # Get model prediction once
            img_tensor = transform(img_pil_resized).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
            
            # Generate LIME explanation once for predicted class
            start_lime = time.time()
            lime_map = generate_lime_explanation(model, img_array, predicted_class, num_samples=1000)
            lime_time = time.time() - start_lime
            
            # Save heatmap once
            heatmap_file = HEATMAP_DIR / f"{img_name.replace('.png', '')}.npy"
            np.save(heatmap_file, lime_map)
            
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
                
                # Resize LIME map to match mask
                lime_map_resized = resize_to_original(lime_map, mask.shape)
                
                # Find optimal threshold
                optimal_threshold = find_optimal_threshold(lime_map_resized, mask)
                
                # Compute IoU and Dice with optimal threshold
                iou, dice, threshold = compute_iou_dice(lime_map_resized, mask, optimal_threshold)
                
                # End timing
                end_time = time.time()
                computation_time = end_time - start_time + (lime_time / len(img_data['available_lesions']))
                lesion_times[lesion_type].append(computation_time)
                
                # Create visualization
                output_subdir = OUTPUT_DIR / lesion_type
                output_subdir.mkdir(exist_ok=True)
                viz_path = output_subdir / f"{img_name.replace('.png', '')}_{lesion_type}.png"
                create_visualization(img_original, lime_map, mask, viz_path)
                
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
    print("LIME GENERATION COMPLETE!")
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
    print("\nNext: Run step9_compare_xai_methods.py for comprehensive comparison")

if __name__ == "__main__":
    main()
