"""
Grad-CAM Implementation for ROP Detection
Generates explainability heatmaps for quantitative validation

Research Purpose:
- Generate visual explanations showing WHERE the model looks
- Validate that model focuses on clinically relevant regions
- Enable quantitative comparison with expert segmentation masks
"""

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import json
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    
    Reference: Selvaraju et al., 2017
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    
    Why we use this:
    - Most popular XAI method for CNNs (>10,000 citations)
    - Provides class-discriminative localization
    - Doesn't require model retraining
    - Works with any CNN architecture
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: Trained ResNet50 model
            target_layer: Layer to extract gradients from (layer4 for ResNet50)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture gradients and activations
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Capture forward pass activations"""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Capture backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, input_tensor, target_class):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Preprocessed image tensor (1, 3, 224, 224)
            target_class: Class to generate heatmap for (0=Normal, 1=ROP)
        
        Returns:
            heatmap: Numpy array (224, 224) with values [0, 1]
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (512, 7, 7) for ResNet50 layer4
        activations = self.activations[0]  # (512, 7, 7)
        
        # Global average pooling of gradients
        # This gives importance weights for each feature map
        weights = gradients.mean(dim=(1, 2))  # (512,)
        
        # Weighted combination of activation maps
        heatmap = torch.zeros(activations.shape[1:])  # (7, 7)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
        
        # Apply ReLU (only positive influence)
        heatmap = F.relu(heatmap)
        
        # Normalize to [0, 1]
        heatmap = heatmap / (heatmap.max() + 1e-8)
        
        # Resize to input image size (224, 224)
        heatmap = heatmap.cpu().numpy()
        heatmap = cv2.resize(heatmap, (224, 224))
        
        return heatmap
    
    def apply_heatmap_to_image(self, image, heatmap, alpha=0.4):
        """
        Overlay heatmap on original image
        
        Args:
            image: Original PIL Image or numpy array
            heatmap: Grad-CAM heatmap (224, 224)
            alpha: Transparency of heatmap overlay
        
        Returns:
            overlay: RGB image with heatmap overlay
        """
        # Convert image to numpy if PIL
        if isinstance(image, Image.Image):
            image = np.array(image.resize((224, 224)))
        
        # Convert heatmap to colormap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = (1 - alpha) * image + alpha * heatmap_colored
        overlay = np.uint8(overlay)
        
        return overlay


def load_best_model(model_path):
    """Load the trained model"""
    # Create model architecture
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(2048, 2)  # Binary classification
    
    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    return model


def preprocess_image(image_path):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return input_tensor, image


def main():
    print("=" * 80)
    print("GRAD-CAM GENERATION FOR QUANTITATIVE XAI VALIDATION")
    print("=" * 80)
    
    # Load overlapping images map
    with open('results/overlapping_images_map.json', 'r') as f:
        data = json.load(f)
    
    overlapping_images = data['images']
    print(f"\n✓ Loaded {len(overlapping_images)} images with segmentation masks")
    
    # Count by lesion type
    lesion_counts = {'optic_disc': 0, 'vessels': 0, 'ridge': 0}
    for img_data in overlapping_images:
        for lesion in img_data['available_lesions']:
            lesion_counts[lesion] += 1
    
    print(f"\nLesion distribution:")
    for lesion, count in lesion_counts.items():
        print(f"  - {lesion}: {count} images")
    
    # Load best model (augmented model - better performance)
    print("\n⏳ Loading augmented model (92% test accuracy)...")
    model = load_best_model('models/augmented_best_model.pth')
    print("✓ Model loaded")
    
    # Initialize Grad-CAM
    # Target layer: layer4 (final residual block before fc)
    # This captures high-level semantic features
    target_layer = model.layer4[-1]  # Last block of layer4
    grad_cam = GradCAM(model, target_layer)
    print("✓ Grad-CAM initialized (target layer: layer4)")
    
    # Create output directories
    output_dir = Path('results/gradcam_visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for lesion_type in ['optic_disc', 'vessels', 'ridge']:
        (output_dir / lesion_type).mkdir(exist_ok=True)
    
    # Process each image
    print(f"\n⏳ Generating Grad-CAM heatmaps for {len(overlapping_images)} images...")
    
    results = {}
    processed = 0
    
    for img_data in tqdm(overlapping_images, desc="Processing images"):
        class_label = img_data['classification_label']
        target_class = 1 if class_label == 'ROP' else 0
        
        # Load and preprocess image
        img_path = Path(img_data['classification_path'])
        if not img_path.exists():
            print(f"⚠ Warning: Image not found: {img_path}")
            continue
        
        input_tensor, original_image = preprocess_image(img_path)
        
        # Generate Grad-CAM heatmap
        heatmap = grad_cam.generate_heatmap(input_tensor, target_class)
        
        # Create visualization
        overlay = grad_cam.apply_heatmap_to_image(original_image, heatmap, alpha=0.4)
        
        # Get image name for saving — use unique key to avoid collisions
        # e.g. Normal_Neo_Normal_1.png, ROP_Neo_ROP_1.png
        parts = img_path.parts
        # Build unique name from last 3 path components: class/subclass/filename
        img_name = "_".join(parts[-3:]).replace(" ", "_")
        
        # Save results for each available mask type
        for lesion_type in img_data['available_lesions']:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(original_image.resize((224, 224)))
            axes[0].set_title('Original Image', fontweight='bold')
            axes[0].axis('off')
            axes[1].imshow(heatmap, cmap='jet')
            axes[1].set_title('Grad-CAM Heatmap', fontweight='bold')
            axes[1].axis('off')
            axes[2].imshow(overlay)
            axes[2].set_title('Grad-CAM Overlay', fontweight='bold')
            axes[2].axis('off')
            mask_path = Path(img_data['mask_paths'][lesion_type])
            if mask_path.exists():
                expert_mask = Image.open(mask_path).convert('L').resize((224, 224))
                axes[3].imshow(expert_mask, cmap='gray')
                axes[3].set_title(f'Expert Mask ({lesion_type.replace("_", " ").title()})', fontweight='bold')
                axes[3].axis('off')
            plt.suptitle(f'{img_name} - {class_label} - {lesion_type.replace("_", " ").title()}',
                         fontsize=12, fontweight='bold')
            plt.tight_layout()
            save_name = img_name.replace(".png", "")
            save_path = output_dir / lesion_type / f'{save_name}_{lesion_type}.png'
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
        
        # Store heatmap metadata with unique key
        heatmap_dir = output_dir / 'heatmaps'
        heatmap_dir.mkdir(exist_ok=True)
        npy_name = img_name.replace('.png', '.npy')
        np.save(heatmap_dir / npy_name, heatmap)
        results[img_name] = {
            'classification_path': str(img_path),
            'class': class_label,
            'heatmap_path': str(heatmap_dir / npy_name),
            'available_lesions': img_data['available_lesions'],
            'mask_paths': img_data['mask_paths']
        }
        
        processed += 1
    
    print(f"\n✓ Processed {processed} images")
    print(f"✓ Visualizations saved to: {output_dir}")
    
    # Save results metadata
    with open(output_dir / 'gradcam_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("GRAD-CAM GENERATION COMPLETE")
    print("=" * 80)
    
    print("\nSummary:")
    print(f"  - Total images processed: {processed}")
    print(f"  - Visualizations per lesion type:")
    for lesion, count in lesion_counts.items():
        print(f"    • {lesion}: {count} visualizations")
    
    print("\n Output structure:")
    print(f"  - Visualizations: {output_dir}/[OpticDisc|Vessels|Ridge]/")
    print(f"  - Raw heatmaps: {output_dir}/heatmaps/")
    print(f"  - Metadata: {output_dir}/gradcam_results.json")
    
    print("\n Next Step: Compute IoU/Dice metrics")
    print("   → Compare Grad-CAM heatmaps with expert segmentation masks")
    print("   → Quantitatively validate XAI explanations")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
