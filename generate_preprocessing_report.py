"""
PREPROCESSING METHODS & MODEL COMPARISON REPORT
Documents all preprocessing steps and compares baseline vs augmented models
"""

import json
from pathlib import Path
import time

print("="*80)
print("PREPROCESSING METHODS & MODEL COMPARISON")
print("="*80)

# Wait for both trainings to complete
print("\n⏳ Waiting for training to complete...")
print("This report will be generated after both models finish training.\n")

OUTPUT_DIR = Path(r"C:\Users\ASUS\FYP - MidPoint\models")
RESULTS_DIR = Path(r"C:\Users\ASUS\FYP - MidPoint\results")

# Check every 30 seconds if training is complete
max_wait = 7200  # 2 hours maximum wait
waited = 0
interval = 30

while waited < max_wait:
    baseline_done = (OUTPUT_DIR / 'baseline_results.json').exists()
    augmented_done = (OUTPUT_DIR / 'augmented_results.json').exists()
    
    if baseline_done and augmented_done:
        print("✓ Both trainings complete!\n")
        break
    
    status = []
    if not baseline_done:
        status.append("Baseline training in progress...")
    if not augmented_done:
        status.append("Augmented training in progress...")
    
    print(f"[{waited//60}m {waited%60}s] {' | '.join(status)}")
    time.sleep(interval)
    waited += interval
else:
    print("⚠ Timeout waiting for training completion. Generating partial report...")

# Load results if available
baseline_results = None
augmented_results = None

if (OUTPUT_DIR / 'baseline_results.json').exists():
    with open(OUTPUT_DIR / 'baseline_results.json') as f:
        baseline_results = json.load(f)

if (OUTPUT_DIR / 'augmented_results.json').exists():
    with open(OUTPUT_DIR / 'augmented_results.json') as f:
        augmented_results = json.load(f)

# Generate comprehensive report
report = []

report.append("="*80)
report.append("COMPREHENSIVE PREPROCESSING & COMPARISON REPORT")
report.append("Quantitative Explainable AI in Retinopathy of Prematurity")
report.append("="*80)
report.append("")

# Section 1: Dataset Overview
report.append("1. DATASET OVERVIEW")
report.append("-" * 80)
report.append("")
report.append("HVDROPDB Dataset:")
report.append("  - Total original images: 185")
report.append("  - Normal: 100 images")
report.append("  - ROP: 85 images")
report.append("  - Devices: Neonatal (Neo) and RetCam")
report.append("  - Resolutions: 2040×2040 (Neo), 640×480 (RetCam)")
report.append("")
report.append("Overlapping Images (for Grad-CAM validation):")
report.append("  - Total: 97 images with classification labels + segmentation masks")
report.append("  - Optic Disc masks: 60 images")
report.append("  - Vessel masks: 35 images")
report.append("  - Ridge masks: 16 images")
report.append("")

# Section 2: Preprocessing Methods
report.append("2. PREPROCESSING METHODS")
report.append("-" * 80)
report.append("")

report.append("2.1 BASELINE MODEL (No Augmentation):")
report.append("  Step 1: Image Loading")
report.append("    - Load original .png images from raw_data/classification/")
report.append("    - Maintain original quality")
report.append("")
report.append("  Step 2: Resizing")
report.append("    - Method: PIL.Image.resize() with LANCZOS interpolation")
report.append("    - Target resolution: 224×224 pixels")
report.append("    - Rationale: ResNet50 input requirement")
report.append("")
report.append("  Step 3: Tensor Conversion")
report.append("    - Convert PIL Image → PyTorch Tensor")
report.append("    - Normalize pixel values: [0, 255] → [0.0, 1.0]")
report.append("")
report.append("  Step 4: ImageNet Normalization")
report.append("    - Mean: [0.485, 0.456, 0.406] (R, G, B)")
report.append("    - Std:  [0.229, 0.224, 0.225] (R, G, B)")
report.append("    - Formula: normalized = (pixel - mean) / std")
report.append("    - Rationale: Transfer learning from ImageNet-pretrained ResNet50")
report.append("")
report.append("  TOTAL PREPROCESSING: Resize + Normalize ONLY")
report.append("")

report.append("2.2 AUGMENTED MODEL:")
report.append("  Step 1: Augmentation Pipeline (Albumentations library)")
report.append("    Applied augmentations (each with probability p):")
report.append("")
report.append("    A. GEOMETRIC TRANSFORMS:")
report.append("       - HorizontalFlip (p=0.5)")
report.append("         Purpose: Simulate different imaging angles")
report.append("         Reference: Sankari et al., 2023")
report.append("")
report.append("       - Rotation (±15°, p=0.7)")
report.append("         Purpose: Account for patient head position variability")
report.append("         Border mode: CONSTANT (black padding)")
report.append("")
report.append("       - ShiftScaleRotate (p=0.5)")
report.append("         Shift: ±10% of image dimensions")
report.append("         Scale: ±10%")
report.append("         Purpose: Simulate camera distance and positioning")
report.append("")
report.append("    B. PHOTOMETRIC TRANSFORMS:")
report.append("       - RandomBrightnessContrast (p=0.7)")
report.append("         Brightness: ±20%")
report.append("         Contrast: ±20%")
report.append("         Purpose: Account for lighting variations between devices")
report.append("")
report.append("       - CLAHE (Contrast Limited Adaptive Histogram Equalization, p=0.5)")
report.append("         Clip limit: 2.0")
report.append("         Tile grid: 8×8")
report.append("         Purpose: Enhance vessel contrast (Peng et al., 2021)")
report.append("         Reference: Used in ROP preprocessing")
report.append("")
report.append("       - HueSaturationValue (p=0.5)")
report.append("         Hue shift: ±10")
report.append("         Saturation: ±20")
report.append("         Value (brightness): ±10")
report.append("         Purpose: Simulate color calibration differences")
report.append("")
report.append("    C. NOISE/BLUR TRANSFORMS:")
report.append("       - GaussianBlur (p=0.3)")
report.append("         Kernel size: 3-5 pixels")
report.append("         Purpose: Simulate slight focus issues")
report.append("")
report.append("       - GaussNoise (p=0.3)")
report.append("         Variance: 10-50")
report.append("         Purpose: Simulate sensor noise")
report.append("")
report.append("  Step 2: Resize to 224×224")
report.append("    - Applied AFTER augmentation")
report.append("    - Interpolation: LANCZOS4 (high quality)")
report.append("")
report.append("  Step 3: ImageNet Normalization")
report.append("    - Same as baseline model")
report.append("")
report.append("  Step 4: Additional Training-Time Augmentation")
report.append("    - RandomHorizontalFlip (p=0.3) during training")
report.append("    - Light additional variation")
report.append("")
report.append("  AUGMENTATION FACTOR: 10-12× per class")
report.append("  - Normal: 100 → 1000 images")
report.append("  - ROP: 85 → 1000 images")
report.append("  - Total: 185 → 2000 images")
report.append("")

# Section 3: Model Architecture
report.append("3. MODEL ARCHITECTURE")
report.append("-" * 80)
report.append("")
report.append("Base Architecture: ResNet50 (He et al., 2016)")
report.append("  - Total layers: 50 (48 convolutional + 2 fully connected)")
report.append("  - Total parameters: ~25.6 million")
report.append("  - Input: 224×224×3 RGB images")
report.append("  - Pretrained on: ImageNet (1.2 million images)")
report.append("")
report.append("Transfer Learning Configuration:")
report.append("  - Frozen layers: conv1, layer1, layer2, layer3")
report.append("  - Trainable layers: layer4 (final residual block) + fc (classifier)")
report.append("  - Trainable parameters: ~14.97 million (58.4% of total)")
report.append("")
report.append("Modified Final Layer:")
report.append("  - Original fc: 2048 → 1000 classes (ImageNet)")
report.append("  - Modified fc: 2048 → 2 classes (Normal vs ROP)")
report.append("")
report.append("Rationale for Transfer Learning:")
report.append("  - Low-level features (edges, textures) transfer from natural images")
report.append("  - Reduces training time 10-20× vs training from scratch")
report.append("  - Prevents overfitting on small dataset (185 images)")
report.append("  - Reference: Feng et al., 2024 (ROP classification)")
report.append("")

# Section 4: Training Configuration
report.append("4. TRAINING CONFIGURATION")
report.append("-" * 80)
report.append("")
report.append("Hyperparameters:")
report.append("  - Optimizer: Adam")
report.append("    Learning rate (initial): 0.001")
report.append("    Beta1: 0.9, Beta2: 0.999")
report.append("    Rationale: Adaptive learning rates handle small medical datasets")
report.append("")
report.append("  - Loss Function: CrossEntropyLoss")
report.append("    Purpose: Binary classification with softmax outputs")
report.append("")
report.append("  - Batch Size: 16")
report.append("    Constraint: Maximum for 4GB RAM without GPU")
report.append("")
report.append("  - Learning Rate Scheduler: ReduceLROnPlateau")
report.append("    Factor: 0.1 (reduce LR by 10× when plateau)")
report.append("    Patience: 5 epochs")
report.append("    Monitor: Validation loss")
report.append("")
report.append("  - Early Stopping:")
report.append("    Patience: 10 epochs without val_acc improvement")
report.append("    Purpose: Prevent overfitting")
report.append("")
report.append("  - Maximum Epochs: 50")
report.append("")
report.append("Data Split (Stratified):")
report.append("  - Training: 70% (maintains class balance)")
report.append("  - Validation: 15%")
report.append("  - Test: 15% (held out for final evaluation)")
report.append("")

# Section 5: Comparison Results
if baseline_results and augmented_results:
    report.append("5. MODEL COMPARISON RESULTS")
    report.append("-" * 80)
    report.append("")
    
    report.append("5.1 BASELINE MODEL (No Augmentation):")
    report.append(f"  Training images: {baseline_results['training_images']}")
    report.append(f"  Validation accuracy: {baseline_results['best_val_acc']*100:.2f}%")
    report.append(f"  Test accuracy: {baseline_results['test_acc']*100:.2f}%")
    report.append(f"  Test loss: {baseline_results['test_loss']:.4f}")
    report.append(f"  Total epochs: {baseline_results['total_epochs']}")
    report.append(f"  Training time: {baseline_results['training_time_seconds']/60:.1f} minutes")
    report.append("")
    
    report.append("5.2 AUGMENTED MODEL:")
    report.append(f"  Training images: {augmented_results['training_images']}")
    report.append(f"  Validation accuracy: {augmented_results['best_val_acc']*100:.2f}%")
    report.append(f"  Test accuracy: {augmented_results['test_acc']*100:.2f}%")
    report.append(f"  Test loss: {augmented_results['test_loss']:.4f}")
    report.append(f"  Total epochs: {augmented_results['total_epochs']}")
    report.append(f"  Training time: {augmented_results['training_time_seconds']/60:.1f} minutes")
    report.append("")
    
    # Calculate improvements
    acc_improvement = (augmented_results['test_acc'] - baseline_results['test_acc']) * 100
    loss_improvement = (baseline_results['test_loss'] - augmented_results['test_loss']) / baseline_results['test_loss'] * 100
    
    report.append("5.3 IMPROVEMENT ANALYSIS:")
    report.append(f"  Test Accuracy Improvement: {acc_improvement:+.2f}%")
    report.append(f"  Test Loss Reduction: {loss_improvement:.2f}%")
    report.append(f"  Training Data Increase: {augmented_results['training_images'] - baseline_results['training_images']} images ({augmented_results['training_images']/baseline_results['training_images']:.1f}× larger)")
    report.append("")
    
    if acc_improvement > 0:
        report.append("  ✓ CONCLUSION: Augmentation IMPROVES model performance")
    else:
        report.append("  ⚠ OBSERVATION: Augmentation did not improve test accuracy")
        report.append("    Possible reasons: Overfitting to augmented patterns, need hyperparameter tuning")
    report.append("")

else:
    report.append("5. MODEL COMPARISON RESULTS")
    report.append("-" * 80)
    report.append("")
    report.append("⏳ Training still in progress. Results will be available after completion.")
    report.append("")

# Section 6: Literature Alignment
report.append("6. LITERATURE ALIGNMENT")
report.append("-" * 80)
report.append("")
report.append("6.1 Preprocessing Methods:")
report.append("  ✓ CLAHE: Peng et al., 2021 (ROP image enhancement)")
report.append("  ✓ Geometric augmentation: Sankari et al., 2023 (rotation, flips)")
report.append("  ✓ Photometric augmentation: Ullah et al., 2025 (Albumentations library)")
report.append("  ✓ Resize to 224×224: Feng et al., 2024 (ResNet50 ROP classification)")
report.append("")
report.append("6.2 Model Architecture:")
report.append("  ✓ ResNet50: Feng et al., 2024 (92.87% ROP accuracy)")
report.append("  ✓ Transfer learning: Huang et al., 2025 (baseline ROP model)")
report.append("  ✓ ImageNet pretraining: Sankari et al., 2023")
report.append("")
report.append("6.3 Training Strategy:")
report.append("  ✓ Adam optimizer: Feng et al., 2024; Yurdakul et al., 2025")
report.append("  ✓ Early stopping: Standard ML practice")
report.append("  ✓ ReduceLROnPlateau: PyTorch standard for medical imaging")
report.append("")

# Section 7: Next Steps
report.append("7. NEXT STEPS FOR QUANTITATIVE XAI")
report.append("-" * 80)
report.append("")
report.append("After model training completes:")
report.append("")
report.append("Step 1: Grad-CAM Implementation")
report.append("  - Generate heatmaps for 97 overlapping images")
report.append("  - Target layer: ResNet50 layer4 (final conv block)")
report.append("  - Threshold optimization: Grid search 0.2-0.4")
report.append("")
report.append("Step 2: Quantitative Validation")
report.append("  - Compute IoU (Intersection over Union)")
report.append("  - Compute Dice coefficient")
report.append("  - Compare against expert segmentation masks")
report.append("  - Target: IoU ≥0.50, Dice ≥0.60")
report.append("")
report.append("Step 3: Statistical Analysis")
report.append("  Experiment 1: Baseline Grad-CAM performance")
report.append("  Experiment 2: Lesion-type comparison (Ridge vs OD vs Vessels)")
report.append("  Experiment 3: Augmentation impact on XAI (THIS comparison)")
report.append("  Experiment 4: Accuracy-IoU correlation")
report.append("")

report.append("="*80)

# Save report
report_text = "\n".join(report)
print(report_text)

# Save to file
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
report_path = RESULTS_DIR / "preprocessing_comparison_report.txt"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"\n✓ Full report saved to: {report_path}")

# Save JSON summary
if baseline_results and augmented_results:
    summary = {
        "baseline_model": {
            "training_images": baseline_results['training_images'],
            "test_accuracy": baseline_results['test_acc'],
            "test_loss": baseline_results['test_loss'],
            "training_time_minutes": baseline_results['training_time_seconds'] / 60
        },
        "augmented_model": {
            "training_images": augmented_results['training_images'],
            "test_accuracy": augmented_results['test_acc'],
            "test_loss": augmented_results['test_loss'],
            "training_time_minutes": augmented_results['training_time_seconds'] / 60
        },
        "improvement": {
            "accuracy_gain_percentage": acc_improvement,
            "loss_reduction_percentage": loss_improvement,
            "dataset_size_multiplier": augmented_results['training_images'] / baseline_results['training_images']
        }
    }
    
    with open(RESULTS_DIR / "model_comparison_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved to: {RESULTS_DIR / 'model_comparison_summary.json'}")

print("="*80)
