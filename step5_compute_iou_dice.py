"""
IoU/Dice Computation for Quantitative XAI Validation
Compares Grad-CAM heatmaps with expert segmentation masks

Research Purpose:
- Quantitatively measure how well Grad-CAM explanations align with expert annotations
- Compute IoU (Intersection over Union) and Dice coefficient metrics
- Per-lesion analysis (Ridge vs Optic Disc vs Vessels)
- Threshold optimization to maximize overlap
"""

import numpy as np
import json
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def compute_iou(pred_mask, gt_mask):
    """
    Compute Intersection over Union (IoU)
    
    IoU = |A ∩ B| / |A ∪ B|
    
    Args:
        pred_mask: Binary predicted mask (Grad-CAM)
        gt_mask: Binary ground truth mask (Expert)
    
    Returns:
        iou: IoU score [0, 1], where 1 is perfect overlap
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return iou


def compute_dice(pred_mask, gt_mask):
    """
    Compute Dice coefficient (F1 score for segmentation)
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Args:
        pred_mask: Binary predicted mask (Grad-CAM)
        gt_mask: Binary ground truth mask (Expert)
    
    Returns:
        dice: Dice coefficient [0, 1], where 1 is perfect overlap
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    
    if pred_mask.sum() + gt_mask.sum() == 0:
        return 0.0
    
    dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum())
    return dice


def threshold_heatmap(heatmap, threshold=0.5):
    """
    Convert continuous Grad-CAM heatmap to binary mask
    
    Args:
        heatmap: Continuous heatmap [0, 1]
        threshold: Threshold value (pixels > threshold = 1, else 0)
    
    Returns:
        binary_mask: Binary mask
    """
    return (heatmap > threshold).astype(np.uint8)


def find_optimal_threshold(heatmap, gt_mask, thresholds=np.arange(0.1, 0.9, 0.05)):
    """
    Find threshold that maximizes IoU
    
    Args:
        heatmap: Grad-CAM heatmap [0, 1]
        gt_mask: Ground truth binary mask
        thresholds: List of thresholds to test
    
    Returns:
        best_threshold: Threshold with highest IoU
        best_iou: Maximum IoU achieved
    """
    best_iou = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        pred_mask = threshold_heatmap(heatmap, thresh)
        iou = compute_iou(pred_mask, gt_mask)
        
        if iou > best_iou:
            best_iou = iou
            best_threshold = thresh
    
    return best_threshold, best_iou


def load_mask(mask_path, target_size=(224, 224)):
    """Load and preprocess expert segmentation mask"""
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize(target_size, Image.NEAREST)  # Use NEAREST to preserve binary
    mask = np.array(mask)
    
    # Binarize (some masks might not be pure binary)
    mask = (mask > 127).astype(np.uint8)
    
    return mask


def main():
    print("=" * 80)
    print("IoU/DICE COMPUTATION: QUANTITATIVE XAI VALIDATION")
    print("=" * 80)
    
    # Load Grad-CAM results
    with open('results/gradcam_visualizations/gradcam_results.json', 'r') as f:
        gradcam_results = json.load(f)
    
    print(f"\n✓ Loaded Grad-CAM results for {len(gradcam_results)} images")
    
    # Storage for results
    all_results = []
    per_lesion_results = {
        'optic_disc': [],
        'vessels': [],
        'ridge': []
    }
    
    # Process each image
    print("\n⏳ Computing IoU/Dice metrics...")
    
    for img_name, img_data in tqdm(gradcam_results.items(), desc="Processing images"):
        # Load Grad-CAM heatmap
        heatmap_path = Path(img_data['heatmap_path'])
        
        if not heatmap_path.exists():
            print(f"⚠ Warning: Heatmap not found: {heatmap_path}")
            continue
        
        heatmap = np.load(heatmap_path)
        
        # Process each available lesion for this image
        for lesion_type in img_data['available_lesions']:
            # Load expert mask
            mask_path = Path(img_data['mask_paths'][lesion_type])
            
            if not mask_path.exists():
                print(f"⚠ Warning: Mask not found: {mask_path}")
                continue
            
            gt_mask = load_mask(mask_path)
            
            # Find optimal threshold for this image
            optimal_threshold, max_iou = find_optimal_threshold(heatmap, gt_mask)
            
            # Compute metrics with optimal threshold
            pred_mask = threshold_heatmap(heatmap, optimal_threshold)
            iou = compute_iou(pred_mask, gt_mask)
            dice = compute_dice(pred_mask, gt_mask)
            
            # Also compute with fixed threshold (0.3) for comparison
            pred_mask_fixed = threshold_heatmap(heatmap, 0.3)
            iou_fixed = compute_iou(pred_mask_fixed, gt_mask)
            dice_fixed = compute_dice(pred_mask_fixed, gt_mask)
            
            # Store results
            result = {
                'image_name': img_name,
                'class': img_data['class'],
                'lesion_type': lesion_type,
                'iou_optimal': float(iou),
                'dice_optimal': float(dice),
                'optimal_threshold': float(optimal_threshold),
                'iou_fixed_0.3': float(iou_fixed),
                'dice_fixed_0.3': float(dice_fixed),
                'mask_area': int(gt_mask.sum()),
                'heatmap_mean': float(heatmap.mean()),
                'heatmap_max': float(heatmap.max())
            }
            
            all_results.append(result)
            per_lesion_results[lesion_type].append(result)
    
    print(f"\n✓ Computed metrics for {len(all_results)} image-lesion pairs")
    
    # ========================================================================
    # ANALYSIS & STATISTICS
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("QUANTITATIVE VALIDATION RESULTS")
    print("=" * 80)
    
    # Overall statistics
    all_iou = [r['iou_optimal'] for r in all_results]
    all_dice = [r['dice_optimal'] for r in all_results]
    
    print("\n📊 OVERALL PERFORMANCE (Optimal Threshold):")
    print("-" * 80)
    print(f"  IoU:  Mean = {np.mean(all_iou):.4f}, Median = {np.median(all_iou):.4f}, Std = {np.std(all_iou):.4f}")
    print(f"  Dice: Mean = {np.mean(all_dice):.4f}, Median = {np.median(all_dice):.4f}, Std = {np.std(all_dice):.4f}")
    print(f"  Images with IoU ≥ 0.50: {sum(1 for x in all_iou if x >= 0.5)} / {len(all_iou)} ({100*sum(1 for x in all_iou if x >= 0.5)/len(all_iou):.1f}%)")
    print(f"  Images with Dice ≥ 0.60: {sum(1 for x in all_dice if x >= 0.6)} / {len(all_dice)} ({100*sum(1 for x in all_dice if x >= 0.6)/len(all_dice):.1f}%)")
    
    # Per-lesion statistics
    print("\n📊 PER-LESION PERFORMANCE:")
    print("-" * 80)
    
    lesion_stats = {}
    for lesion_type, results in per_lesion_results.items():
        if not results:
            continue
        
        iou_vals = [r['iou_optimal'] for r in results]
        dice_vals = [r['dice_optimal'] for r in results]
        
        lesion_stats[lesion_type] = {
            'count': len(results),
            'iou_mean': np.mean(iou_vals),
            'iou_std': np.std(iou_vals),
            'dice_mean': np.mean(dice_vals),
            'dice_std': np.std(dice_vals)
        }
        
        print(f"\n{lesion_type.replace('_', ' ').title()} (n={len(results)}):")
        print(f"  IoU:  {np.mean(iou_vals):.4f} ± {np.std(iou_vals):.4f}")
        print(f"  Dice: {np.mean(dice_vals):.4f} ± {np.std(dice_vals):.4f}")
    
    # Statistical comparison (ANOVA)
    print("\n📊 STATISTICAL COMPARISON (ANOVA):")
    print("-" * 80)
    
    if len(per_lesion_results) >= 2:
        lesion_types = [k for k in per_lesion_results.keys() if per_lesion_results[k]]
        
        # IoU comparison
        iou_groups = [[r['iou_optimal'] for r in per_lesion_results[lt]] for lt in lesion_types if per_lesion_results[lt]]
        if len(iou_groups) >= 2 and all(len(g) > 0 for g in iou_groups):
            f_stat, p_value = stats.f_oneway(*iou_groups)
            print(f"\nIoU across lesion types:")
            print(f"  F-statistic: {f_stat:.4f}")
            print(f"  P-value: {p_value:.4f}")
            if p_value < 0.05:
                print(f"  ✓ SIGNIFICANT difference between lesion types (p < 0.05)")
            else:
                print(f"  ✗ No significant difference (p ≥ 0.05)")
        
        # Dice comparison
        dice_groups = [[r['dice_optimal'] for r in per_lesion_results[lt]] for lt in lesion_types if per_lesion_results[lt]]
        if len(dice_groups) >= 2 and all(len(g) > 0 for g in dice_groups):
            f_stat, p_value = stats.f_oneway(*dice_groups)
            print(f"\nDice across lesion types:")
            print(f"  F-statistic: {f_stat:.4f}")
            print(f"  P-value: {p_value:.4f}")
            if p_value < 0.05:
                print(f"  ✓ SIGNIFICANT difference between lesion types (p < 0.05)")
            else:
                print(f"  ✗ No significant difference (p ≥ 0.05)")
    
    # Class comparison (Normal vs ROP)
    print("\n📊 NORMAL vs ROP COMPARISON:")
    print("-" * 80)
    
    normal_results = [r for r in all_results if r['class'] == 'Normal']
    rop_results = [r for r in all_results if r['class'] == 'ROP']
    
    if normal_results and rop_results:
        normal_iou = [r['iou_optimal'] for r in normal_results]
        rop_iou = [r['iou_optimal'] for r in rop_results]
        
        print(f"\nNormal images (n={len(normal_results)}):")
        print(f"  IoU: {np.mean(normal_iou):.4f} ± {np.std(normal_iou):.4f}")
        
        print(f"\nROP images (n={len(rop_results)}):")
        print(f"  IoU: {np.mean(rop_iou):.4f} ± {np.std(rop_iou):.4f}")
        
        # T-test
        t_stat, p_value = stats.ttest_ind(normal_iou, rop_iou)
        print(f"\nT-test:")
        print(f"  T-statistic: {t_stat:.4f}")
        print(f"  P-value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"  ✓ SIGNIFICANT difference (p < 0.05)")
        else:
            print(f"  ✗ No significant difference (p ≥ 0.05)")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    output_dir = Path('results/iou_dice_metrics')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    with open(output_dir / 'detailed_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save summary statistics
    summary = {
        'overall': {
            'count': len(all_results),
            'iou_mean': float(np.mean(all_iou)),
            'iou_median': float(np.median(all_iou)),
            'iou_std': float(np.std(all_iou)),
            'dice_mean': float(np.mean(all_dice)),
            'dice_median': float(np.median(all_dice)),
            'dice_std': float(np.std(all_dice)),
            'iou_above_0.5': sum(1 for x in all_iou if x >= 0.5),
            'dice_above_0.6': sum(1 for x in all_dice if x >= 0.6)
        },
        'per_lesion': lesion_stats
    }
    
    with open(output_dir / 'summary_statistics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir}")
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    
    print("\n⏳ Creating visualizations...")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. IoU distribution by lesion type
    ax1 = plt.subplot(2, 3, 1)
    lesion_names = []
    lesion_ious = []
    for lesion_type, results in per_lesion_results.items():
        if results:
            lesion_names.append(lesion_type.replace('_', ' ').title())
            lesion_ious.append([r['iou_optimal'] for r in results])
    
    bp = ax1.boxplot(lesion_ious, labels=lesion_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax1.axhline(y=0.5, color='r', linestyle='--', label='IoU ≥ 0.5 target')
    ax1.set_ylabel('IoU Score', fontweight='bold')
    ax1.set_title('IoU Distribution by Lesion Type', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Dice distribution by lesion type
    ax2 = plt.subplot(2, 3, 2)
    lesion_dices = []
    for lesion_type, results in per_lesion_results.items():
        if results:
            lesion_dices.append([r['dice_optimal'] for r in results])
    
    bp = ax2.boxplot(lesion_dices, labels=lesion_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
    ax2.axhline(y=0.6, color='r', linestyle='--', label='Dice ≥ 0.6 target')
    ax2.set_ylabel('Dice Score', fontweight='bold')
    ax2.set_title('Dice Distribution by Lesion Type', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Normal vs ROP comparison
    ax3 = plt.subplot(2, 3, 3)
    if normal_results and rop_results:
        data = [normal_iou, rop_iou]
        bp = ax3.boxplot(data, labels=['Normal', 'ROP'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('salmon')
        ax3.set_ylabel('IoU Score', fontweight='bold')
        ax3.set_title('Normal vs ROP Comparison', fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # 4. IoU histogram
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(all_iou, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax4.axvline(x=0.5, color='r', linestyle='--', linewidth=2, label='Target: 0.5')
    ax4.axvline(x=np.mean(all_iou), color='g', linestyle='-', linewidth=2, label=f'Mean: {np.mean(all_iou):.3f}')
    ax4.set_xlabel('IoU Score', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('IoU Score Distribution', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Dice histogram
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(all_dice, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    ax5.axvline(x=0.6, color='r', linestyle='--', linewidth=2, label='Target: 0.6')
    ax5.axvline(x=np.mean(all_dice), color='g', linestyle='-', linewidth=2, label=f'Mean: {np.mean(all_dice):.3f}')
    ax5.set_xlabel('Dice Score', fontweight='bold')
    ax5.set_ylabel('Frequency', fontweight='bold')
    ax5.set_title('Dice Score Distribution', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Optimal threshold distribution
    ax6 = plt.subplot(2, 3, 6)
    thresholds = [r['optimal_threshold'] for r in all_results]
    ax6.hist(thresholds, bins=20, color='lightyellow', edgecolor='black', alpha=0.7)
    ax6.axvline(x=np.mean(thresholds), color='r', linestyle='-', linewidth=2, label=f'Mean: {np.mean(thresholds):.3f}')
    ax6.set_xlabel('Optimal Threshold', fontweight='bold')
    ax6.set_ylabel('Frequency', fontweight='bold')
    ax6.set_title('Optimal Threshold Distribution', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Quantitative XAI Validation: IoU/Dice Metrics Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_dir / 'iou_dice_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_dir / 'iou_dice_analysis.png'}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("QUANTITATIVE VALIDATION COMPLETE")
    print("=" * 80)
    
    print(f"\n✅ Key Findings:")
    print(f"  • Processed {len(all_results)} image-lesion pairs")
    print(f"  • Overall IoU: {np.mean(all_iou):.4f} ± {np.std(all_iou):.4f}")
    print(f"  • Overall Dice: {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    print(f"  • {sum(1 for x in all_iou if x >= 0.5)}/{len(all_iou)} ({100*sum(1 for x in all_iou if x >= 0.5)/len(all_iou):.1f}%) meet IoU ≥ 0.5 target")
    print(f"  • {sum(1 for x in all_dice if x >= 0.6)}/{len(all_dice)} ({100*sum(1 for x in all_dice if x >= 0.6)/len(all_dice):.1f}%) meet Dice ≥ 0.6 target")
    
    print(f"\n📁 Output Files:")
    print(f"  • Detailed results: {output_dir / 'detailed_results.json'}")
    print(f"  • Summary statistics: {output_dir / 'summary_statistics.json'}")
    print(f"  • Visualizations: {output_dir / 'iou_dice_analysis.png'}")
    
    print(f"\n🎯 Clinical Interpretation:")
    if np.mean(all_iou) >= 0.5:
        print(f"  ✅ EXCELLENT: Mean IoU ≥ 0.5 indicates Grad-CAM explanations are clinically accurate")
    elif np.mean(all_iou) >= 0.3:
        print(f"  ⚠️  GOOD: Mean IoU ≥ 0.3 shows moderate alignment with expert knowledge")
    else:
        print(f"  ❌ POOR: Mean IoU < 0.3 suggests explanations need improvement")
    
    print(f"\n🎓 Next Step: Statistical Analysis")
    print(f"  → Compare baseline vs augmented model XAI quality")
    print(f"  → Correlation analysis (accuracy vs IoU)")
    print(f"  → Generate final research report")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
