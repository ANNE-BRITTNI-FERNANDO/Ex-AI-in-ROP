"""
TRAINING PROGRESS MONITOR
Real-time comparison of baseline vs augmented model training
"""

import json
from pathlib import Path
import time
import os

OUTPUT_DIR = Path(r"C:\Users\ASUS\FYP - MidPoint\models")
RESULTS_DIR = Path(r"C:\Users\ASUS\FYP - MidPoint\results")

print("="*80)
print("TRAINING PROGRESS MONITOR")
print("="*80)
print("\n📊 MODELS CURRENTLY TRAINING:")
print("  1. Baseline Model (NO augmentation): 129 training images")
print("  2. Augmented Model (WITH augmentation): 1400 training images")
print("\n⏳ Estimated completion time: 1-2 hours on CPU")
print("\nBoth models will train until early stopping (max 50 epochs)")
print("-"*80)

# Monitor training progress
print("\n🔍 MONITORING STATUS:")
print("Checking for completed models every 60 seconds...\n")

checked_count = 0
while checked_count < 120:  # Check for up to 2 hours
    baseline_done = (OUTPUT_DIR / 'baseline_results.json').exists()
    augmented_done = (OUTPUT_DIR / 'augmented_results.json').exists()
    
    status_line = f"[Check #{checked_count + 1}] "
    
    if baseline_done and augmented_done:
        print("\n" + "="*80)
        print("✓✓✓ BOTH MODELS TRAINING COMPLETE! ✓✓✓")
        print("="*80)
        
        # Load and display results
        with open(OUTPUT_DIR / 'baseline_results.json') as f:
            baseline = json.load(f)
        
        with open(OUTPUT_DIR / 'augmented_results.json') as f:
            augmented = json.load(f)
        
        print("\n📊 FINAL RESULTS COMPARISON:")
        print("-"*80)
        print(f"\nBASELINE MODEL (No Augmentation):")
        print(f"  Training images: {baseline['training_images']}")
        print(f"  Test Accuracy:   {baseline['test_acc']*100:.2f}%")
        print(f"  Test Loss:       {baseline['test_loss']:.4f}")
        print(f"  Training time:   {baseline['training_time_seconds']/60:.1f} minutes")
        print(f"  Total epochs:    {baseline['total_epochs']}")
        
        print(f"\nAUGMENTED MODEL (With Augmentation):")
        print(f"  Training images: {augmented['training_images']}")
        print(f"  Test Accuracy:   {augmented['test_acc']*100:.2f}%")
        print(f"  Test Loss:       {augmented['test_loss']:.4f}")
        print(f"  Training time:   {augmented['training_time_seconds']/60:.1f} minutes")
        print(f"  Total epochs:    {augmented['total_epochs']}")
        
        # Calculate improvement
        acc_improvement = (augmented['test_acc'] - baseline['test_acc']) * 100
        loss_improvement = ((baseline['test_loss'] - augmented['test_loss']) / baseline['test_loss']) * 100
        
        print(f"\n📈 IMPROVEMENT METRICS:")
        print(f"  Accuracy Change:     {acc_improvement:+.2f}%")
        print(f"  Loss Reduction:      {loss_improvement:+.2f}%")
        print(f"  Dataset Size Ratio:  {augmented['training_images'] / baseline['training_images']:.1f}×")
        
        if acc_improvement > 0:
            print(f"\n✓ CONCLUSION: Augmentation IMPROVED test accuracy by {acc_improvement:.2f}%")
        elif acc_improvement < -2:
            print(f"\n⚠ OBSERVATION: Augmentation DECREASED accuracy by {abs(acc_improvement):.2f}%")
            print("  Possible causes: Overfitting to augmented patterns, need hyperparameter tuning")
        else:
            print(f"\n≈ OBSERVATION: Similar performance (difference: {acc_improvement:.2f}%)")
        
        print("\n" + "="*80)
        print("NEXT STEP: Generate preprocessing report")
        print("Run: python generate_preprocessing_report.py")
        print("="*80)
        break
    
    status_parts = []
    if baseline_done:
        with open(OUTPUT_DIR / 'baseline_results.json') as f:
            baseline = json.load(f)
        status_parts.append(f"✓ Baseline done ({baseline['test_acc']*100:.2f}% test acc)")
    else:
        status_parts.append("⏳ Baseline training...")
    
    if augmented_done:
        with open(OUTPUT_DIR / 'augmented_results.json') as f:
            augmented = json.load(f)
        status_parts.append(f"✓ Augmented done ({augmented['test_acc']*100:.2f}% test acc)")
    else:
        status_parts.append("⏳ Augmented training...")
    
    print(status_line + " | ".join(status_parts))
    
    time.sleep(60)  # Check every minute
    checked_count += 1

else:
    print("\n⏰ Monitor timeout (2 hours). Training may still be in progress.")
    print("Check models/ directory for *_results.json files manually.")

print("\n" + "="*80)
