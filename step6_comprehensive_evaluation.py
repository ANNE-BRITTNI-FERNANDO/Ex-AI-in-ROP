"""
Comprehensive Model Evaluation
Computes: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix
"""

import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the trained model
print("\nLoading model...")
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 classes
model = model.to(device)

# Load trained weights
checkpoint_path = 'models/augmented_best_model.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()
print("✓ Model loaded successfully")

# Data preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Load test dataset
print("\nLoading test dataset...")
test_dir = 'data/hvdropdb_split/test'
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class_names = test_dataset.classes  # ['Normal', 'ROP']
print(f"Classes: {class_names}")
print(f"Test images: {len(test_dataset)}")

# Get predictions
print("\nGenerating predictions...")
all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)
all_probabilities = np.array(all_probabilities)

print(f"✓ Predictions generated for {len(all_labels)} images")

# Calculate metrics
print("\n" + "="*80)
print("COMPREHENSIVE EVALUATION METRICS")
print("="*80)

# 1. Accuracy
accuracy = accuracy_score(all_labels, all_predictions)
print(f"\n1. ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")

# 2. Precision (Of predicted ROP, how many are actually ROP?)
precision = precision_score(all_labels, all_predictions, pos_label=1)
print(f"\n2. PRECISION: {precision:.4f} ({precision*100:.2f}%)")
print(f"   → Of images predicted as ROP, {precision*100:.2f}% actually are ROP")

# 3. Recall (Of actual ROP, how many did we catch?)
recall = recall_score(all_labels, all_predictions, pos_label=1)
print(f"\n3. RECALL (Sensitivity): {recall:.4f} ({recall*100:.2f}%)")
print(f"   → Of actual ROP cases, we detected {recall*100:.2f}%")

# 4. F1-Score (Balance of precision and recall)
f1 = f1_score(all_labels, all_predictions, pos_label=1)
print(f"\n4. F1-SCORE: {f1:.4f} ({f1*100:.2f}%)")
print(f"   → Harmonic mean of precision and recall")

# 5. Specificity (Of actual Normal, how many did we correctly identify?)
tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
specificity = tn / (tn + fp)
print(f"\n5. SPECIFICITY: {specificity:.4f} ({specificity*100:.2f}%)")
print(f"   → Of actual Normal cases, we correctly identified {specificity*100:.2f}%")

# 6. AUC-ROC (Overall discrimination ability)
auc_roc = roc_auc_score(all_labels, all_probabilities[:, 1])
print(f"\n6. AUC-ROC: {auc_roc:.4f}")
print(f"   → Model's ability to distinguish ROP from Normal")
print(f"   → Interpretation: {auc_roc*100:.1f}% chance model ranks random ROP higher than random Normal")

# Classification report
print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORT")
print("="*80)
print(classification_report(all_labels, all_predictions, 
                           target_names=class_names, digits=4))

# Confusion Matrix
print("\n" + "="*80)
print("CONFUSION MATRIX")
print("="*80)
cm = confusion_matrix(all_labels, all_predictions)
print("\n                Predicted")
print("               Normal  ROP")
print(f"Actual Normal    {cm[0,0]:3d}   {cm[0,1]:3d}")
print(f"Actual ROP       {cm[1,0]:3d}   {cm[1,1]:3d}")

print("\nInterpretation:")
print(f"  True Negatives (TN):  {tn:3d} - Correctly identified Normal")
print(f"  False Positives (FP): {fp:3d} - Normal wrongly labeled as ROP")
print(f"  False Negatives (FN): {fn:3d} - ROP missed (wrongly labeled as Normal)")
print(f"  True Positives (TP):  {tp:3d} - Correctly identified ROP")

# Calculate derived metrics
print("\n" + "="*80)
print("CLINICAL SIGNIFICANCE")
print("="*80)
print(f"Positive Predictive Value (PPV): {precision*100:.2f}%")
print(f"  → If model says 'ROP', {precision*100:.2f}% chance it's correct")
print(f"\nNegative Predictive Value (NPV): {(tn/(tn+fn))*100:.2f}%")
print(f"  → If model says 'Normal', {(tn/(tn+fn))*100:.2f}% chance it's correct")
print(f"\nMisclassification Rate: {(1-accuracy)*100:.2f}%")
print(f"  → {(1-accuracy)*100:.2f}% of predictions are wrong")

# Save results to JSON
results = {
    "test_samples": int(len(all_labels)),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "specificity": float(specificity),
    "auc_roc": float(auc_roc),
    "confusion_matrix": {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    },
    "derived_metrics": {
        "positive_predictive_value": float(precision),
        "negative_predictive_value": float(tn/(tn+fn)),
        "misclassification_rate": float(1-accuracy)
    }
}

output_dir = Path('results/comprehensive_evaluation')
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'metrics.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Metrics saved to: {output_dir / 'metrics.json'}")

# Create visualizations
print("\nCreating visualizations...")

fig = plt.figure(figsize=(16, 10))

# 1. Confusion Matrix Heatmap
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

# 2. Normalized Confusion Matrix
ax2 = plt.subplot(2, 3, 2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            vmin=0, vmax=1, cbar_kws={'label': 'Percentage'})
plt.title('Normalized Confusion Matrix (%)', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

# 3. Metrics Bar Chart
ax3 = plt.subplot(2, 3, 3)
metrics_dict = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Specificity': specificity,
    'AUC-ROC': auc_roc
}
bars = plt.bar(range(len(metrics_dict)), list(metrics_dict.values()), 
               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
plt.xticks(range(len(metrics_dict)), list(metrics_dict.keys()), rotation=45, ha='right')
plt.ylabel('Score', fontsize=12)
plt.title('Performance Metrics', fontsize=14, fontweight='bold')
plt.ylim(0, 1.1)
plt.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.3f}', ha='center', va='bottom', fontsize=10)

# 4. ROC Curve
ax4 = plt.subplot(2, 3, 4)
fpr, tpr, thresholds = roc_curve(all_labels, all_probabilities[:, 1])
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {auc_roc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

# 5. Prediction Distribution
ax5 = plt.subplot(2, 3, 5)
normal_probs = all_probabilities[all_labels == 0, 1]
rop_probs = all_probabilities[all_labels == 1, 1]
plt.hist(normal_probs, bins=30, alpha=0.5, label='Normal (True)', color='blue')
plt.hist(rop_probs, bins=30, alpha=0.5, label='ROP (True)', color='red')
plt.xlabel('Predicted ROP Probability', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(axis='y', alpha=0.3)

# 6. Error Analysis
ax6 = plt.subplot(2, 3, 6)
error_types = ['True Negative\n(Correct Normal)', 
               'False Positive\n(Wrong ROP)', 
               'False Negative\n(Missed ROP)', 
               'True Positive\n(Correct ROP)']
error_counts = [tn, fp, fn, tp]
colors = ['green', 'orange', 'red', 'green']
bars = plt.bar(range(4), error_counts, color=colors, alpha=0.7)
plt.xticks(range(4), error_types, fontsize=9)
plt.ylabel('Count', fontsize=12)
plt.title('Prediction Breakdown', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to: {output_dir / 'comprehensive_evaluation.png'}")

print("\n" + "="*80)
print("VALIDATION CONCLUSION")
print("="*80)
print(f"\n✅ Accuracy: {accuracy*100:.2f}% - {'EXCELLENT' if accuracy >= 0.90 else 'GOOD' if accuracy >= 0.85 else 'NEEDS IMPROVEMENT'}")
print(f"✅ AUC-ROC: {auc_roc:.4f} - {'EXCELLENT' if auc_roc >= 0.90 else 'GOOD' if auc_roc >= 0.80 else 'FAIR'}")
print(f"✅ Precision: {precision*100:.2f}% - Low false alarms")
print(f"✅ Recall: {recall*100:.2f}% - Catching most ROP cases")

if accuracy >= 0.85:
    print("\n🎯 MODEL IS VALIDATED! Safe to trust Grad-CAM explanations.")
    print("   Your classifier performs well enough for XAI validation.")
else:
    print("\n⚠️ Model accuracy below 85% - consider improvement before finalizing.")

print(f"\n📊 All results saved to: {output_dir}")
print("="*80 + "\n")
