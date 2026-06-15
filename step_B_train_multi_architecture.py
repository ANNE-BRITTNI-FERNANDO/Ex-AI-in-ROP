"""
STEP B: TRAIN & COMPARE MULTIPLE ARCHITECTURES ON CZECH ROP DATASET
Models: ResNet50, EfficientNet-B0, DenseNet121
All use ImageNet pretrained weights + fine-tuning with early stopping
Saves individual model files and comparison summary
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import time
import os
import copy
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR       = Path("data/czech_rop_organized")
MODELS_DIR     = Path("models/multi_arch")
RESULTS_DIR    = Path("results/multi_arch_comparison")
BATCH_SIZE     = 64
MAX_EPOCHS     = 15
PATIENCE       = 5         # early stopping
LR_PATIENCE    = 3
LEARNING_RATE  = 0.001
SUBSET_PER_CLASS = 1500   # use 1500 Normal + 1500 ROP from train (3000 total) for CPU speed
IMG_SIZE       = 224
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("STEP B: MULTI-ARCHITECTURE TRAINING")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Data:   {DATA_DIR}")

# ─── TRANSFORMS ──────────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Load full datasets
full_train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
val_ds        = datasets.ImageFolder(DATA_DIR / "val",   transform=val_tf)
test_ds       = datasets.ImageFolder(DATA_DIR / "test",  transform=val_tf)

# Balanced subset from training set for CPU speed
from torch.utils.data import Subset
import random as _rnd
_rnd.seed(42)
_indices_per_class = {c: [] for c in [0, 1]}
for idx, (_, lbl) in enumerate(full_train_ds.samples):
    _indices_per_class[lbl].append(idx)
subset_indices = []
for lbl in [0, 1]:
    chosen = _rnd.sample(_indices_per_class[lbl], min(SUBSET_PER_CLASS, len(_indices_per_class[lbl])))
    subset_indices.extend(chosen)
train_ds = Subset(full_train_ds, subset_indices)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

print(f"\nTrain subset: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
print(f"Classes: {full_train_ds.classes}")

# ─── MODEL BUILDERS ──────────────────────────────────────────────────────────
def build_resnet50():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def build_efficientnet_b0():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for name, param in model.named_parameters():
        if "features.7" not in name and "features.8" not in name and "classifier" not in name:
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 2)
    )
    return model

def build_densenet121():
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    for name, param in model.named_parameters():
        if "denseblock4" not in name and "classifier" not in name:
            param.requires_grad = False
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    return model

ARCHITECTURES = {
    "ResNet50":       build_resnet50,
    "EfficientNet-B0": build_efficientnet_b0,
    "DenseNet121":    build_densenet121,
}

# ─── TRAINING LOOP ───────────────────────────────────────────────────────────
def train_model(name, model):
    print(f"\n{'-'*60}")
    print(f"Training: {name}")
    print(f"{'-'*60}")

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=LR_PATIENCE,
                                  factor=0.5)

    best_val_acc = 0.0
    best_weights = None
    no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ──
        model.train()
        run_loss, run_correct, run_total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * inputs.size(0)
            run_correct += (outputs.argmax(1) == labels).sum().item()
            run_total += inputs.size(0)
        t_loss = run_loss / run_total
        t_acc  = run_correct / run_total

        # ── Validate ──
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * inputs.size(0)
                v_correct += (outputs.argmax(1) == labels).sum().item()
                v_total += inputs.size(0)
        v_loss /= v_total
        v_acc = v_correct / v_total

        history["train_loss"].append(round(t_loss, 4))
        history["train_acc"].append(round(t_acc, 4))
        history["val_loss"].append(round(v_loss, 4))
        history["val_acc"].append(round(v_acc, 4))
        history["lr"].append(optimizer.param_groups[0]["lr"])

        scheduler.step(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_weights = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        print(f"  Epoch {epoch:2d}/{MAX_EPOCHS} | "
              f"train_loss={t_loss:.4f} acc={t_acc:.4f} | "
              f"val_loss={v_loss:.4f} acc={v_acc:.4f} | "
              f"best={best_val_acc:.4f} | no_improve={no_improve}")

        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    total_time = time.time() - start_time
    print(f"\n  Training complete in {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")

    # ── Save best model ──
    model.load_state_dict(best_weights)
    model_path = MODELS_DIR / f"{name.lower().replace('-', '_').replace(' ', '_')}_best.pth"
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved: {model_path}")

    # ── Test evaluation ──
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc  = roc_auc_score(all_labels, all_probs)

    results = {
        "model": name,
        "test_accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall_sensitivity": round(float(rec), 4),
        "specificity": round(float(spec), 4),
        "f1_score": round(float(f1), 4),
        "auc_roc": round(float(auc), 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "best_val_accuracy": round(float(best_val_acc), 4),
        "training_time_seconds": round(total_time, 2),
        "epochs_trained": len(history["train_loss"]),
        "history": history,
    }

    print(f"\n  TEST RESULTS:")
    print(f"    Accuracy:    {acc:.4f}")
    print(f"    Precision:   {prec:.4f}")
    print(f"    Recall:      {rec:.4f}")
    print(f"    Specificity: {spec:.4f}")
    print(f"    F1-Score:    {f1:.4f}")
    print(f"    AUC-ROC:     {auc:.4f}")

    result_path = RESULTS_DIR / f"{name.lower().replace('-', '_').replace(' ', '_')}_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    return results, model


# ─── TRAIN ALL ARCHITECTURES ─────────────────────────────────────────────────
all_results = {}

for arch_name, build_fn in ARCHITECTURES.items():
    model = build_fn()
    results, trained_model = train_model(arch_name, model)
    all_results[arch_name] = results

# ─── COMPARISON SUMMARY ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MULTI-ARCHITECTURE COMPARISON SUMMARY")
print("=" * 70)
print(f"{'Model':<20} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'Spec':>7} {'F1':>7} {'AUC':>7} {'Time(s)':>9} {'Epochs':>7}")
print("-" * 70)
for name, r in all_results.items():
    print(f"{name:<20} {r['test_accuracy']:>7.4f} {r['precision']:>7.4f} "
          f"{r['recall_sensitivity']:>7.4f} {r['specificity']:>7.4f} "
          f"{r['f1_score']:>7.4f} {r['auc_roc']:>7.4f} "
          f"{r['training_time_seconds']:>9.1f} {r['epochs_trained']:>7d}")

# Find best model
best_name = max(all_results, key=lambda k: all_results[k]["auc_roc"])
print(f"\nBest model by AUC-ROC: {best_name}")

summary = {
    "models": all_results,
    "best_model_by_auc": best_name,
    "dataset": "Czech ROP dataset (6004 images)",
    "training_images": len(train_ds),
    "val_images": len(val_ds),
    "test_images": len(test_ds),
}

summary_path = RESULTS_DIR / "comparison_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nComparison summary saved: {summary_path}")
print("=" * 70)
