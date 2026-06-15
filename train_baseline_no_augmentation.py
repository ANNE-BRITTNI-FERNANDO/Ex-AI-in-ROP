"""
BASELINE MODEL TRAINING (No Augmentation)
Train ResNet50 on 185 original HVDROPDB images for comparison
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import time
import shutil
from PIL import Image
import os

print("="*80)
print("BASELINE MODEL TRAINING (NO AUGMENTATION)")
print("="*80)

# Configuration
RAW_DATA = Path(r"C:\Users\ASUS\FYP - MidPoint\data\raw_data\classification")
OUTPUT_DIR = Path(r"C:\Users\ASUS\FYP - MidPoint\models")
TEMP_DIR = Path(r"C:\Users\ASUS\FYP - MidPoint\data\baseline_split")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
LEARNING_RATE = 0.001
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\nDevice: {DEVICE}")
print(f"Training on: 185 ORIGINAL images (NO augmentation)")

# Collect all original images
print("\n⏳ Collecting original images...")
all_images = []
for label in ['Normal', 'ROP']:
    for device_folder in ['Neo_Normal', 'Neo_ROP', 'RetCam_Normal', 'RetCam_ROP']:
        folder_path = RAW_DATA / label / device_folder
        if folder_path.exists():
            for img_file in folder_path.glob("*.png"):
                all_images.append((str(img_file), label))

print(f"✓ Found {len(all_images)} images")
labels = [label for _, label in all_images]

# Split dataset: 70/15/15
from collections import Counter
print(f"\nClass distribution: {Counter(labels)}")

image_paths = [img for img, _ in all_images]
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths, labels, test_size=0.3, stratify=labels, random_state=42
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

print(f"\nSplit sizes:")
print(f"  Train: {len(train_paths)} ({train_labels.count('Normal')} Normal, {train_labels.count('ROP')} ROP)")
print(f"  Val:   {len(val_paths)} ({val_labels.count('Normal')} Normal, {val_labels.count('ROP')} ROP)")
print(f"  Test:  {len(test_paths)} ({test_labels.count('Normal')} Normal, {test_labels.count('ROP')} ROP)")

# Organize into ImageFolder structure
print("\n⏳ Organizing images...")
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)

for split_name, paths, split_labels in [
    ('train', train_paths, train_labels),
    ('val', val_paths, val_labels),
    ('test', test_paths, test_labels)
]:
    for label in ['Normal', 'ROP']:
        (TEMP_DIR / split_name / label).mkdir(parents=True, exist_ok=True)
    
    for img_path, label in zip(paths, split_labels):
        dst = TEMP_DIR / split_name / label / Path(img_path).name
        shutil.copy2(img_path, dst)

print("✓ Images organized!")

# Data transforms (NO augmentation for baseline)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
print("\n⏳ Loading datasets...")
train_dataset = datasets.ImageFolder(TEMP_DIR / 'train', transform=train_transform)
val_dataset = datasets.ImageFolder(TEMP_DIR / 'val', transform=val_transform)
test_dataset = datasets.ImageFolder(TEMP_DIR / 'test', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"✓ Datasets loaded")

# Model setup
print("\n⏳ Creating ResNet50 model...")
model = models.resnet50(pretrained=True)

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer4 and fc
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Linear(2048, 2)
model = model.to(DEVICE)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✓ Model created (trainable params: {trainable_params:,})")

# Optimizer and loss
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
criterion = nn.CrossEntropyLoss()

# Training functions
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / total, correct / total

# Training loop
print("\n" + "="*80)
print("TRAINING BASELINE MODEL")
print("="*80)

best_val_acc = 0.0
patience_counter = 0
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
    'learning_rates': []
}

start_time = time.time()

for epoch in range(1, MAX_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{MAX_EPOCHS}")
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
    
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['learning_rates'].append(current_lr)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
    print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc*100:.2f}%")
    
    if val_acc > best_val_acc:
        print(f"*** NEW BEST MODEL ***")
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
        }, OUTPUT_DIR / 'baseline_best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}")
        break

training_time = time.time() - start_time

# Test evaluation
print("\n" + "="*80)
print("TEST EVALUATION")
print("="*80)

checkpoint = torch.load(OUTPUT_DIR / 'baseline_best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)

print(f"\nBaseline Model Results (NO Augmentation):")
print(f"  Training images: {len(train_dataset)}")
print(f"  Best Val Acc: {best_val_acc*100:.2f}%")
print(f"  Test Acc: {test_acc*100:.2f}%")
print(f"  Training time: {training_time/60:.1f} minutes")

# Save results
results = {
    'model_type': 'baseline_no_augmentation',
    'training_images': len(train_dataset),
    'val_images': len(val_dataset),
    'test_images': len(test_dataset),
    'best_val_acc': best_val_acc,
    'test_acc': test_acc,
    'test_loss': test_loss,
    'total_epochs': epoch,
    'training_time_seconds': training_time,
    'preprocessing': [
        'Resize to 224×224',
        'ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])',
        'NO augmentation applied'
    ],
    'history': history
}

with open(OUTPUT_DIR / 'baseline_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: {OUTPUT_DIR / 'baseline_results.json'}")
print("="*80)
