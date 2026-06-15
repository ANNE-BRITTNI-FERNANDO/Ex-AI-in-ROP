"""
AUGMENTED MODEL TRAINING
Train ResNet50 on 2000 augmented HVDROPDB images
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import json
import time

print("="*80)
print("AUGMENTED MODEL TRAINING (2000 IMAGES)")
print("="*80)

# Configuration
DATA_DIR = Path(r"C:\Users\ASUS\FYP - MidPoint\data\hvdropdb_split")
OUTPUT_DIR = Path(r"C:\Users\ASUS\FYP - MidPoint\models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
LEARNING_RATE = 0.001
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\nDevice: {DEVICE}")
print(f"Training on: 2000 AUGMENTED images (1400 train)")

# Data transforms (only basic transforms, augmentation already applied)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),  # Light additional augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
print("\n⏳ Loading datasets...")
train_dataset = datasets.ImageFolder(DATA_DIR / 'train', transform=train_transform)
val_dataset = datasets.ImageFolder(DATA_DIR / 'val', transform=val_transform)
test_dataset = datasets.ImageFolder(DATA_DIR / 'test', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"✓ Train: {len(train_dataset)} images")
print(f"✓ Val:   {len(val_dataset)} images")
print(f"✓ Test:  {len(test_dataset)} images")

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
print("TRAINING AUGMENTED MODEL")
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
        }, OUTPUT_DIR / 'augmented_best_model.pth')
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

checkpoint = torch.load(OUTPUT_DIR / 'augmented_best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)

print(f"\nAugmented Model Results:")
print(f"  Training images: {len(train_dataset)}")
print(f"  Best Val Acc: {best_val_acc*100:.2f}%")
print(f"  Test Acc: {test_acc*100:.2f}%")
print(f"  Training time: {training_time/60:.1f} minutes")

# Save results
results = {
    'model_type': 'augmented_model',
    'training_images': len(train_dataset),
    'val_images': len(val_dataset),
    'test_images': len(test_dataset),
    'best_val_acc': best_val_acc,
    'test_acc': test_acc,
    'test_loss': test_loss,
    'total_epochs': epoch,
    'training_time_seconds': training_time,
    'augmentation_techniques': [
        'HorizontalFlip (p=0.5)',
        'Rotation (±15°)',
        'ShiftScaleRotate',
        'RandomBrightnessContrast (±20%)',
        'CLAHE',
        'HueSaturationValue',
        'GaussianBlur',
        'GaussNoise',
        'Resize to 224×224'
    ],
    'preprocessing': [
        'Augmentation factor: 10-12×',
        'ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])',
        'Additional RandomHorizontalFlip during training'
    ],
    'history': history
}

with open(OUTPUT_DIR / 'augmented_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: {OUTPUT_DIR / 'augmented_results.json'}")
print("="*80)
