"""
STEP 4: TRAIN CZECH ROP PRETRAINING MODEL
ResNet50 on 4437 Czech ROP images for robust binary classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import json
import time
from tqdm import tqdm

print("="*80)
print("STEP 4: CZECH ROP PRETRAINING")
print("="*80)

# Configuration
DATA_DIR = Path(r"C:\Users\ASUS\FYP - MidPoint\data\czech_rop_organized")
OUTPUT_DIR = Path(r"C:\Users\ASUS\FYP - MidPoint\models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16  # Adjust based on RAM
LEARNING_RATE = 0.001
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\nDevice: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max epochs: {MAX_EPOCHS}")

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
train_dataset = datasets.ImageFolder(DATA_DIR / 'train', transform=train_transform)
val_dataset = datasets.ImageFolder(DATA_DIR / 'val', transform=val_transform)
test_dataset = datasets.ImageFolder(DATA_DIR / 'test', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"✓ Train: {len(train_dataset)} images")
print(f"✓ Val:   {len(val_dataset)} images")
print(f"✓ Test:  {len(test_dataset)} images")
print(f"✓ Classes: {train_dataset.classes}")

# Model setup
print("\n⏳ Creating ResNet50 model...")
model = models.resnet50(pretrained=True)

# Freeze early layers (transfer learning)
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last residual block (layer4) and fc layer
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Linear(2048, 2)  # Binary: Normal vs ROP

model = model.to(DEVICE)

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model created")
print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")

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
    
    pbar = tqdm(loader, desc="Training")
    for inputs, labels in pbar:
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
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Training loop
print("\n" + "="*80)
print("TRAINING CZECH ROP MODEL")
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
    print("-" * 80)
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
    
    # Update scheduler
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['learning_rates'].append(current_lr)
    
    print(f"\nResults:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc*100:.2f}%")
    print(f"  LR: {current_lr:.6f}")
    
    # Save best model
    if val_acc > best_val_acc:
        print(f"  *** NEW BEST MODEL (Val Acc: {val_acc*100:.2f}%) ***")
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, OUTPUT_DIR / 'czech_rop_best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Save latest checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }, OUTPUT_DIR / 'czech_rop_latest.pth')
    
    # Early stopping
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\n⚠ Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement")
        break

training_time = time.time() - start_time

# Final test evaluation
print("\n" + "="*80)
print("FINAL TEST EVALUATION")
print("="*80)

# Load best model
checkpoint = torch.load(OUTPUT_DIR / 'czech_rop_best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)

print(f"\nTest Results:")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Accuracy: {test_acc*100:.2f}%")

# Save training history
history['best_val_acc'] = best_val_acc
history['test_acc'] = test_acc
history['test_loss'] = test_loss
history['training_time_seconds'] = training_time
history['total_epochs'] = epoch

history_path = OUTPUT_DIR / 'czech_rop_training_history.json'
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)

print("\n" + "="*80)
print("CZECH ROP PRETRAINING COMPLETE")
print("="*80)
print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
print(f"Test accuracy: {test_acc*100:.2f}%")
print(f"Training time: {training_time/60:.1f} minutes")
print(f"Model saved: {OUTPUT_DIR / 'czech_rop_best_model.pth'}")
print(f"History saved: {history_path}")
print("\nNext: Fine-tune on HVDROPDB dataset")
print("="*80)
