import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time
import copy
import os
from pathlib import Path
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
DATA_DIR = Path(r"C:\Users\ASUS\FYP - MidPoint\data\classification")
MODEL_DIR = Path(r"C:\Users\ASUS\FYP - MidPoint\models")
MODEL_DIR.mkdir(exist_ok=True)

# Training parameters
NUM_EPOCHS = 30              # Maximum epochs (early stopping will likely stop sooner)
BATCH_SIZE = 16              # Process 16 images at a time (adjust based on your GPU/RAM)
LEARNING_RATE = 0.001        # How fast the model learns (0.001 is a safe default)
EARLY_STOPPING_PATIENCE = 7  # Stop if no improvement for 7 epochs

# Device configuration (use GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# =============================================================================
# DATA AUGMENTATION & PREPROCESSING
# =============================================================================

# Data augmentation for training (makes model more robust)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),           # ResNet50 expects 224x224 images
    transforms.RandomHorizontalFlip(p=0.5),  # Flip 50% of images (retinas can be left/right)
    transforms.RandomRotation(15),           # Rotate up to 15 degrees
    transforms.ColorJitter(brightness=0.2,   # Vary brightness, contrast, saturation
                          contrast=0.2, 
                          saturation=0.2),
    transforms.ToTensor(),                   # Convert to tensor
    transforms.Normalize(                    # Normalize using ImageNet stats
        mean=[0.485, 0.456, 0.406],         # (ResNet50 was trained with these)
        std=[0.229, 0.224, 0.225]
    )
])

# Validation/Test transforms (NO augmentation - we want real performance)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =============================================================================
# LOAD DATASETS
# =============================================================================

print("\n" + "="*70)
print("LOADING DATASETS")
print("="*70)

# ImageFolder automatically creates labels from folder names (Normal=0, ROP=1)
train_dataset = datasets.ImageFolder(DATA_DIR / 'train', transform=train_transforms)
val_dataset = datasets.ImageFolder(DATA_DIR / 'val', transform=val_transforms)
test_dataset = datasets.ImageFolder(DATA_DIR / 'test', transform=val_transforms)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}")

# Create data loaders (these feed batches to the model)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Batches per epoch: {len(train_loader)}")

# =============================================================================
# BUILD MODEL
# =============================================================================

print("\n" + "="*70)
print("BUILDING MODEL")
print("="*70)

# Load pretrained ResNet50 (trained on ImageNet - 1 million images)
model = models.resnet50(pretrained=True)
print("✓ Loaded pretrained ResNet50 (trained on ImageNet)")

# Freeze early layers (they already know basic features like edges, textures)
# We'll only train the last few layers for retinal images
for param in model.parameters():
    param.requires_grad = False  # Don't update these weights

# Replace final layer: 1000 classes (ImageNet) → 2 classes (Normal, ROP)
num_features = model.fc.in_features  # 2048 features from ResNet50
model.fc = nn.Linear(num_features, 2)  # New layer: 2048 → 2 classes

print(f"✓ Modified final layer: {num_features} features → 2 classes")
print(f"✓ Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

# Move model to GPU/CPU
model = model.to(device)

# =============================================================================
# TRAINING SETUP
# =============================================================================

# Loss function: Cross-Entropy (standard for classification)
criterion = nn.CrossEntropyLoss()

# Optimizer: Adam (adaptive learning rate, works well in practice)
# We only optimize the final layer (fc) since we froze the rest
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler: Reduce learning rate if validation loss plateaus
# This helps the model converge better
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch and return average loss and accuracy"""
    model.train()  # Set model to training mode (enables dropout, etc.)
    
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in loader:
        # Move data to GPU/CPU
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero the gradients (PyTorch accumulates them by default)
        optimizer.zero_grad()
        
        # Forward pass: compute predictions
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)  # Get predicted class
        loss = criterion(outputs, labels)  # Compute loss
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    # Calculate epoch averages
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects.double() / len(loader.dataset)
    
    return epoch_loss, epoch_acc.item()


def validate(model, loader, criterion, device):
    """Validate model and return average loss and accuracy"""
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    
    running_loss = 0.0
    running_corrects = 0
    
    # No gradient computation during validation (saves memory)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects.double() / len(loader.dataset)
    
    return epoch_loss, epoch_acc.item()


# =============================================================================
# TRAINING LOOP
# =============================================================================

print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)
print(f"Training for maximum {NUM_EPOCHS} epochs")
print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE} epochs")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print("="*70)

# Track training history
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'learning_rates': []
}

# Early stopping variables
best_val_loss = float('inf')
best_val_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
best_epoch = 0
epochs_no_improve = 0

# Training start time
start_time = time.time()

# Main training loop
for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 70)
    
    # Train for one epoch
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # Update learning rate based on validation loss
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save to history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['learning_rates'].append(current_lr)
    
    # Print epoch results
    epoch_time = time.time() - epoch_start
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
    print(f"Learning Rate: {current_lr:.6f}")
    print(f"Time: {epoch_time:.2f}s")
    
    # Save checkpoint after every epoch (so you can resume if interrupted)
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'history': history
    }
    torch.save(checkpoint, MODEL_DIR / 'checkpoint_latest.pth')
    
    # Check if this is the best model so far
    if val_acc > best_val_acc:
        print(f"*** New best model! Val Acc improved from {best_val_acc:.4f} to {val_acc:.4f}")
        best_val_acc = val_acc
        best_val_loss = val_loss
        best_epoch = epoch + 1
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        
        # Save best model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'classes': train_dataset.classes
        }, MODEL_DIR / 'best_model.pth')
        print(f"*** Saved best model to {MODEL_DIR / 'best_model.pth'}")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s)")
    
    # Early stopping check
    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"\n" + "="*70)
        print(f"EARLY STOPPING triggered after {epoch + 1} epochs")
        print(f"No improvement for {EARLY_STOPPING_PATIENCE} consecutive epochs")
        print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
        print("="*70)
        break

# Training complete
total_time = time.time() - start_time
print(f"\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"Total time: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
print(f"Best epoch: {best_epoch}")
print(f"Best validation accuracy: {best_val_acc:.4f}")
print(f"Best validation loss: {best_val_loss:.4f}")

# Load best model weights
model.load_state_dict(best_model_wts)

# =============================================================================
# FINAL EVALUATION ON TEST SET
# =============================================================================

print("\n" + "="*70)
print("EVALUATING ON TEST SET")
print("="*70)

test_loss, test_acc = validate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# =============================================================================
# SAVE TRAINING HISTORY
# =============================================================================

history['best_epoch'] = best_epoch
history['best_val_acc'] = best_val_acc
history['best_val_loss'] = best_val_loss
history['test_acc'] = test_acc
history['test_loss'] = test_loss
history['total_time_seconds'] = total_time

with open(MODEL_DIR / 'training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

print(f"\n✓ Saved training history to {MODEL_DIR / 'training_history.json'}")
print(f"✓ Best model saved at {MODEL_DIR / 'best_model.pth'}")
print(f"✓ Latest checkpoint at {MODEL_DIR / 'checkpoint_latest.pth'}")

print("\n" + "="*70)
print("TRAINING PIPELINE COMPLETE!")
print("="*70)
print(f"\nYour trained model is ready at:")
print(f"  {MODEL_DIR / 'best_model.pth'}")
print(f"\nValidation Accuracy: {best_val_acc*100:.2f}%")
print(f"Test Accuracy: {test_acc*100:.2f}%")
print("\nNext steps:")
print("  1. Visualize training curves (plot training_history.json)")
print("  2. Analyze predictions on test set")
print("  3. Generate Grad-CAM visualizations")
