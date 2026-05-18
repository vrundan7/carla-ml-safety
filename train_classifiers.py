"""
Train three separate binary classifiers (Pedestrian, Traffic Light, Vehicle)
using a pre-trained ResNet-18 backbone fine-tuned for binary classification.

Architecture:
  - ResNet-18 pre-trained on ImageNet
  - Final FC layer replaced: Linear(512, 1) for binary output

Training Setup:
  - Loss: BCEWithLogitsLoss with pos_weight to handle class imbalance
  - Optimizer: Adam (lr=1e-4)
  - Scheduler: ReduceLROnPlateau (patience=2, factor=0.5)
  - Epochs: 10
  - Batch size: 32
  - Images resized to 224x224, normalized with ImageNet stats
  - Data augmentation: RandomHorizontalFlip, ColorJitter
"""

import os
import sys
import csv
import time
import copy
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# Configuration  *** UPDATE BASE_PATH FOR YOUR ENVIRONMENT ***
# ──────────────────────────────────────────────

# For Colab: BASE_PATH = "/content/drive/MyDrive/2026/2026"
BASE_PATH = "/content/drive/MyDrive/2026/2026"

TRAIN_CSV = f"{BASE_PATH}/train/train/labels.csv"
TRAIN_IMG = f"{BASE_PATH}/train/train/rgb-front"
VAL_CSV   = f"{BASE_PATH}/validation/validation/labels.csv"
VAL_IMG   = f"{BASE_PATH}/validation/validation/rgb-front"

LABELS = ['has_traffic_light', 'has_pedestrian', 'has_vehicle']
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
OUTPUT_DIR = f"{BASE_PATH}/model_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class DrivingDataset(Dataset):
    """Custom Dataset for loading driving images with binary labels."""

    def __init__(self, csv_path, img_dir, label_col, transform=None):
        self.img_dir = img_dir
        self.label_col = label_col
        self.transform = transform
        self.samples = []

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame = row['frame']
                label = 1.0 if row[label_col] == 'True' else 0.0
                img_path = os.path.join(img_dir, f"{frame}.jpg")
                if os.path.exists(img_path):
                    self.samples.append((img_path, label))

        print(f"  Loaded {len(self.samples)} samples for '{label_col}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
def build_model():
    """ResNet-18 with final FC replaced for binary classification (512 -> 1)."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 1)
    return model.to(DEVICE)


# ──────────────────────────────────────────────
# Class weight for imbalanced data
# ──────────────────────────────────────────────
def compute_pos_weight(dataset):
    """pos_weight = num_negative / num_positive for BCEWithLogitsLoss."""
    labels = [s[1] for s in dataset.samples]
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos
    if num_pos == 0:
        return torch.tensor([1.0])
    weight = num_neg / num_pos
    return torch.tensor([weight], dtype=torch.float32).to(DEVICE)


# ──────────────────────────────────────────────
# Training / Validation loops
# ──────────────────────────────────────────────
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


# ──────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────
def train_classifier(label_col):
    """Train a single binary classifier for the given label."""
    print(f"\n{'='*60}")
    print(f"  Training classifier for: {label_col}")
    print(f"{'='*60}")

    # Datasets
    train_dataset = DrivingDataset(TRAIN_CSV, TRAIN_IMG, label_col, train_transform)
    val_dataset   = DrivingDataset(VAL_CSV, VAL_IMG, label_col, val_transform)

    # DataLoaders
    nw = 2 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=nw, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=nw, pin_memory=torch.cuda.is_available())

    # Model, loss, optimizer, scheduler
    model = build_model()
    pos_weight = compute_pos_weight(train_dataset)
    print(f"  pos_weight: {pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=0.5, patience=2)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_model_wts = None

    print(f"  Device: {DEVICE} | Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"  Epochs: {NUM_EPOCHS} | BS: {BATCH_SIZE} | LR: {LR}\n")

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc     = validate(model, val_loader, criterion)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"{time.time()-t0:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    # Save best weights
    path = os.path.join(OUTPUT_DIR, f"best_{label_col}.pth")
    torch.save(best_model_wts, path)
    print(f"\n  Best model saved → {path}  (val_loss: {best_val_loss:.4f})")

    return history


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
def plot_histories(all_histories):
    """Plot loss and accuracy curves for all three classifiers."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (label_col, h) in enumerate(all_histories.items()):
        ax = axes[idx]
        epochs = range(1, len(h['train_loss']) + 1)
        ax.plot(epochs, h['train_loss'], 'b-o', label='Train Loss', ms=4)
        ax.plot(epochs, h['val_loss'],   'r-o', label='Val Loss',   ms=4)
        ax.set_title(label_col.replace('has_', '').replace('_', ' ').title(),
                      fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle('Training & Validation Loss', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curves.png'), dpi=150, bbox_inches='tight')
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (label_col, h) in enumerate(all_histories.items()):
        ax = axes[idx]
        epochs = range(1, len(h['train_acc']) + 1)
        ax.plot(epochs, h['train_acc'], 'b-o', label='Train Acc', ms=4)
        ax.plot(epochs, h['val_acc'],   'r-o', label='Val Acc',   ms=4)
        ax.set_title(label_col.replace('has_', '').replace('_', ' ').title(),
                      fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_ylim([0.5, 1.0])
    plt.suptitle('Training & Validation Accuracy', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_curves.png'), dpi=150, bbox_inches='tight')
    plt.show()


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
if __name__ == '__main__':
    print("="*60)
    print(f"  Binary Classifier Training Pipeline  |  Device: {DEVICE}")
    print("="*60)

    all_histories = {}
    for label in LABELS:
        all_histories[label] = train_classifier(label)

    plot_histories(all_histories)

    # Summary table
    print("\n" + "="*76)
    print(f"{'Label':<25} {'Best Train Loss':<18} {'Best Val Loss':<18} {'Best Val Acc':<15}")
    print("-"*76)
    for label, h in all_histories.items():
        print(f"{label:<25} {min(h['train_loss']):<18.4f} {min(h['val_loss']):<18.4f} {max(h['val_acc']):<15.4f}")
    print("="*76)
