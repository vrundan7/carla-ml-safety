"""
Evaluate the three trained binary classifiers on the test split.
Reports: Accuracy, Precision, Recall, F1-score for each model.

Run this AFTER train_classifiers.py has finished.
Uses the saved model weights from model_outputs/best_has_*.pth
"""

import os
import csv
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# ──────────────────────────────────────────────
# Configuration  *** UPDATE BASE_PATH FOR YOUR ENVIRONMENT ***
# ──────────────────────────────────────────────

# For Colab: BASE_PATH = "/content/drive/MyDrive/2026/2026"
BASE_PATH = "/content/drive/MyDrive/2026/2026"

TEST_CSV = f"{BASE_PATH}/test/test/labels.csv"
TEST_IMG = f"{BASE_PATH}/test/test/rgb-front"
MODEL_DIR = f"{BASE_PATH}/model_outputs"

LABELS = ['has_traffic_light', 'has_pedestrian', 'has_vehicle']
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ──────────────────────────────────────────────
# Dataset (same as training)
# ──────────────────────────────────────────────
class DrivingDataset(Dataset):
    def __init__(self, csv_path, img_dir, label_col, transform=None):
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


# ──────────────────────────────────────────────
# Transform (no augmentation — test time)
# ──────────────────────────────────────────────
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────
# Load model with saved weights
# ──────────────────────────────────────────────
def load_model(label_col):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 1)
    weights_path = os.path.join(MODEL_DIR, f"best_{label_col}.pth")
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print(f"  Loaded weights from {weights_path}")
    return model


# ──────────────────────────────────────────────
# Run inference and collect predictions
# ──────────────────────────────────────────────
def get_predictions(model, dataloader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


# ──────────────────────────────────────────────
# Evaluate one model
# ──────────────────────────────────────────────
def evaluate_model(label_col):
    print(f"\n{'='*60}")
    print(f"  Evaluating: {label_col}")
    print(f"{'='*60}")

    # Load model
    model = load_model(label_col)

    # Create test dataset & loader
    test_dataset = DrivingDataset(TEST_CSV, TEST_IMG, label_col, test_transform)
    print(f"  Test samples: {len(test_dataset)}")

    nw = 2 if torch.cuda.is_available() else 0
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=nw, pin_memory=torch.cuda.is_available())

    # Get predictions
    y_true, y_pred = get_predictions(model, test_loader)

    # Compute metrics
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"               Pred=0   Pred=1")
    print(f"    Actual=0   {cm[0][0]:>6}   {cm[0][1]:>6}")
    print(f"    Actual=1   {cm[1][0]:>6}   {cm[1][1]:>6}")

    # Full classification report
    print(f"\n{classification_report(y_true, y_pred, target_names=['False','True'])}")

    return {'label': label_col, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    print(f"Test CSV: {TEST_CSV}")

    results = []
    for label in LABELS:
        results.append(evaluate_model(label))

    # ── Summary Table ──
    print("\n" + "="*80)
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*80)
    for r in results:
        print(f"{r['label']:<25} {r['accuracy']:<12.4f} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1']:<12.4f}")
    print("="*80)

    # ── Find worst model ──
    worst = min(results, key=lambda x: x['f1'])
    print(f"\nWorst performing model (by F1): {worst['label']}  (F1 = {worst['f1']:.4f})")
