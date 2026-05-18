import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# *** UPDATE THIS PATH FOR YOUR ENVIRONMENT ***
# For Colab: BASE_PATH = "/content/drive/MyDrive/2026/2026"
BASE_PATH = "/content/drive/MyDrive/2026/2026"

TRAIN_CSV = f"{BASE_PATH}/train/train/labels.csv"
TEST_CSV  = f"{BASE_PATH}/test/test/labels.csv"
TRAIN_IMG_DIR = f"{BASE_PATH}/train/train/rgb-front"

# Load labels, parsing 'frame' as string to preserve leading zeros
df_train = pd.read_csv(TRAIN_CSV, dtype={'frame': str})
df_test  = pd.read_csv(TEST_CSV, dtype={'frame': str})

# 1. Split sizes
print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")

# 2. Class distribution for each label
labels = ["has_pedestrian", "has_traffic_light", "has_vehicle"]
print("\n--- Training Distribution ---")
for col in labels:
    counts = df_train[col].value_counts(normalize=True) * 100
    true_pct = counts.get(True, 0)
    false_pct = counts.get(False, 0)
    print(f"  {col:<25}: {true_pct:5.1f}% True  |  {false_pct:5.1f}% False")

print("\n--- Test Distribution ---")
for col in labels:
    counts = df_test[col].value_counts(normalize=True) * 100
    true_pct = counts.get(True, 0)
    false_pct = counts.get(False, 0)
    print(f"  {col:<25}: {true_pct:5.1f}% True  |  {false_pct:5.1f}% False")

# 3. Visualize sample images per label combination
combos = [
    (True, True, True, "TL+Ped+Veh"),
    (True, True, False, "TL+Ped"),
    (True, False, True, "TL+Veh"),
    (True, False, False, "TL only"),
    (False, True, True, "Ped+Veh"),
    (False, True, False, "Ped only"),
    (False, False, True, "Veh only"),
    (False, False, False, "None"),
]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for idx, (tl, ped, veh, title) in enumerate(combos):
    row, col = idx // 4, idx % 4
    mask = (
        (df_train['has_traffic_light'] == tl) &
        (df_train['has_pedestrian'] == ped) &
        (df_train['has_vehicle'] == veh)
    )
    subset = df_train[mask]
    if len(subset) > 0:
        frame = subset.iloc[0]['frame']
        img = Image.open(os.path.join(TRAIN_IMG_DIR, f"{frame}.jpg"))
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"{title}\n(n={len(subset)})", fontsize=11)
    else:
        axes[row, col].set_title(f"{title}\n(n=0)", fontsize=11)
    axes[row, col].axis('off')

plt.suptitle("Example Images per Label Combination", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
