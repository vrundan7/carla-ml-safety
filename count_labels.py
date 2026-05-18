import pandas as pd
import os

# *** UPDATE THIS PATH FOR YOUR ENVIRONMENT ***
# For Colab: BASE_PATH = "/content/drive/MyDrive/2026/2026"
BASE_PATH = "/content/drive/MyDrive/2026/2026"

def print_dist(csv_path, split_name):
    df = pd.read_csv(csv_path)
    print(f"--- {split_name} Split ---")
    for col in ['has_traffic_light', 'has_pedestrian', 'has_vehicle']:
        counts = df[col].value_counts(normalize=True) * 100
        true_pct = counts.get(True, 0)
        false_pct = counts.get(False, 0)
        print(f"{col:<20} : {true_pct:>5.1f}% True  |  {false_pct:>5.1f}% False")
    print()

train_csv = f"{BASE_PATH}/train/train/labels.csv"
test_csv = f"{BASE_PATH}/test/test/labels.csv"

print_dist(train_csv, "Training")
print_dist(test_csv, "Test")
