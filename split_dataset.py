import os
import random
import shutil
from sklearn.model_selection import train_test_split

# Config
base_path = r"D:\kaggle_datasets\animals141\dataset\dataset"
output_dirs = [".train", ".dev", ".test"]
split_ratios = (0.7, 0.15, 0.15)  # train, dev, test

# Step 1: Get all class folders
all_classes = [d for d in os.listdir(base_path)
               if os.path.isdir(os.path.join(base_path, d)) and not d.startswith(".")]

# Step 2: Shuffle and split
random.seed(42)
random.shuffle(all_classes)

train_classes, temp = train_test_split(all_classes, test_size=(1 - split_ratios[0]), random_state=42)
dev_classes, test_classes = train_test_split(temp, test_size=split_ratios[2] / (split_ratios[1] + split_ratios[2]), random_state=42)

splits = {
    ".train": train_classes,
    ".dev": dev_classes,
    ".test": test_classes
}

# Step 3: Create split directories and move class folders
for split_dir, class_list in splits.items():
    split_path = os.path.join(base_path, split_dir)
    os.makedirs(split_path, exist_ok=True)
    
    for cls in class_list:
        src = os.path.join(base_path, cls)
        dst = os.path.join(split_path, cls)
        if not os.path.exists(dst):
            shutil.move(src, dst)
            print(f"Moved {cls} → {split_dir}")
