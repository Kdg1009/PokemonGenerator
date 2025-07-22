import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader

from models.discriminator import Discriminator
from models.generator import Generator
from utils.datasets import StylizationDatasetPhase1
from training import train_phase1

def test_pipeline(device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    print("⏳ Initializing models...")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    print("📁 Loading Phase 1 dataset...")
    dataset = StylizationDatasetPhase1(
        content_root='D:/kaggle_datasets/animal141',
        style_root_list=['D:/kaggle_datasets/animal141', 'D:/kaggle_datasets/best-artworks-of-all-time']
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    print("📦 Fetching one batch...")
    content_img, style_img, class_idx, self_style_img = next(iter(dataloader))

    content_img = content_img.to(device)
    style_img = style_img.to(device)
    self_style_img = self_style_img.to(device)
    class_idx = class_idx.to(device)

    print("✅ Shapes:")
    print(" - content_img:", content_img.shape)
    print(" - style_img:", style_img.shape)
    print(" - class_idx:", class_idx.shape)
    print(" - self_style_img:", self_style_img.shape)

    print("🧠 Running forward pass of generator...")
    out = generator(content_img, style_img, return_all=True)
    fake_img = out['gen_img']
    print(" - fake_img:", fake_img.shape)

    print("🧠 Running discriminator...")
    score = discriminator(fake_img)
    print(" - disc score shape:", score.shape)

    print("✅ Passed all shape and device checks!")

import os

def find_small_classes(root_dir, ref_k, exts=('.png', '.jpg', '.jpeg')):
    """
    Lists subdirectories in `root_dir` that contain fewer than `ref_k` image files.

    Args:
        root_dir (str): Path to the dataset root.
        ref_k (int): Minimum number of images required.
        exts (tuple): Valid image file extensions.
    
    Returns:
        List[str]: Names of subdirectories with fewer than `ref_k` images.
    """
    too_small_classes = []

    for cls_name in os.listdir(root_dir):
        print(f"cheking {cls_name}...")
        cls_path = os.path.join(root_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue

        # Count valid image files
        img_files = [f for f in os.listdir(cls_path)
                     if f.lower().endswith(exts)]
        
        if len(img_files) < ref_k:
            print(f"[WARNING] Class '{cls_name}' has only {len(img_files)} images (needs at least {ref_k})")
            too_small_classes.append(cls_name)

    return too_small_classes


import os
from PIL import Image
import random

def crop_and_save_image(input_path, output_path, crop_size):
    img = Image.open(input_path).convert('RGB')
    width, height = img.size

    if width < crop_size or height < crop_size:
        print(f"Skipping small image: {input_path}")
        return

    left = random.randint(0, width - crop_size)
    top = random.randint(0, height - crop_size)
    crop = img.crop((left, top, left + crop_size, top + crop_size))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    crop.save(output_path)

def crop_dataset(input_dir, output_dir, crop_size=224, valid_exts=('.jpg', '.jpeg', '.png')):
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith(valid_exts):
                continue

            in_path = os.path.join(root, fname)
            rel_path = os.path.relpath(in_path, input_dir)
            out_path = os.path.join(output_dir, rel_path)

            try:
                crop_and_save_image(in_path, out_path, crop_size)
            except Exception as e:
                print(f"Error processing {in_path}: {e}")

def resize_images_in_directory(root_dir, target_size=(224, 224), valid_exts=('.jpg', '.jpeg', '.png')):
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if not fname.lower().endswith(valid_exts):
                continue

            img_path = os.path.join(root, fname)

            try:
                img = Image.open(img_path).convert('RGB')
                if img.size != target_size:
                    print(f"Resizing: {img_path} (was {img.size})")
                    img = img.resize(target_size, Image.BICUBIC)
                    img.save(img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

import os
from PIL import Image

def find_non_224_images(root_dir, valid_exts=('.jpg', '.jpeg', '.png')):
    count = 0
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if not fname.lower().endswith(valid_exts):
                continue
            img_path = os.path.join(root, fname)
            try:
                with Image.open(img_path) as img:
                    if img.size != (224, 224):
                        print(f"Not 224x224: {img_path} | size: {img.size}")
                        count += 1
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
    print(f"\nTotal images not 224x224: {count}")

if __name__ == "__main__":
    find_non_224_images("D:/kaggle_datasets/animal141/test")
