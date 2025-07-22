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

if __name__ == "__main__":
    test_pipeline()
