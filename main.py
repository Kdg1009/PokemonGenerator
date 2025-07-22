import torch
from torch.utils.data import DataLoader

from models.discriminator import Discriminator
from models.generator import Generator
from utils.datasets import StylizationDatasetPhase1, StylizationDatasetPhase2
from training.train_phase1 import train_phase1
from training.train_phase2 import train_phase2

import os
import re

def get_latest_checkpoint(checkpoint_dir='checkpoints'):
    if not os.path.exists(checkpoint_dir):
        return None

    # Match filenames like 'funit_epoch10.pth'
    pattern = re.compile(r'funit_epoch(\d+)\.pth')
    checkpoint_files = []

    for fname in os.listdir(checkpoint_dir):
        match = pattern.match(fname)
        if match:
            epoch_num = int(match.group(1))
            checkpoint_files.append((epoch_num, fname))

    if not checkpoint_files:
        return None

    # Get the file with the highest epoch number
    latest_file = max(checkpoint_files, key=lambda x: x[0])[1]
    return os.path.join(checkpoint_dir, latest_file)

def main(phase='phase1', batch_size=8, num_epochs=10, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    if phase == 'phase1':
        print("Starting Phase 1 training...")

        # === Setup dataset and dataloader ===
        dataset = StylizationDatasetPhase1(
            content_dir='D:/kaggle_datasets/animal141',
            style_dir='D:/kaggle_datasets/cropped_artworks_224'
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=7, pin_memory=True)

        # === Initialize models ===
        generator = Generator()
        discriminator = Discriminator(num_classes=len(dataset.style_classes))

        # === Train ===
        train_phase1(generator, discriminator, dataloader,
                     num_epochs=num_epochs, device=device,
                     adv_weight=1.0, rec_weight=10.0,
                     style_weight=5.0, recon_weight=10.0,
                     warmup_epochs=50)

    elif phase == 'phase2':
        print("Starting Phase 2 training...")
        # === Setup dataset and dataloader ===
        dataset = StylizationDatasetPhase2(
            content_dir='D:/kaggle_datasets/animal141',
            style_dir='D:/kaggle_datasets/PokemonData'
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # === Initialize models ===
        generator = Generator()
        discriminator = Discriminator(num_classes=len(dataset.style_classes))
        # === Load generator pretrained from phase 1 ===
        latest_checkpoint = get_latest_checkpoint('checkpoints_phase1')
        generator.load_state_dict(torch.load(latest_checkpoint))

        # === Train ===
        train_phase2(generator, discriminator, dataloader,
                     num_epochs=num_epochs, device=device,
                     adv_weight=1.0, rec_weight=10.0,
                     style_weight=5.0)

    else:
        raise ValueError("phase must be either 'phase1' or 'phase2'")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(batch_size=16, num_epochs=2000, device=device)
