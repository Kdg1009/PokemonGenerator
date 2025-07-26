import torch
from GenComps import FUNITGenerator
import os
import re
from PokemonFUNITAnimalDataset import PokemonFUNITAnimalDataset
import cv2
import numpy as np

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

model = FUNITGenerator()
latest_pth_location = get_latest_checkpoint()
checkpoint = torch.load(latest_pth_location, weights_only=True)
state_dict = checkpoint['G']
new_state_dict = {}
prefix = '_orig_mod.'
for k, v in state_dict.items():
    if k.startswith(prefix):
        new_key = k[len(prefix):]
    else:
        new_key = k
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)
model.eval()

dataset = PokemonFUNITAnimalDataset(
    "D:/kaggle_datasets/animal141",
    "D:/kaggle_datasets/PokemonData",
    mode='test'
    )

test_loader = torch.utils.data.DataLoader(dataset,1,shuffle=False)

with torch.no_grad():
    for data in test_loader:
        content_img = data['content_img']
        style_imgs = data['style_imgs']
        style_cls = data['style_cls']
        gen_img = model(content_img, style_imgs)

        img = gen_img.squeeze(0).cpu().detach().numpy()
        img = np.transpose(img, (1,2,0))
        img = (img*255).clip(0,255).astype(np.uint8)

        print(f'style ref by {style_cls[0]} img size:{img.shape[0]},{img.shape[1]}')
        cv2.imshow('test img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break