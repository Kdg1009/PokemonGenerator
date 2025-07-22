import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class StylizationDatasetPhase1(Dataset):
    def __init__(self, content_dir, style_dir, image_size=224, ref_k=5, mode='train'):
        """
        content_dir: path to animal dataset root (e.g., content/.train)
        style_dir: path to Pokémon dataset root (e.g., style/)
        image_size: output image size
        ref_k: number of reference images from a style class
        """
        super().__init__()
        self.ref_k = ref_k
        self.image_size = image_size
        content_dir = f'{content_dir}/{mode}'

        # Load content (animal) class info
        self.content_classes = [d for d in os.listdir(content_dir)
                                if os.path.isdir(os.path.join(content_dir, d))]
        self.content_class_to_imgs = {
            cls: [os.path.join(content_dir, cls, f)
                  for f in os.listdir(os.path.join(content_dir, cls))
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for cls in self.content_classes
        }
        self.content_samples = [(cls, img_path)
                                for cls, imgs in self.content_class_to_imgs.items()
                                for img_path in imgs]

        # Load style (Pokémon) class info
        self.style_classes = [d for d in os.listdir(style_dir)
                              if os.path.isdir(os.path.join(style_dir, d))]
        self.style_class_to_imgs = {
            cls: [os.path.join(style_dir, cls, f)
                  for f in os.listdir(os.path.join(style_dir, cls))
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for cls in self.style_classes
        }
        self.style_classes = self.style_classes.extend(self.content_classes)
        self.style_class_to_imgs = self.style_class_to_imgs | self.content_class_to_imgs

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.content_samples)

    def __getitem__(self, idx):
        # Content image (animal)
        content_cls, content_path = self.content_samples[idx]
        content_img = self.transform(Image.open(content_path).convert('RGB'))

        # Style images (Pokémon class)
        style_cls = random.choice(self.style_classes)
        style_paths = random.sample(self.style_class_to_imgs[style_cls], self.ref_k)
        style_imgs = [self.transform(Image.open(p).convert('RGB')) for p in style_paths]
        style_imgs_tensor = torch.stack(style_imgs)  # shape: [K, C, H, W]

        return {
            'content_img': content_img,          # [C, H, W]
            'style_imgs': style_imgs_tensor,     # [K, C, H, W]
            'content_cls': content_cls,
            'style_cls': style_cls
        }


class StylizationDatasetPhase2(Dataset):
    def __init__(self, content_dir, style_dir, image_size=224, ref_k=5, mode='train'):
        """
        content_dir: path to animal dataset root (e.g., content/.train)
        style_dir: path to Pokémon dataset root (e.g., style/)
        image_size: output image size
        ref_k: number of reference images from a style class
        """
        super().__init__()
        self.ref_k = ref_k
        self.image_size = image_size
        content_dir = f'{content_dir}/{mode}'

        # Load content (animal) class info
        self.content_classes = [d for d in os.listdir(content_dir)
                                if os.path.isdir(os.path.join(content_dir, d))]
        self.content_class_to_imgs = {
            cls: [os.path.join(content_dir, cls, f)
                  for f in os.listdir(os.path.join(content_dir, cls))
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for cls in self.content_classes
        }
        self.content_samples = [(cls, img_path)
                                for cls, imgs in self.content_class_to_imgs.items()
                                for img_path in imgs]

        # Load style (Pokémon) class info
        self.style_classes = [d for d in os.listdir(style_dir)
                              if os.path.isdir(os.path.join(style_dir, d))]
        self.style_class_to_imgs = {
            cls: [os.path.join(style_dir, cls, f)
                  for f in os.listdir(os.path.join(style_dir, cls))
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for cls in self.style_classes
        }

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.content_samples)

    def __getitem__(self, idx):
        # Content image (animal)
        content_cls, content_path = self.content_samples[idx]
        content_img = self.transform(Image.open(content_path).convert('RGB'))

        # Style images (Pokémon class)
        style_cls = random.choice(self.style_classes)
        style_paths = random.sample(self.style_class_to_imgs[style_cls], self.ref_k)
        style_imgs = [self.transform(Image.open(p).convert('RGB')) for p in style_paths]
        style_imgs_tensor = torch.stack(style_imgs)  # shape: [K, C, H, W]

        return {
            'content_img': content_img,          # [C, H, W]
            'style_imgs': style_imgs_tensor,     # [K, C, H, W]
            'content_cls': content_cls,
            'style_cls': style_cls
        }
