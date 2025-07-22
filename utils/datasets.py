from torch.utils.data import Dataset
from PIL import Image
import os
import random
import torchvision.transforms as T

class StylizationDatasetPhase1(Dataset):
    def __init__(self, content_root, style_root_list, transform=None):
        self.transform = transform or T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

        # Load content paths
        self.content_paths = self._load_images(content_root)

        # Load all style paths (paintings + animal141)
        self.style_paths = []
        self.class_to_idx = {}
        self.style_dict = {} # class name -> list of (path, class_idx)
        class_idx = 0

        for root in style_root_list:
            for cls in sorted(os.listdir(root)):
                cls_path = os.path.join(root, cls)
                if os.path.isdir(cls_path):
                    if cls not in self.class_to_idx:
                        self.class_to_idx[cls] = class_idx
                        class_idx += 1
                    cls_idx = self.class_to_idx[cls]
                    for img in os.listdir(cls_path):
                        path = os.path.join(cls_path, img)
                        self.style_paths.append((path, cls_idx))
                        self.style_dict.setdefault(cls, []).append((path, cls_idx))

    def _load_images(self, root):
        all_paths = []
        for cls in os.listdir(root):
            cls_path = os.path.join(root, cls)
            if os.path.isdir(cls_path):
                for img in os.listdir(cls_path):
                    img_path = os.path.join(cls_path, img)
                    all_paths.append(img_path)
        return all_paths

    def __len__(self):
        return len(self.content_paths)

    def __getitem__(self, idx):
        content_path = self.content_paths[idx]
        content_cls = os.path.basename(os.path.dirname(content_path))

        content_img = Image.open(content_path).convert('RGB')
        content_img = self.transform(content_img)

        style_path, style_class_idx = random.choice(self.style_paths)
        style_img = Image.open(style_path).convert('RGB')
        style_img = self.transform(style_img)

        if content_cls in self.style_dict:
            same_class_style_list = self.style_dict[content_cls]
            self_style_path, self_style_class_idx = random.choice(same_class_style_list)
            self_style_img = Image.open(self_style_path).convert('RGB')
            self_style_img = self.transform(self_style_img)
        else:
            self_style_img = content_img  # Fallback to content image if no same class styles
        return content_img, style_img, style_class_idx, self_style_img

class StylizationDatasetPhase2(Dataset):
    def __init__(self, content_root, pokemon_root, transform=None):
        self.transform = transform or T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

        # === Load content paths and build class lookup ===
        self.content_paths = []
        self.content_class_map = {}  # path → class name
        for cls in sorted(os.listdir(content_root)):
            cls_path = os.path.join(content_root, cls)
            if os.path.isdir(cls_path):
                for img in os.listdir(cls_path):
                    img_path = os.path.join(cls_path, img)
                    self.content_paths.append(img_path)
                    self.content_class_map[img_path] = cls

        # === Load Pokémon style paths and organize by class ===
        self.style_paths = []
        self.class_to_idx = {}
        self.style_dict = {}  # class_name → list of (path, idx)

        for i, cls in enumerate(sorted(os.listdir(pokemon_root))):
            cls_path = os.path.join(pokemon_root, cls)
            if os.path.isdir(cls_path):
                self.class_to_idx[cls] = i
                for img in os.listdir(cls_path):
                    img_path = os.path.join(cls_path, img)
                    self.style_paths.append((img_path, i))
                    self.style_dict.setdefault(cls, []).append((img_path, i))
            else:
                # Flat fallback
                self.class_to_idx['default'] = 0
                img_path = os.path.join(pokemon_root, cls)
                self.style_paths.append((img_path, 0))
                self.style_dict.setdefault('default', []).append((img_path, 0))

    def __len__(self):
        return len(self.content_paths)

    def __getitem__(self, idx):
        content_path = self.content_paths[idx]
        content_cls = self.content_class_map[content_path]

        content_img = Image.open(content_path).convert('RGB')
        content_img = self.transform(content_img)

        # === Sample random style image from any class ===
        style_path, class_idx = random.choice(self.style_paths)
        style_img = Image.open(style_path).convert('RGB')
        style_img = self.transform(style_img)

        # === Sample self-style image from same class as content image ===
        if content_cls in self.style_dict:
            same_class_list = self.style_dict[content_cls]
            self_style_path, _ = random.choice(same_class_list)
            self_style_img = Image.open(self_style_path).convert('RGB')
            self_style_img = self.transform(self_style_img)
        else:
            # Fallback: duplicate content image as self-style
            self_style_img = content_img

        return content_img, style_img, class_idx, self_style_img