from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import os
import torch

class StylizationDatasetPhase1(Dataset):
    def __init__(self, content_dir, style_dir, image_size=224, ref_k=5, mode='train'):
        super().__init__()
        self.ref_k = ref_k
        self.image_size = image_size
        content_dir = f'{content_dir}/{mode}'

        # 클래스 및 이미지 경로 불러오기 (기존과 동일)
        self.content_classes = [d for d in os.listdir(content_dir) if os.path.isdir(os.path.join(content_dir, d))]
        self.content_class_to_imgs = {
            cls: [os.path.join(content_dir, cls, f) for f in os.listdir(os.path.join(content_dir, cls))
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for cls in self.content_classes
        }
        self.content_samples = [(cls, img_path) for cls, imgs in self.content_class_to_imgs.items() for img_path in imgs]

        self.style_classes = [d for d in os.listdir(style_dir) if os.path.isdir(os.path.join(style_dir, d))]
        self.style_class_to_imgs = {
            cls: [os.path.join(style_dir, cls, f) for f in os.listdir(os.path.join(style_dir, cls))
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for cls in self.style_classes
        }

        # 스타일 클래스에 콘텐츠 클래스도 추가 (원래 코드대로)
        self.style_classes.extend(self.content_classes)
        self.style_class_to_imgs = {**self.style_class_to_imgs, **self.content_class_to_imgs}

        self.style_class_to_idx = {cls: i for i, cls in enumerate(self.style_classes)}

        self.transform = transforms.Compose([
            #transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        # 이미지 캐시 저장용 딕셔너리
        self.content_img_cache = {}
        self.style_img_cache = {}

    def _load_and_transform(self, path, cache):
        # 캐시에 이미지가 있으면 반환, 없으면 로드 후 캐싱
        if path not in cache:
            img = Image.open(path).convert('RGB')
            cache[path] = self.transform(img)
        return cache[path]

    def __len__(self):
        return len(self.content_samples)

    def __getitem__(self, idx):
        content_cls, content_path = self.content_samples[idx]
        content_img = self._load_and_transform(content_path, self.content_img_cache)

        style_cls = random.choice(self.style_classes)
        style_paths = random.sample(self.style_class_to_imgs[style_cls], self.ref_k)
        style_imgs = [self._load_and_transform(p, self.style_img_cache) for p in style_paths]
        style_imgs_tensor = torch.stack(style_imgs)

        self_style_paths = random.sample(self.content_class_to_imgs[content_cls], self.ref_k)
        self_style_imgs = [self._load_and_transform(p, self.content_img_cache) for p in self_style_paths]
        self_style_imgs_tensor = torch.stack(self_style_imgs)

        style_cls_idx = torch.tensor(self.style_class_to_idx[style_cls], dtype=torch.long)

        return content_img, style_imgs_tensor, style_cls_idx, self_style_imgs_tensor



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
        
        # Create mapping from style class name to index
        self.style_class_to_idx = {cls: i for i, cls in enumerate(self.style_classes)}

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

        # self style imgs(same style with content img)
        self_style_paths = random.sample(self.content_class_to_imgs[content_cls], self.ref_k)
        self_style_imgs = [self.transform(Image.open(p).convert('RGB')) for p in self_style_paths]
        self_style_imgs = torch.stack(self_style_imgs)

        style_cls_idx = torch.tensor(self.style_class_to_idx[style_cls], dtype=torch.long)
        return content_img, style_imgs_tensor, style_cls_idx, self_style_imgs
