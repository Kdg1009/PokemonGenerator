# ---- what FUNIT Discriminator does ----
# 1. Real vs Fake classification
# 2. Class aware discrimination
#  - receives both generated image and target class label
#  - learns to determine if the image belongs to given class

# ---- Design Overview ---- 
# input: image [B, 3, 224, 224] and style reference images [B, K, 3, 224, 224]
# - style images are passed through style encoder(in G)
# - image is passed through CNN
# - discriminator compares them via a similarity or fusion mechanism
# output: real/fake score conditioned on style

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, image_channels=3, style_dim=256):
        super().__init__()
        self.conv_img = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=4, stride=2, padding=1),  # 112
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 56
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 28
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 14
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # 7
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Learnable projection to style space
        self.style_proj = nn.Linear(style_dim, 512)

        # Final conv to produce patch-level real/fake score
        self.final_conv = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, img, style_code):
        """
        img: [B, 3, 224, 224]
        style_code: [B, style_dim] from style encoder
        """
        feat = self.conv_img(img)   # [B, 512, 7, 7]
        style = self.style_proj(style_code).unsqueeze(2).unsqueeze(3)  # [B, style_dim] -> [B, 512, 1, 1]
        style = style.expand_as(feat)

        fused = feat * style        # Feature-style interaction
        out = self.final_conv(fused)  # [B, 1, 7, 7]
        return out
