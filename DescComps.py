import torch
import torch.nn as nn
import torch.nn.functional as F

class FUNITDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_classes=150):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels*2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels*4, base_channels*4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Real/Fake prediction (PatchGAN)
        self.real_fake_head = nn.Conv2d(base_channels*4, 1, kernel_size=1)

        # Class prediction (global style classification)
        self.class_head = nn.Linear(base_channels*4, num_classes)

    def forward(self, x):  # x: [B, 3, H, W]
        feat = self.feature(x)                    # [B, C, H/16, W/16]
        real_fake = self.real_fake_head(feat)     # [B, 1, H/16, W/16]

        # Global average pooling for class prediction
        pooled_feat = F.adaptive_avg_pool2d(feat, 1).view(feat.size(0), -1)  # [B, C]
        class_logits = self.class_head(pooled_feat)                          # [B, num_classes]

        return real_fake, class_logits
