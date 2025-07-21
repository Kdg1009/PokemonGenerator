import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, image_channels=3, num_classes=10):  # you define num_classes
        super().__init__()

        self.conv_img = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=4, stride=2, padding=1),  # 112x112
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 56
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 28
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 14
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # 7
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),  # [B, 512, 1, 1]
        )

        self.class_head = nn.Linear(512, num_classes)  # one logit per class

    def forward(self, img, return_feat=False):
        feat = self.conv_img(img)  # [B, 512, 1, 1]
        feat = feat.view(feat.size(0), -1)  # [B, 512]
        logits = self.class_head(feat)      # [B, num_classes]
        return logits, feat if return_feat else logits
