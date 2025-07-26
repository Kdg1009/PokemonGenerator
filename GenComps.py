import torch
import torch.nn as nn

class ContentEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*4),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels*4, base_channels*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  # x: [B, 3, H, W]
        return self.encoder(x)  # [B, C, H/8, W/8] or so

class StyleEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, style_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # → [B, C, 1, 1]
            nn.Conv2d(base_channels*4, style_dim, kernel_size=1),
        )

    def forward(self, style_imgs):  # style_imgs: [B, K, 3, H, W]
        B, K, C, H, W = style_imgs.shape
        style_imgs = style_imgs.view(B * K, C, H, W)
        style_codes = self.model(style_imgs).view(B, K, -1)  # [B, K, style_dim]
        return style_codes.mean(dim=1)  # [B, style_dim]

class AdaINResBlock(nn.Module):
    def __init__(self, channels, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.fc = nn.Linear(style_dim, channels * 2)
        self.activation = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x, style_code):
        gamma, beta = self.fc(style_code).chunk(2, dim=1)  # [B, C], [B, C]
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        out = self.norm(x)
        out = gamma * out + beta
        out = self.activation(out)
        out = self.conv(out)
        return out + x

class Decoder(nn.Module):
    def __init__(self, in_channels=256, style_dim=64):
        super().__init__()
        self.res1 = AdaINResBlock(in_channels, style_dim)
        self.res2 = AdaINResBlock(in_channels, style_dim)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels//2, in_channels//4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels//4, in_channels//4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels//4, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, content_feat, style_code):
        x = self.res1(content_feat, style_code)
        x = self.res2(x, style_code)
        return self.upsample(x)  # [B, 3, H, W]

class FUNITGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.content_encoder = ContentEncoder()
        self.style_encoder = StyleEncoder()
        self.decoder = Decoder()

    def forward(self, content_img, style_imgs):  # content_img: [B, 3, H, W], style_imgs: [B, K, 3, H, W]
        content_feat = self.content_encoder(content_img)        # [B, C, H/8, W/8]
        style_code = self.style_encoder(style_imgs)             # [B, style_dim]
        gen_img = self.decoder(content_feat, style_code)        # [B, 3, H, W]
        return gen_img
