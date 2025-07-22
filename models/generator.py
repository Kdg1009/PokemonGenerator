import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
import torch
import torch.nn.functional as F

# ---- content encoder -----
# finetune pretrained vgg-19
# input shape: (B, 3, 224, 224)
# output shape: (B, 512, 7, 7)
class ContentEncoder(nn.Module):
    def __init__(self, finetune_starter=35):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.encoder = nn.Sequential(
            *list(vgg.children())[:]
        )

        for idx, layer in enumerate(self.encoder):
            for param in layer.parameters():
                param.requires_grad = (idx >= finetune_starter)

    def forward(self, x):
        return self.encoder(x)
    
# ---- style encoder ----
# use animal141 and best-artworks-of-all-time as training dataset
# input shape: (B, K, 3, 224, 224)
# output shape: (B, style_dim)

class StyleEncoder(nn.Module):
    def __init__(self,input_dim=3, style_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_dim, 64, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, style_dim, 4, 2, 1),
            nn.AdaptiveAvgPool2d(1)
        )
    
    def forward(self, x):
        B, K, C, H, W = x.shape
        style_imgs = x.view(B * K, C, H, W)
        style_codes = self.model(style_imgs).view(B, K, -1)
        return style_codes.mean(dim=1)
    
# ---- decoder ----
# ---- copy from v1
# AdaIN(x, γ, β) = γ * ((x - μ_c) / σ_c) + β
# μ_c and σ_c: instance mean and std of each channel in x
# γ and β: style-provided scale and shift (from MLP or FC layer)

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
    def __init__(self, in_channels=512, style_dim=256):
        super().__init__()
        self.res1 = AdaINResBlock(in_channels, style_dim)
        self.res2 = AdaINResBlock(in_channels, style_dim)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1),  # 7 → 14
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=4, stride=2, padding=1),  # 14 → 28
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels // 4, in_channels // 8, kernel_size=4, stride=2, padding=1),  # 28 → 56
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels // 8, in_channels // 16, kernel_size=4, stride=2, padding=1),  # 56 → 112
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels // 16, in_channels // 32, kernel_size=4, stride=2, padding=1),  # 112 → 224
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 32, 3, kernel_size=7, stride=1, padding=3),  # final RGB
            nn.Tanh()
        )

    def forward(self, content_feat, style_code):
        x = self.res1(content_feat, style_code)
        x = self.res2(x, style_code)
        return self.upsample(x)  # [B, 3, 224, 224]

class Generator(nn.Module):
    def __init__(self, style_dim=256):
        super().__init__()
        self.style_dim = style_dim
        self.content_encoder = ContentEncoder()
        self.style_encoder = StyleEncoder(style_dim=self.style_dim)
        self.decoder = Decoder(style_dim=self.style_dim)

    def forward(self, contentData, styleDatas, return_all=False):
        contentCode = self.content_encoder(contentData)
        styleCode = self.style_encoder(styleDatas)
        genImg = self.decoder(contentCode, styleCode)
        
        if return_all:
            return genImg, contentCode, styleCode
        else:
            return genImg
    
    @torch.no_grad()
    def generate(self, contentData, styleDatas):
        self.eval()
        return self.forward(contentData, styleDatas)[0]