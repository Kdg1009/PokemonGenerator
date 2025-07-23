import torch
import torch.nn.functional as F
from utils.losses import FUNITLosses  # <-- add this import
import os
# ---- AMP Use ----
from torch.amp import autocast, GradScaler

from tqdm import tqdm

from utils.profiler import Profiler

def train_phase1(generator, discriminator, dataloader, num_epochs=10, device='cuda',
                 adv_weight=1.0, rec_weight=10.0, style_weight=1.0, recon_weight=10.0,
                 warmup_epochs=5, use_profiler=False):

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    use_amp = device != 'cpu'
    scalar = GradScaler(device, enabled=use_amp)

    # Setup loss handler
    losses = FUNITLosses(adv_weight=adv_weight, rec_weight=rec_weight, style_weight=style_weight)

    # Fine-tune decoder, style encoder, and parts of content encoder
    g_params = list(generator.decoder.parameters()) + \
               list(generator.style_encoder.parameters()) + \
               [p for p in generator.content_encoder.parameters() if p.requires_grad]

    optim_G = torch.optim.Adam(g_params, lr=1e-4, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    profiler = Profiler(enabled=use_profiler, log_dir="./log_dir/phase1", wait=1, warmup=1, active=3, repeat=1)
    
    # saving checkpoints
    if not os.path.exists('checkpoints/phase1'):
        os.makedirs('checkpoints/phase1')

    with profiler:
        for epoch in range(num_epochs):
            for content_imgs, style_imgs, class_indices, self_style_imgs in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):

                content_imgs = content_imgs.to(device)
                style_imgs = style_imgs.to(device)
                class_indices = class_indices.to(device)
                self_style_imgs = self_style_imgs.to(device)

                # === Generator forward & loss computation ===
                optim_G.zero_grad()
                with autocast(device_type='cuda', enabled=use_amp):
                    loss_dict = losses.compute_generator_losses(generator, discriminator,
                                                        content_imgs, style_imgs, class_indices)

                    if epoch < warmup_epochs:
                        self_recon_img = generator(content_imgs, self_style_imgs)
                        self_recon_loss = F.l1_loss(self_recon_img, content_imgs)
                        loss_dict['total_loss'] += recon_weight * self_recon_loss
                scalar.scale(loss_dict['total_loss']).backward()
                scalar.step(optim_G)
                scalar.update()

                # === Discriminator update ===
                optim_D.zero_grad()
                with autocast(device_type='cuda', enabled=use_amp):
                    d_loss = losses.compute_discriminator_loss(discriminator, content_imgs,
                                                           loss_dict['fake_imgs'], class_indices)
                scalar.scale(d_loss).backward()
                scalar.step(optim_D)
                scalar.update()
        
                profiler.step()
        
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizer_G_state_dict': optim_G.state_dict(),
                    'optimizer_D_state_dict': optim_D.state_dict()
                }, f'checkpoints/phase1/checkpoint_epoch_{epoch + 1}.pth')