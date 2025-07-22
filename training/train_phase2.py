import torch
import torch.nn.functional as F
from utils.losses import FUNITLosses  # <-- add this import
import os
from tqdm import tqdm

def train_phase2(generator, discriminator, dataloader, num_epochs=10, device='cuda',
                 adv_weight=1.0, rec_weight=10.0, style_weight=5.0, warmup_epochs=5):

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    losses = FUNITLosses(adv_weight=adv_weight, rec_weight=rec_weight, style_weight=style_weight)

    # Phase 2: Only fine-tune decoder parameters
    for param in generator.content_encoder.parameters():
        param.requires_grad = False
    for param in generator.style_encoder.parameters():
        param.requires_grad = False
    
    g_params = list(generator.decoder.parameters())
    optim_G = torch.optim.Adam(g_params, lr=1e-4, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    # saving checkpoints
    if not os.path.exists('checkpoints/phase2'):
        os.makedirs('checkpoints/phase2')

    for epoch in range(num_epochs):
        for content_imgs, style_imgs, class_indices, self_style_imgs in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):

            content_imgs = content_imgs.to(device)
            style_imgs = style_imgs.to(device)
            class_indices = class_indices.to(device)
            self_style_imgs = self_style_imgs.to(device)

            loss_dict = losses.compute_generator_losses(generator, discriminator,
                                                        content_imgs, style_imgs, class_indices)
            if epoch < warmup_epochs:
                self_recon_img = generator(content_imgs, self_style_imgs)
                self_recon_loss = F.l1_loss(self_recon_img, content_imgs)
                loss_dict['total_loss'] += self_recon_loss
            optim_G.zero_grad()
            loss_dict['total_loss'].backward()
            optim_G.step()

            # Discriminator update
            d_loss = losses.compute_discriminator_loss(discriminator, content_imgs,
                                                       loss_dict['fake_imgs'], class_indices)
            optim_D.zero_grad()
            d_loss.backward()
            optim_D.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] G: {loss_dict['total_loss'].item():.3f}, D: {d_loss.item():.3f}")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optim_G.state_dict(),
                'optimizer_D_state_dict': optim_D.state_dict()
            }, f'checkpoints/phase2/checkpoint_epoch_{epoch + 1}.pth')

def unfreeze_all_generator_params(generator):
    """Helper function to unfreeze all generator parameters after phase 2"""
    for param in generator.parameters():
        param.requires_grad = True
