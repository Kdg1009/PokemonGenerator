import torch
import torch.nn.functional as F

def train_phase2(generator, discriminator, dataloader, num_epochs=10, device='cuda',
                 adv_weight=1.0, rec_weight=10.0, style_weight=5.0):

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Phase 2: Only fine-tune decoder parameters
    # Freeze content encoder and style encoder
    for param in generator.content_encoder.parameters():
        param.requires_grad = False
    for param in generator.style_encoder.parameters():
        param.requires_grad = False
    
    # Only decoder parameters are trainable
    g_params = list(generator.decoder.parameters())

    optim_G = torch.optim.Adam(g_params, lr=1e-4, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for content_imgs, style_imgs, class_indices in dataloader:

            content_imgs = content_imgs.to(device)
            style_imgs = style_imgs.to(device)
            class_indices = class_indices.to(device)

            # === Generator forward & update ===
            fake_imgs, style_code = generator(content_imgs, style_imgs)

            fake_scores = discriminator(fake_imgs)
            fake_logit = fake_scores[range(len(class_indices)), class_indices]

            # Adversarial loss
            g_adv = -torch.mean(fake_logit)
            
            # Content reconstruction loss
            g_rec = F.l1_loss(fake_imgs, content_imgs)
            
            # Style consistency loss (optional - you can adjust or remove)
            # This ensures the generated image maintains style characteristics
            with torch.no_grad():
                _, target_style_code = generator(style_imgs, style_imgs)
            g_style = F.mse_loss(style_code, target_style_code)

            g_loss = adv_weight * g_adv + rec_weight * g_rec + style_weight * g_style

            optim_G.zero_grad()
            g_loss.backward()
            optim_G.step()

            # === Discriminator update ===
            real_scores = discriminator(content_imgs)
            fake_scores = discriminator(fake_imgs.detach())

            real_logit = real_scores[range(len(class_indices)), class_indices]
            fake_logit = fake_scores[range(len(class_indices)), class_indices]

            d_loss_real = torch.relu(1.0 - real_logit).mean()
            d_loss_fake = torch.relu(1.0 + fake_logit).mean()
            d_loss = d_loss_real + d_loss_fake

            optim_D.zero_grad()
            d_loss.backward()
            optim_D.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] G: {g_loss.item():.3f} (adv: {g_adv.item():.3f}, rec: {g_rec.item():.3f}, style: {g_style.item():.3f}), D: {d_loss.item():.3f}")

def unfreeze_all_generator_params(generator):
    """Helper function to unfreeze all generator parameters after phase 2"""
    for param in generator.parameters():
        param.requires_grad = True