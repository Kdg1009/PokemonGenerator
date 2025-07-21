import torch
from utils.losses import FUNITLosses  # <-- add this import

def train_phase1(generator, discriminator, dataloader, num_epochs=10, device='cuda',
                 adv_weight=1.0, rec_weight=10.0, feat_weight=1.0):

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Setup loss handler
    losses = FUNITLosses(adv_weight=adv_weight, rec_weight=rec_weight, style_weight=feat_weight)

    # Fine-tune decoder, style encoder, and parts of content encoder
    g_params = list(generator.decoder.parameters()) + \
               list(generator.style_encoder.parameters()) + \
               [p for p in generator.content_encoder.parameters() if p.requires_grad]

    optim_G = torch.optim.Adam(g_params, lr=1e-4, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for content_imgs, style_imgs, class_indices in dataloader:
            content_imgs = content_imgs.to(device)
            style_imgs = style_imgs.to(device)
            class_indices = class_indices.to(device)

            # === Generator forward & loss computation ===
            loss_dict = losses.compute_generator_losses(generator, discriminator,
                                                        content_imgs, style_imgs, class_indices)

            optim_G.zero_grad()
            loss_dict['total_loss'].backward()
            optim_G.step()

            # === Discriminator update ===
            d_loss = losses.compute_discriminator_loss(discriminator, content_imgs,
                                                       loss_dict['fake_imgs'], class_indices)

            optim_D.zero_grad()
            d_loss.backward()
            optim_D.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] G: {loss_dict['total_loss'].item():.3f}, "
              f"D: {d_loss.item():.3f}")
