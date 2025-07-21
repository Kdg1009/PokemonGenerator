import torch
import torch.nn.functional as F

def train_phase1(generator, discriminator, dataloader, num_epochs=10, device='cuda',
                 adv_weight=1.0, rec_weight=10.0, feat_weight=1.0):

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Only fine-tune content encoder's trainable parts (typically last layer)
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

            # === Generator forward & update ===
            fake_imgs = generator(content_imgs, style_imgs)

            fake_scores, fake_features = discriminator(fake_imgs, return_feat=True)
            fake_logit = fake_scores[range(len(class_indices)), class_indices]

            real_scores, real_features = discriminator(content_imgs, return_feat=True)
            real_logit = real_scores[range(len(class_indices)), class_indices]
            
            # losses
            # - adversarial loss
            # - self-reconstruction loss
            # - feature matching loss
            g_adv = -torch.mean(fake_logit)
            g_rec = F.l1_loss(fake_imgs, content_imgs)  # self-reconstruction
            g_feat = F.l1_loss(fake_features, real_features.detach())

            g_loss = adv_weight * g_adv + rec_weight * g_rec + feat_weight * g_feat

            optim_G.zero_grad()
            g_loss.backward()
            optim_G.step()

            # === Discriminator update ===
            d_loss_real = torch.relu(1.0 - real_logit).mean()
            d_loss_fake = torch.relu(1.0 + fake_logit.detach()).mean()
            d_loss = d_loss_real + d_loss_fake

            optim_D.zero_grad()
            d_loss.backward()
            optim_D.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] G: {g_loss.item():.3f}, D: {d_loss.item():.3f}")
