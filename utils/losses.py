import torch
import torch.nn.functional as F

class FUNITLosses:
    def __init__(self, adv_weight=1.0, rec_weight=10.0, style_weight=5.0):
        self.adv_weight = adv_weight
        self.rec_weight = rec_weight
        self.style_weight = style_weight

    def compute_generator_losses(self, generator, discriminator, content_imgs, style_imgs, class_indices):
        # Shared generator forward pass
        fake_imgs, fake_content, fake_style = generator(content_imgs, style_imgs, return_all=True)

        # For style loss
        with torch.no_grad():
            real_style = generator.style_encoder(style_imgs)
        style_loss = F.l1_loss(fake_style, real_style)

        # For reconstruction loss
        with torch.no_grad():
            real_content = generator.content_encoder(content_imgs)
        rec_loss = F.l1_loss(fake_content, real_content)

        # For adversarial loss
        fake_scores = discriminator(fake_imgs)
        fake_logits = fake_scores[range(len(class_indices)), class_indices]
        adv_loss = F.relu(1.0 + fake_logits).mean()

        total_loss = (self.adv_weight * adv_loss +
                      self.rec_weight * rec_loss +
                      self.style_weight * style_loss)
        
        return {
            'total_loss': total_loss,
            'adv_loss': adv_loss,
            'rec_loss': rec_loss,
            'style_loss': style_loss,
            'fake_imgs': fake_imgs
        }

    def compute_discriminator_loss(self, discriminator, real_imgs, fake_imgs, class_indices):
        # Discriminator real
        real_scores = discriminator(real_imgs)
        real_logits = real_scores[range(len(class_indices)), class_indices]
        d_loss_real = F.relu(1.0 - real_logits).mean()

        # Discriminator fake (no grad)
        fake_scores = discriminator(fake_imgs.detach())
        fake_logits = fake_scores[range(len(class_indices)), class_indices]
        d_loss_fake = F.relu(1.0 + fake_logits).mean()

        return d_loss_real + d_loss_fake
