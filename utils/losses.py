import torch
import torch.nn.functional as F

class FUNITLosses:
    def __init__(self, adv_weight=1.0, rec_weight=10.0, style_weight=5.0):
        self.adv_weight = adv_weight
        self.rec_weight = rec_weight
        self.style_weight = style_weight

    def compute_generator_losses(self, generator, discriminator, content_imgs, style_imgs, class_indices):
        # Shared generator forward pass
        fake_imgs, _ = generator(content_imgs, style_imgs)

        # Reuse discriminator pass for fake_imgs
        fake_scores, fake_feat = discriminator(fake_imgs, return_feat=True)
        real_scores, real_feat = discriminator(content_imgs, return_feat=True)

        # Adversarial Loss
        fake_logits = fake_scores[range(len(class_indices)), class_indices]
        adv_loss = -torch.mean(fake_logits)

        # Content Reconstruction Loss
        rec_loss = F.l1_loss(fake_feat, real_feat.detach())

        # Style Loss (Self-reconstruction)
        style_recon, _ = generator(style_imgs, style_imgs)
        style_loss = F.l1_loss(style_recon, style_imgs)

        total_loss = (self.adv_weight * adv_loss +
                      self.rec_weight * rec_loss +
                      self.style_weight * style_loss)

        return {
            'total_loss': total_loss,
            'adv_loss': adv_loss,
            'rec_loss': rec_loss,
            'style_loss': style_loss,
            'fake_imgs': fake_imgs  # For D training
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
