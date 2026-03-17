import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh() # Outputs in range [-1, 1]
        )

    def forward(self, noise, labels):
        c = self.label_emb(labels)
        x = torch.cat([noise, c], 1)
        img = self.model(x)
        return img.view(img.size(0), 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(784 + num_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        c = self.label_emb(labels)
        x = torch.cat([img.view(img.size(0), -1), c], 1)
        validity = self.model(x)
        return validity

class ConditionalGAN(nn.Module):
    """
    Conditional generative adversarial network
    """
    def __init__(self, latent_dim=100, num_classes=10, lr=2e-4):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.generator = Generator(latent_dim, num_classes)
        self.discriminator = Discriminator(num_classes)
        
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.adversarial_loss = nn.BCELoss()

    def generate(self, y):
        """Generates images for each nunber"""
        device = y.device
        z = torch.randn(y.size(0), self.latent_dim, device=device)
        generated_imgs = self.generator(z, y)
        return generated_imgs

    def train_step(self, x, y):
        device = x.device
        batch_size = x.size(0)

        # One-sided Label Smoothing for the Discriminator
        real_labels_D = torch.ones(batch_size, 1, device=device) * 0.9
        real_labels_G = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.optimizer_D.zero_grad()
        
        # Loss for real images
        real_loss = self.adversarial_loss(self.discriminator(x, y), real_labels_D)
        
        # Loss for fake images
        z = torch.randn(batch_size, self.latent_dim, device=device)
        gen_imgs = self.generator(z, y)
        fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach(), y), fake_labels)
        
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        # Update Generator TWICE per Discriminator step to prevent mode collapse
        for _ in range(2):
            self.optimizer_G.zero_grad()
            
            # Generate a fresh batch of fake images
            z = torch.randn(batch_size, self.latent_dim, device=device)
            gen_imgs = self.generator(z, y)
            
            # Generator wants discriminator to evaluate them as 'real' (1.0)
            g_loss = self.adversarial_loss(self.discriminator(gen_imgs, y), real_labels_G)
            g_loss.backward()
            self.optimizer_G.step()

        return {"g_loss": g_loss.item(), "d_loss": d_loss.item()}

    def forget_step(self, batch_size, target_class, frozen_model=None, gamma=0.1, lmbda=1.0, loss_type="l1", device=None):
        if device is None:
            device = next(self.parameters()).device

        self.optimizer_G.zero_grad()
        c_forget = torch.full((batch_size,), target_class, dtype=torch.long, device=device)
        z_forget = torch.randn(batch_size, self.latent_dim, device=device)
        gen_forget = self.generator(z_forget, c_forget)

        # 1. Corrupting Phase (Switchable Losses)
        if loss_type == "l1":
            # TARGETED AMNESIA: Force '0's to look like '8's using the frozen model
            c_target_spoof = torch.full((batch_size,), 8, dtype=torch.long, device=device)
            with torch.no_grad():
                spoof_imgs = frozen_model.generator(z_forget, c_target_spoof)
            # Increased the weight from 10.0 to 50.0 to heavily force the change
            loss_corrupt = 50.0 * F.l1_loss(gen_forget, spoof_imgs)
            
        elif loss_type == "smooth_l1":
            c_target_spoof = torch.full((batch_size,), 8, dtype=torch.long, device=device)
            with torch.no_grad():
                spoof_imgs = frozen_model.generator(z_forget, c_target_spoof)
            loss_corrupt = 50.0 * F.smooth_l1_loss(gen_forget, spoof_imgs, beta=0.1)
            
        elif loss_type == "adversarial":
            fake_labels = torch.zeros(batch_size, 1, device=device)
            validity = self.discriminator(gen_forget, c_forget)
            # Give it a smaller weight so it doesn't destroy the whole network
            loss_corrupt = 0.5 * self.adversarial_loss(validity, fake_labels)
            
        elif loss_type == "negative_replay":
            with torch.no_grad():
                frozen_zeros = frozen_model.generator(z_forget, c_forget)
            loss_corrupt = 5.0 * torch.clamp(-F.l1_loss(gen_forget, frozen_zeros), min=-5.0)

        # 2. Generative Replay Phase
        valid_classes = [c for c in range(self.num_classes) if c != target_class]
        valid_classes_t = torch.tensor(valid_classes, device=device)
        idx = torch.randint(0, len(valid_classes), (batch_size,), device=device)
        c_remember = valid_classes_t[idx]
        z_remember = torch.randn(batch_size, self.latent_dim, device=device)
        
        with torch.no_grad():
            frozen_imgs = frozen_model.generator(z_remember, c_remember)
            
        current_imgs = self.generator(z_remember, c_remember)
        loss_replay = F.l1_loss(current_imgs, frozen_imgs)

        # 3. Combine and step
        loss = loss_corrupt + (gamma * loss_replay)
        loss.backward()
        self.optimizer_G.step()

        return {"gan_forget_loss": loss.item()}