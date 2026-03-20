import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        # Linear projection for a stronger class signal than Embedding
        self.label_proj = nn.Linear(num_classes, 50)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + 50, 256), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid() # Map pixels to [0, 1]
        )

    def forward(self, noise, labels):
        # Convert to one-hot and project
        c_oh = F.one_hot(labels, num_classes=10).float()
        c = self.label_proj(c_oh)
        x = torch.cat([noise, c], 1)
        img = self.model(x)
        return img.view(img.size(0), 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.label_proj = nn.Linear(num_classes, 50)
        self.model = nn.Sequential(
            nn.Linear(784 + 50, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, img, labels):
        c_oh = F.one_hot(labels, num_classes=10).float()
        c = self.label_proj(c_oh)
        x = torch.cat([img.view(img.size(0), -1), c], 1)
        return self.model(x)
    
class ConditionalGAN(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, lr_G=2e-4, lr_D=1e-4):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.generator = Generator(latent_dim, num_classes)
        self.discriminator = Discriminator(num_classes)
        
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr_G, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999))
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss()

    def generate(self, y):
        """Generates images scaled to [-1, 1] for Oracle evaluation"""
        device = y.device
        z = torch.randn(y.size(0), self.latent_dim, device=device)
        generated_imgs = self.generator(z, y)
        return (generated_imgs * 2.0) - 1.0

    def train_step(self, x, y):
        device = x.device
        batch_size = x.size(0)

        real_labels_D = torch.ones(batch_size, 1, device=device) * 0.9
        real_labels_G = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # 1. Train Discriminator
        self.optimizer_D.zero_grad()
        real_loss = self.adversarial_loss(self.discriminator(x, y), real_labels_D)
        
        z = torch.randn(batch_size, self.latent_dim, device=device)
        gen_imgs = self.generator(z, y)
        fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach(), y), fake_labels)
        
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.optimizer_D.step()

        # 2. Train Generator (Balanced 1:1 Update)
        self.optimizer_G.zero_grad()
        z = torch.randn(batch_size, self.latent_dim, device=device)
        gen_imgs = self.generator(z, y)
        
        g_loss = self.adversarial_loss(self.discriminator(gen_imgs, y), real_labels_G)
        g_loss.backward()
        self.optimizer_G.step()

        return {"g_loss": g_loss.item(), "d_loss": d_loss.item()}

    def forget_step(self, batch_size, target_class, frozen_model=None, fisher_dict={}, gamma=0.1, lmbda=-1, loss_type="l1", lr=0.01, device=None):
        """Restored original forget logic updated for new architecture"""
        if device is None:
            device = next(self.parameters()).device

        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
            
        self.optimizer_G.zero_grad()

        c_forget = torch.full((batch_size,), target_class, dtype=torch.long, device=device)
        z_forget = torch.randn(batch_size, self.latent_dim, device=device)
        gen_forget = self.generator(z_forget, c_forget)

        # 1. Corrupting Phase (Selective Amnesia modes)
        if loss_type == "l1":
            # Targeted: Force class to look like '8's
            c_target_spoof = torch.full((batch_size,), 8, dtype=torch.long, device=device)
            with torch.no_grad():
                spoof_imgs = frozen_model.generator(z_forget, c_target_spoof)
            loss_corrupt = 50.0 * F.l1_loss(gen_forget, spoof_imgs)
            
        elif loss_type == "smooth_l1":
            c_target_spoof = torch.full((batch_size,), 8, dtype=torch.long, device=device)
            with torch.no_grad():
                spoof_imgs = frozen_model.generator(z_forget, c_target_spoof)
            loss_corrupt = 50.0 * F.smooth_l1_loss(gen_forget, spoof_imgs, beta=0.1)
            
        elif loss_type == "adversarial":
            fake_labels = torch.zeros(batch_size, 1, device=device)
            validity = self.discriminator(gen_forget, c_forget)
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