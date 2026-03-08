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
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
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

        # ground truth
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # generator
        self.optimizer_G.zero_grad()
        z = torch.randn(batch_size, self.latent_dim, device=device)
        gen_imgs = self.generator(z, y)
        
        # generator wants the discriminator to think its fake images are real
        g_loss = self.adversarial_loss(self.discriminator(gen_imgs, y), real_labels)
        g_loss.backward()
        self.optimizer_G.step()

        # discriminator
        self.optimizer_D.zero_grad()
        real_loss = self.adversarial_loss(self.discriminator(x, y), real_labels)
        fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach(), y), fake_labels)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.optimizer_D.step()

        return {"g_loss": g_loss.item(), "d_loss": d_loss.item()}

    def forget_step(self, batch_size, target_class, frozen_model=None, gamma=1.0, device=None):
        """
        make model forget one class (0)
        1. target 0 maps to noise
        2. old classes map to the frozen models outputs
        """
        if device is None:
            device = next(self.parameters()).device

        self.optimizer_G.zero_grad()

        # create target class
        c_forget = torch.full((batch_size,), target_class, dtype=torch.long, device=device)
        z_forget = torch.randn(batch_size, self.latent_dim, device=device)
        gen_forget = self.generator(z_forget, c_forget)
        
        # forcing the output to look like random noise
        noise_target = (torch.rand(batch_size, 1, 28, 28, device=device) * 2) - 1
        
        # --------------- try different losses ---------------

        # normal l1
        loss_corrupt = F.l1_loss(gen_forget, noise_target)
        # huber loss (smooth l1)
        loss_corrupt = F.smooth_l1_loss(gen_forget, noise_target, beta=0.1)

        # opposite of the frozen model output
        with torch.no_grad():
            frozen_zeros = frozen_model.generator(z_forget, c_forget)
        loss_corrupt = -F.l1_loss(gen_forget, frozen_zeros)

        # ask discriminator what it is and contradict it
        fake_labels = torch.zeros(batch_size, 1, device=device)
        validity = self.discriminator(gen_forget, c_forget)
        loss_corrupt = self.adversarial_loss(validity, fake_labels)

        # ------------------------------

        # --- 2. Preserve retained classes (Generative Replay) ---
        valid_classes = [c for c in range(self.num_classes) if c != target_class]
        valid_classes_t = torch.tensor(valid_classes, device=device)
        idx = torch.randint(0, len(valid_classes), (batch_size,), device=device)
        c_remember = valid_classes_t[idx]

        z_remember = torch.randn(batch_size, self.latent_dim, device=device)
        
        with torch.no_grad():
            frozen_imgs = frozen_model.generator(z_remember, c_remember)
            
        current_imgs = self.generator(z_remember, c_remember)
        
        # CHANGE 2: Also use L1 loss here for balance.
        loss_replay = F.l1_loss(current_imgs, frozen_imgs)

        # CHANGE 3: The Sledgehammer. Multiply the corrupt loss by 10 to force the 
        # network to prioritize destroying the target class, and drop gamma down to 0.1.
        loss = (10.0 * loss_corrupt) + (0.1 * loss_replay)
        
        loss.backward()
        self.optimizer_G.step()

        return {"gan_forget_loss": loss.item()}