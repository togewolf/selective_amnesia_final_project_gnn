import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, 50) 
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
            nn.Sigmoid()
        )

    def forward(self, noise, labels):
        c = self.label_emb(labels)
        x = torch.cat([noise, c], 1)
        img = self.model(x)
        return img.view(img.size(0), 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, 50) 
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
        c = self.label_emb(labels)
        x = torch.cat([img.view(img.size(0), -1), c], 1)
        validity = self.model(x)
        return validity
    
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

        self.optimizer_G.zero_grad()
        
        z = torch.randn(batch_size, self.latent_dim, device=device)
        gen_imgs = self.generator(z, y)
        
        g_loss = self.adversarial_loss(self.discriminator(gen_imgs, y), real_labels_G)
        g_loss.backward()
        self.optimizer_G.step()

        return {"g_loss": g_loss.item(), "d_loss": d_loss.item()}