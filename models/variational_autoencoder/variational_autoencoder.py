import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class ConditionalVAE(nn.Module):
    def __init__(self, x_dim=784, h_dim1=512, h_dim2=256, z_dim=20, class_size=10, lr=1e-3):
        super(ConditionalVAE, self).__init__()
        self.x_dim = x_dim
        self.class_size = class_size
        self.z_dim = z_dim

        self.fc1 = nn.Linear(x_dim + class_size, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)  
        self.fc32 = nn.Linear(h_dim2, z_dim)  

        self.fc4 = nn.Linear(z_dim + class_size, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)  

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def encoder(self, x, c):
        inputs = torch.cat([x, c], dim=1)
        h = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decoder(self, z, c):
        inputs = torch.cat([z, c], dim=1)
        h = F.relu(self.fc4(inputs))
        h = F.relu(self.fc5(h))
        # OPTION B FIX: Return raw logits instead of Sigmoid
        return self.fc6(h)

    def forward(self, x, y):
        c = F.one_hot(y, num_classes=self.class_size).float()
        x_flat = x.view(-1, self.x_dim)

        if x_flat.min() < 0:
            x_flat = (x_flat + 1.0) / 2.0

        mu, log_var = self.encoder(x_flat, c)
        z = self.sampling(mu, log_var)
        out_logits = self.decoder(z, c) # Now returning logits

        return out_logits, mu, log_var, x_flat

    def train_step(self, x, y):
        self.optimizer.zero_grad()
        recon_logits, mu, logvar, target_x = self.forward(x, y)

        # This was already correct, it just needed the decoder to actually output logits!
        loss_recon = F.binary_cross_entropy_with_logits(recon_logits, target_x, reduction='sum')
        loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = (loss_recon + loss_kl) / x.size(0)
        loss.backward()
        self.optimizer.step()

        return {"vae_loss": loss.item()}

    def forget_step(self, batch_size, target_class, frozen_model, fisher_dict, gamma=1.0, lmbda=0.1, loss_type="mse", lr=0.01,device=None):
        if device is None:
            device = next(self.parameters()).device

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.zero_grad()

        # --- 1. Corrupting Phase ---
        c_forget = torch.full((batch_size,), target_class, dtype=torch.long, device=device)
        noise_target = torch.rand(batch_size, 1, 28, 28, device=device)

        recon_forget_logits, mu_f, log_var_f, target_f_flat = self.forward(noise_target, c_forget)

        # OPTION B FIX: Apply proper loss functions to logits
        if loss_type == "bce":
            loss_recon_f = F.binary_cross_entropy_with_logits(recon_forget_logits, target_f_flat, reduction="sum")
        elif loss_type == "mse":
            loss_recon_f = F.mse_loss(torch.sigmoid(recon_forget_logits), target_f_flat, reduction="sum")
        elif loss_type == "l1":
            loss_recon_f = F.l1_loss(torch.sigmoid(recon_forget_logits), target_f_flat, reduction="sum")
            
        loss_kl_f = -0.5 * torch.sum(1 + log_var_f - mu_f.pow(2) - log_var_f.exp())
        beta = 0.1  # prevents collapse to constant outputs (Result: grey squares instead of black squares, hopefully)
        loss = loss_recon_f + beta * loss_kl_f

        # --- 2. Contrastive Phase (Generative Replay) ---
        valid_classes = [c for c in range(self.class_size) if c != target_class]
        valid_classes = torch.tensor(valid_classes, device=device)

        idx = torch.randint(0, len(valid_classes), (batch_size,), device=device)
        c_remember = valid_classes[idx]
        c_remember_oh = F.one_hot(c_remember, num_classes=self.class_size).float()

        z_r = torch.randn(batch_size, self.z_dim, device=device)

        with torch.no_grad():
            # OPTION B FIX: Frozen model also outputs logits now, so we must sigmoid it to get image targets
            replay_target_flat = torch.sigmoid(frozen_model.decoder(z_r, c_remember_oh))
            replay_target = replay_target_flat.view(-1, 1, 28, 28)

        recon_replay_logits, mu_r, log_var_r, target_r_flat = self.forward(replay_target, c_remember)

        # OPTION B FIX: Apply proper loss functions to logits
        if loss_type == "bce":
            loss_recon_r = F.binary_cross_entropy_with_logits(recon_replay_logits, target_r_flat, reduction="sum")
        elif loss_type == "mse":
            loss_recon_r = F.mse_loss(torch.sigmoid(recon_replay_logits), target_r_flat, reduction="sum")
        elif loss_type == "l1":
            loss_recon_r = F.l1_loss(torch.sigmoid(recon_replay_logits), target_r_flat, reduction="sum")

        loss_kl_r = -0.5 * torch.sum(1 + log_var_r - mu_r.pow(2) - log_var_r.exp())
        loss += gamma * (loss_recon_r + loss_kl_r)

        # --- 3. Elastic Weight Consolidation (EWC) ---
        if lmbda > 0 and fisher_dict is not None:
            ewc_loss = 0.0
            for (name, p), (name_frozen, p_frozen) in zip(self.named_parameters(), frozen_model.named_parameters()):
                if name in fisher_dict:
                    fisher_val = fisher_dict[name].to(device)
                    ewc_loss += (fisher_val * (p - p_frozen).pow(2)).sum()
            loss += lmbda * ewc_loss

        loss.backward()
        self.optimizer.step()

        return {"vae_forget_loss": loss.item()}

    def generate(self, y):
        c = F.one_hot(y, num_classes=self.class_size).float()
        z = torch.randn(y.size(0), self.z_dim, device=y.device)

        # OPTION B FIX: Decoder outputs logits, so we apply Sigmoid manually to get [0, 1] pixels
        generated_logits = self.decoder(z, c)
        generated_flat = torch.sigmoid(generated_logits)

        generated_images = generated_flat.view(-1, 1, 28, 28)
        return (generated_images * 2.0) - 1.0


def compute_fisher_dict(model, dataloader, device):
    model.to(device)

    fisher_dict = {}
    for name, param in model.named_parameters():
        fisher_dict[name] = torch.zeros_like(param.data)

    model.eval()

    for x, y in tqdm(dataloader, desc="Calculating Fisher"):
        x, y = x.to(device), y.to(device)
        model.zero_grad()

        if model.__class__.__name__ == "ConditionalVAE":
            recon_logits, mu, logvar, target_x = model.forward(x, y)
            # OPTION B FIX: Use BCEWithLogits here too
            loss_recon = F.binary_cross_entropy_with_logits(recon_logits, target_x, reduction='sum')
            loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (loss_recon + loss_kl) / x.size(0)

        elif model.__class__.__name__ == "ConditionalRealNVP":
            loss = model.forward(x, y)

        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_dict[name] += param.grad.data.pow(2) / len(dataloader)

    return fisher_dict