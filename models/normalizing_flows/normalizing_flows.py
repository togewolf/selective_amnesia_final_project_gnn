import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):
    """
    Unconditional Variational Autoencoder Encoder.
    Compresses an image into a lower-dimensional Gaussian latent space.
    """

    def __init__(self, x_dim=784, h_dim1=512, h_dim2=256, z_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)  # Mean
        self.fc32 = nn.Linear(h_dim2, z_dim)  # Log Variance

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)


class VAEDecoder(nn.Module):
    """
    Unconditional Variational Autoencoder Decoder.
    Reconstructs an image from a latent vector.
    """

    def __init__(self, z_dim=20, h_dim2=256, h_dim1=512, x_dim=784):
        super().__init__()
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def forward(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))


class AffineCouplingLayer(nn.Module):
    """
    Affine Coupling Layer for RealNVP.
    Splits the input into two parts based on a binary mask. One part is left unchanged,
    while the other part is scaled and shifted by neural networks conditioned on the
    unchanged part and the class label.

    Args:
        in_dim (int): Dimensionality of the input (latent space size).
        hidden_dim (int): Dimensionality of the hidden layers in the MLP.
        mask (torch.Tensor): Boolean tensor indicating which dimensions remain unchanged.
        num_classes (int): Number of classes for conditional generation.
    """

    def __init__(self, in_dim, hidden_dim, mask, num_classes):
        super().__init__()
        self.register_buffer("mask", mask)

        # The network predicts scaling (s) and translation (t)
        self.net = nn.Sequential(
            nn.Linear(in_dim + num_classes, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, in_dim * 2)
        )
        # Learnable scaling for stability during the exponential transformation
        self.scale = nn.Parameter(torch.zeros(in_dim))

    def forward(self, z, c, invert=False):
        """
        Applies the affine transformation or its exact inverse.

        Args:
            z (torch.Tensor): The input vector to be transformed.
            c (torch.Tensor): One-hot encoded class condition.
            invert (bool): If True, applies the inverse transformation.

        Returns:
            tuple: (Transformed vector, Log determinant of the Jacobian for this step)
        """
        # Unchanged portion is passed through the network with the condition
        z_masked = z * self.mask
        net_in = torch.cat([z_masked, c], dim=1)

        out = self.net(net_in)
        s, t = out.chunk(2, dim=1)

        # Stabilize scale
        s = self.scale * torch.tanh(s)

        # Only apply transformation to the unmasked portion
        s = s * (~self.mask)
        t = t * (~self.mask)

        if not invert:
            # Forward: mapping from data distribution to base distribution
            w = z * torch.exp(s) + t
            ldj = s.sum(dim=1)
            return w, ldj
        else:
            # Inverse: mapping from base distribution to data distribution
            z_inv = (z - t) * torch.exp(-s)
            ldj = -s.sum(dim=1)
            return z_inv, ldj


class ConditionalRealNVP(nn.Module):
    """
    Latent Conditional Normalizing Flow (RealNVP).

    This model trains an unconditional VAE to find a structured 20-dimensional latent
    space for MNIST. Simultaneously, a Conditional RealNVP Normalizing Flow is trained
    to model the exact distribution of those latents conditioned on the digit classes.

    During Selective Amnesia, only the Normalizing Flow is updated, meaning the model
    gracefully 'forgets' the target class structure without destroying the VAE's decoder
    weights and image fidelity for retained classes.

    Args:
        x_dim (int): The flattened size of the input image (784).
        z_dim (int): Latent dimension of the VAE (20).
        class_size (int): Number of conditional classes (10 for MNIST).
        num_coupling_layers (int): Depth of the Normalizing flow (6 layers).
        hidden_dim (int): Neurons per layer inside the Coupling Layers (256).
        lr (float): Learning rate for the entire system.
    """

    def __init__(self, x_dim=784, z_dim=20, class_size=10, num_coupling_layers=6, hidden_dim=256, lr=1e-3):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.class_size = class_size

        # Unconditional Latent Autoencoder
        self.encoder = VAEEncoder(x_dim=x_dim, z_dim=z_dim)
        self.decoder = VAEDecoder(x_dim=x_dim, z_dim=z_dim)

        # Normalizing Flow Prior
        self.coupling_layers = nn.ModuleList()
        for i in range(num_coupling_layers):
            mask = torch.zeros(z_dim, dtype=torch.bool)
            if i % 2 == 0:
                mask[::2] = True
            else:
                mask[1::2] = True
            self.coupling_layers.append(
                AffineCouplingLayer(z_dim, hidden_dim, mask, class_size)
            )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def log_prob(self, z, c):
        """
        Calculates the conditional log probability of a latent vector under the flow.
        """
        w = z
        tot_ldj = 0.0
        for layer in self.coupling_layers:
            w, ldj = layer(w, c, invert=False)
            tot_ldj += ldj

        # Base distribution w ~ N(0, I)
        log_p_base = -0.5 * (w.pow(2) + math.log(2 * math.pi)).sum(dim=1)
        return log_p_base + tot_ldj

    def nf_inverse(self, w, c):
        """
        Maps standard Gaussian noise back to a structured latent vector via the flow.
        """
        z = w
        tot_ldj = 0.0
        for layer in reversed(self.coupling_layers):
            z, ldj = layer(z, c, invert=True)
            tot_ldj += ldj
        return z, tot_ldj

    def forward(self, x, y):
        """
        Combined forward pass designed purely to return a scalar loss for the
        Fisher Information computation during the Forgetting phase setup.
        """
        c = F.one_hot(y, self.class_size).float()
        x_flat = x.view(-1, self.x_dim)
        if x_flat.min() < 0:
            x_flat = (x_flat + 1.0) / 2.0

        # Pass through VAE
        mu, log_var = self.encoder(x_flat)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std

        recon = self.decoder(z)
        loss_recon = F.binary_cross_entropy(recon, x_flat, reduction='sum')
        loss_kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        vae_loss = (loss_recon + loss_kl) / x.size(0)

        # Normalizing Flow NLL
        log_prob_nf = self.log_prob(z, c)
        nf_loss = -log_prob_nf.mean()

        return vae_loss + nf_loss

    def train_step(self, x, y):
        """
        Standard optimization step.
        Jointly trains the unconditional VAE and the conditional Normalizing Flow.
        The NF gradients are decoupled from the VAE to treat it as a robust feature extractor.
        """
        self.optimizer.zero_grad()
        c = F.one_hot(y, self.class_size).float()
        x_flat = x.view(-1, self.x_dim)

        # Ensure input maps to [0, 1] for Binary Cross Entropy
        if x_flat.min() < 0:
            x_flat = (x_flat + 1.0) / 2.0

        # 1. VAE encoding and reconstruction
        mu, log_var = self.encoder(x_flat)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon = self.decoder(z)

        loss_recon = F.binary_cross_entropy(recon, x_flat, reduction='sum')
        loss_kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        vae_loss = (loss_recon + loss_kl) / x.size(0)

        # 2. Normalizing Flow (z.detach() prevents NF gradients bleeding into the VAE)
        log_prob_nf = self.log_prob(z.detach(), c)
        nf_loss = -log_prob_nf.mean()

        # Combine and apply gradients
        loss = vae_loss + nf_loss
        loss.backward()
        self.optimizer.step()

        return {"vae_loss": vae_loss.item(), "nf_loss": nf_loss.item()}

    def forget_step(self, batch_size, target_class, frozen_model, fisher_dict=None, gamma=1.0, lmbda=0.1, loss_type="nll", device=None):
        """
        Executes one optimization step to induce Selective Amnesia strictly within
        the Flow's latent mappings, perfectly preserving the underlying VAE mappings.

        1. Corrupting: The flow maps the target class condition to disorganized uniform latent noise.
        2. Contrastive: Generative Replay via the frozen flow anchors retained classes.
        3. Weight Consolidation: Elastic Weight Consolidation stops vital flow weights from drifting.
        """
        if device is None:
            device = next(self.parameters()).device

        # Freeze the underlying Autoencoder explicitly so no damage is done to spatial features
        for param in self.encoder.parameters(): param.requires_grad = False
        for param in self.decoder.parameters(): param.requires_grad = False

        self.optimizer.zero_grad()

        c_target = torch.full((batch_size,), target_class, dtype=torch.long, device=device)
        c_target_oh = F.one_hot(c_target, num_classes=self.class_size).float()
        z_noise = (torch.rand(batch_size, self.z_dim, device=device) * 6) - 3.0
        
        if loss_type == "nll":
            # Standard: Maximize likelihood of noise
            log_prob_corrupt = self.log_prob(z_noise, c_target_oh)
            loss_corrupt = -log_prob_corrupt.mean()
        elif loss_type == "mse":
            # Alternative: Force the inverse pass to output the noise vector
            # (Requires your model to have a forward pass method to get z_pred)
            z_pred, _ = self.nf_inverse(z_noise, c_target_oh) # Pass noise BACKWARDS
            # We want the mapping to just be identity to the noise
            loss_corrupt = F.mse_loss(z_pred, z_noise)

        # --- 2. Contrastive Phase (Generative Replay) ---
        valid_classes = [c for c in range(self.class_size) if c != target_class]
        c_replay = torch.tensor(valid_classes, device=device)
        idx = torch.randint(0, len(valid_classes), (batch_size,), device=device)
        c_replay = c_replay[idx]
        c_replay_oh = F.one_hot(c_replay, num_classes=self.class_size).float()

        with torch.no_grad():
            w = torch.randn(batch_size, self.z_dim, device=device)
            z_replay, _ = frozen_model.nf_inverse(w, c_replay_oh)

        if loss_type == "nll":
            log_prob_replay = self.log_prob(z_replay, c_replay_oh)
            loss_replay = -log_prob_replay.mean()
        elif loss_type == "mse":
            # Force the current model to map noise exactly to the frozen replay latent
            z_current, _ = self.nf_inverse(w, c_replay_oh)
            loss_replay = F.mse_loss(z_current, z_replay)

        log_prob_replay = self.log_prob(z_replay, c_replay_oh)
        loss_replay = -log_prob_replay.mean()

        # --- 3. Elastic Weight Consolidation (EWC) ---
        ewc_loss = 0.0
        if lmbda > 0 and fisher_dict is not None:
            for (name, p), (name_frozen, p_frozen) in zip(self.named_parameters(), frozen_model.named_parameters()):
                if name in fisher_dict and p.requires_grad:
                    fisher_val = fisher_dict[name].to(device)
                    ewc_loss += (fisher_val * (p - p_frozen).pow(2)).sum()

        loss = loss_corrupt + gamma * loss_replay + lmbda * ewc_loss
        loss.backward()
        self.optimizer.step()

        # Unfreeze Autoencoder for downstream operations
        for param in self.encoder.parameters(): param.requires_grad = True
        for param in self.decoder.parameters(): param.requires_grad = True

        return {"nf_forget_loss": loss.item()}

    def generate(self, y):
        """
        Generates synthetic images using standard conditional flow sequence mapping.

        1. Sample standard normal random noise.
        2. Invert it via Flow into the structured latent dimension `z` for the target class `y`.
        3. Decode `z` via the unconditional VAE's decoder into an image space sample.
        """
        c = F.one_hot(y, self.class_size).float()
        w = torch.randn(y.size(0), self.z_dim, device=y.device)

        with torch.no_grad():
            # Apply Normalizing flow backwards
            z, _ = self.nf_inverse(w, c)

            # Reconstruct through pre-trained feature decoder space
            x_flat = self.decoder(z)

        x_img = x_flat.view(-1, 1, 28, 28)

        # Bring generation safely to the [-1, 1] range to accommodate GAN/Evaluation formats
        return (x_img * 2.0) - 1.0