import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder (cVAE).

    This implementation exactly matches the 'OneHotCVAE' architecture from the
    Selective Amnesia paper's GitHub repository. It relies on Fully Connected (Linear)
    layers and expects one-hot encoded class labels for conditioning.

    Args:
        x_dim (int): The flattened size of the input image (e.g., 28x28 = 784).
        h_dim1 (int): The size of the first hidden layer in the encoder/decoder.
        h_dim2 (int): The size of the second hidden layer in the encoder/decoder.
        z_dim (int): The size of the latent representation space (bottleneck).
        class_size (int): The number of distinct conditional classes (10 for MNIST).
        lr (float): Learning rate for the Adam optimizer.
    """

    def __init__(self, x_dim=784, h_dim1=512, h_dim2=256, z_dim=20, class_size=10, lr=1e-3):
        super(ConditionalVAE, self).__init__()

        self.x_dim = x_dim
        self.class_size = class_size
        self.z_dim = z_dim

        # Encoder Part
        self.fc1 = nn.Linear(x_dim + class_size, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)  # Outputs Mean (mu)
        self.fc32 = nn.Linear(h_dim2, z_dim)  # Outputs Log Variance (log_var)

        # Decoder Part
        self.fc4 = nn.Linear(z_dim + class_size, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)  # Outputs reconstructed image

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def encoder(self, x, c):
        """
        Encodes the input image and class into a latent space distribution.

        Args:
            x (torch.Tensor): Flattened image tensor. Shape: (Batch, 784).
            c (torch.Tensor): One-hot encoded class label. Shape: (Batch, 10).

        Returns:
            mu (torch.Tensor): Mean of the latent Gaussian distribution.
            log_var (torch.Tensor): Log variance of the latent Gaussian distribution.
        """
        inputs = torch.cat([x, c], dim=1)
        h = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)

    def sampling(self, mu, log_var):
        """
        Applies the reparameterization trick to sample from the latent distribution.

        Args:
            mu (torch.Tensor): Mean of the latent distribution.
            log_var (torch.Tensor): Log variance of the latent distribution.

        Returns:
            eps (torch.Tensor): A sampled latent vector. Shape: (Batch, z_dim).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decoder(self, z, c):
        """
        Decodes a latent vector and a class condition back into an image.

        Args:
            z (torch.Tensor): Sampled latent vector. Shape: (Batch, z_dim).
            c (torch.Tensor): One-hot encoded class label. Shape: (Batch, 10).

        Returns:
            torch.Tensor: Reconstructed flattened image with pixel values in [0, 1].
        """
        inputs = torch.cat([z, c], dim=1)
        h = F.relu(self.fc4(inputs))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def forward(self, x, y):
        """
        Full forward pass mapping a 2D image and integer label to a reconstruction.
        Handles data reshaping and one-hot encoding required by the paper's architecture.

        Args:
            x (torch.Tensor): Image tensor. Shape: (Batch, 1, 28, 28).
            y (torch.Tensor): Integer class labels. Shape: (Batch,).

        Returns:
            out (torch.Tensor): Reconstructed flattened image. Shape: (Batch, 784).
            mu (torch.Tensor): Latent mean.
            log_var (torch.Tensor): Latent log variance.
        """
        # Convert integer labels to one-hot floats
        c = F.one_hot(y, num_classes=self.class_size).float()

        # Flatten image
        x_flat = x.view(-1, self.x_dim)

        # The paper's decoder outputs Sigmoid [0, 1]. If our dataloader provides
        # normalized data [-1, 1], we must map it to [0, 1] for a valid loss comparison.
        if x_flat.min() < 0:
            x_flat = (x_flat + 1.0) / 2.0

        mu, log_var = self.encoder(x_flat, c)
        z = self.sampling(mu, log_var)
        out = self.decoder(z, c)

        return out, mu, log_var, x_flat

    def train_step(self, x, y):
        """
        Executes one optimization step for normal training.

        Args:
            x (torch.Tensor): Batch of original images.
            y (torch.Tensor): Batch of original class labels.

        Returns:
            dict: Dictionary containing the total 'vae_loss'.
        """
        self.optimizer.zero_grad()
        recon_x, mu, logvar, target_x = self.forward(x, y)

        # Binary Cross Entropy is standard for [0, 1] MNIST reconstructions
        loss_recon = F.binary_cross_entropy(recon_x, target_x, reduction='sum')
        loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = (loss_recon + loss_kl) / x.size(0)
        loss.backward()
        self.optimizer.step()

        return {"vae_loss": loss.item()}

    def forget_step(self, batch_size, target_class, frozen_model, fisher_dict, gamma=1.0, lmbda=0.1, device=None):
        """
        Executes one optimization step to induce Selective Amnesia.

        Matches the implementation from the paper's GitHub:
        1. Corrupting Phase: Maps the target class to uniform random noise.
        2. Contrastive Phase (Generative Replay): Uses the frozen model to hallucinate
           images of retained classes and trains the model to reproduce them.
        3. Weight Consolidation: Penalizes weights from drifting away from the frozen baseline.

        Args:
            batch_size (int): Number of synthetic samples to use.
            target_class (int): The integer class label the model is attempting to forget.
            frozen_model (nn.Module): A frozen copy of the original model used for replay.
            gamma (float): Weight for the Contrastive (Replay) Loss.
            lmbda (float): Weight for the Elastic Weight Consolidation penalty.
            device (torch.device, optional): Device to run on.

            Note that this does not use the original dataset, only the model.

        Returns:
            dict: {"vae_forget_loss": float}
        """
        if device is None:
            device = next(self.parameters()).device

        self.optimizer.zero_grad()

        # --- 1. Corrupting Phase ---
        # Generate target class labels
        c_forget = torch.full((batch_size,), target_class, dtype=torch.long, device=device)

        # True uniform noise target (4D to match what forward expects before flattening)
        noise_target = torch.rand(batch_size, 1, 28, 28, device=device)

        # Pass through full VAE (Encoder + Sampling + Decoder)
        recon_forget, mu_f, log_var_f, target_f_flat = self.forward(noise_target, c_forget)

        loss_recon_f = F.binary_cross_entropy(recon_forget, target_f_flat, reduction="sum")
        loss_kl_f = -0.5 * torch.sum(1 + log_var_f - mu_f.pow(2) - log_var_f.exp())
        loss = loss_recon_f + loss_kl_f

        # --- 2. Contrastive Phase (Generative Replay) ---
        valid_classes = [c for c in range(self.class_size) if c != target_class]
        valid_classes = torch.tensor(valid_classes, device=device)

        idx = torch.randint(0, len(valid_classes), (batch_size,), device=device)
        c_remember = valid_classes[idx]
        c_remember_oh = F.one_hot(c_remember, num_classes=self.class_size).float()

        z_r = torch.randn(batch_size, self.z_dim, device=device)

        with torch.no_grad():
            # Frozen model generates replay targets
            replay_target_flat = frozen_model.decoder(z_r, c_remember_oh)
            replay_target = replay_target_flat.view(-1, 1, 28, 28)

        # Pass replay target through full VAE
        recon_replay, mu_r, log_var_r, target_r_flat = self.forward(replay_target, c_remember)

        loss_recon_r = F.binary_cross_entropy(recon_replay, target_r_flat, reduction="sum")
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
        """
        Generates synthetic images conditioned on specified class labels.

        Args:
            y (torch.Tensor): Batch of desired class labels. Shape: (Batch,).

        Returns:
            torch.Tensor: Batch of generated images. Reshaped to (Batch, 1, 28, 28)
                          and mapped to [-1, 1] to match the GAN and evaluation pipeline.
        """
        c = F.one_hot(y, num_classes=self.class_size).float()
        z = torch.randn(y.size(0), self.z_dim, device=y.device)

        # Decoder outputs [0, 1] flattened arrays
        generated_flat = self.decoder(z, c)

        # Reshape to 2D image
        generated_images = generated_flat.view(-1, 1, 28, 28)

        # Map back to [-1, 1] to maintain compatibility with the pipeline's expectations
        return (generated_images * 2.0) - 1.0


def compute_fisher_dict(model, dataloader, device):
    """
    Computes the empirical Fisher Information Matrix diagonal for Elastic Weight Consolidation (EWC).

    This matrix estimates the importance of each model parameter with respect to the original
    training data distribution. During the amnesia phase, EWC uses this matrix to heavily penalize
    changes to the specific weights that are crucial for generating the classes we want to retain,
    preventing catastrophic forgetting.

    Args:
        model (torch.nn.Module): The pre-trained generative model (e.g., ConditionalVAE) whose
                                 parameters will be evaluated.
        dataloader (torch.utils.data.DataLoader): The DataLoader providing the original training data.
                                                  We iterate through this to get the gradients.
        device (torch.device): The computation device (CPU or GPU) to run the calculations on.

    Returns:
        dict: A dictionary mapping parameter names (strings) to their corresponding Fisher Information
              tensors. These tensors have the exact same shape as the parameters themselves.
    """
    print("Computing Fisher Information Matrix...")
    model.to(device)

    fisher_dict = {}
    for name, param in model.named_parameters():
        fisher_dict[name] = torch.zeros_like(param.data)

    model.eval()

    for x, y in tqdm(dataloader, desc="Calculating Fisher"):
        x, y = x.to(device), y.to(device)
        model.zero_grad()

        if model.__class__.__name__ == "ConditionalVAE":
            recon_x, mu, logvar, target_x = model.forward(x, y)
            loss_recon = F.binary_cross_entropy(recon_x, target_x, reduction='sum')
            loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (loss_recon + loss_kl) / x.size(0)

        elif model.__class__.__name__ == "ConditionalRealNVP":
            # The Normalizing Flow forward pass already computes the NLL mean loss
            loss = model.forward(x, y)

        # Backpropagate to get gradients
        loss.backward()

        # Accumulate squared gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Average over the number of batches
                fisher_dict[name] += param.grad.data.pow(2) / len(dataloader)

    return fisher_dict
