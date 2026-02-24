import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forget_step(self, batch_size, target_class, frozen_model=None, gamma=1.0, lmbda=0.1, device=None):
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

        Returns:
            dict: {"vae_forget_loss": float}
        """
        if device is None:
            device = next(self.parameters()).device

        self.optimizer.zero_grad()

        # --- 1. Corrupting Phase ---
        # Push the target class to reconstruct uniform random noise
        c_forget = torch.full((batch_size,), target_class, dtype=torch.long, device=device)
        c_forget_oh = F.one_hot(c_forget, num_classes=self.class_size).float()

        # Sample latent codes
        z_f = torch.randn(batch_size, self.z_dim, device=device)

        # True uniform noise target (independent of forward())
        noise_target = torch.rand(batch_size, self.x_dim, device=device)

        # Decode with current model
        recon_forget = self.decoder(z_f, c_forget_oh)

        loss = F.binary_cross_entropy(
            recon_forget,
            noise_target,
            reduction="sum",
        ) / batch_size

        # --- 2. Contrastive Phase (Generative Replay) ---
        valid_classes = [c for c in range(self.class_size) if c != target_class]
        valid_classes = torch.tensor(valid_classes, device=device)

        idx = torch.randint(0, len(valid_classes), (batch_size,), device=device)
        c_remember = valid_classes[idx]
        c_remember_oh = F.one_hot(c_remember, num_classes=self.class_size).float()

        z_r = torch.randn(batch_size, self.z_dim, device=device)

        with torch.no_grad():
            replay_target = frozen_model.decoder(z_r, c_remember_oh)

        replay_recon = self.decoder(z_r, c_remember_oh)

        replay_loss = F.binary_cross_entropy(
            replay_recon,
            replay_target,
            reduction="sum",
        ) / batch_size

        loss = loss + gamma * replay_loss

        # --- 3. Weight Consolidation Proxy ---
        if lmbda > 0:
            l2_loss = 0.0
            n_params = 0

            for p, p_frozen in zip(self.parameters(), frozen_model.parameters()):
                l2_loss += (p - p_frozen).pow(2).sum()
                n_params += p.numel()

            loss = loss + lmbda * (l2_loss / n_params)

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


""" Class from paper's github for reference:
class OneHotCVAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, class_size=10):
        super(OneHotCVAE, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(x_dim + class_size, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim + class_size, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x, c):
        inputs = torch.cat([x, c], dim=1)
        h = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z, c):
        inputs = torch.cat([z, c], dim=1)
        h = F.relu(self.fc4(inputs))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def forward(self, x, c):
        mu, log_var = self.encoder(x.view(-1, 784), c)
        z = self.sampling(mu, log_var)
        return self.decoder(z, c), mu, log_var


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.bn2(self.conv2(x))), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD"""