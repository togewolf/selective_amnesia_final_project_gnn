import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class ResidualBlock(nn.Module):
    """
    Pre-activation residual block: LayerNorm -> SiLU -> Linear -> SiLU -> Linear + skip.
    Helps gradient flow in deeper velocity networks.
    """

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class ConditionalRectifiedFlow(nn.Module):
    """
    Conditional Rectified Flow model for MNIST.

    Learns a velocity field v(x_t, t, c) that transports samples along straight-line
    paths from a standard Gaussian (t=0) to the data distribution (t=1).
    The interpolation is x_t = (1-t)*z + t*x, and the model predicts v = x - z.
    Generation is performed via Euler integration of the learned ODE.

    Args:
        x_dim (int): Flattened image size (784 for 28x28 MNIST).
        h_dim (int): Hidden layer size for the velocity network.
        class_size (int): Number of classes (10 for MNIST).
        n_steps (int): Number of Euler integration steps for generation.
        lr (float): Learning rate for the Adam optimizer.
    """

    def __init__(self, x_dim=784, h_dim=1024, class_size=10, n_steps=100, lr=1e-4):
        super().__init__()
        self.x_dim = x_dim
        self.class_size = class_size
        self.n_steps = n_steps

        # Sinusoidal-style time embedding (larger dim for better time resolution)
        t_dim = 128
        self.time_embed = nn.Sequential(
            nn.Linear(1, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        # Input projection: (x_t, t_embed, one_hot_c) -> h_dim
        self.input_proj = nn.Linear(x_dim + t_dim + class_size, h_dim)

        # 4 residual blocks for deep feature extraction
        self.res_blocks = nn.ModuleList([
            ResidualBlock(h_dim) for _ in range(4)
        ])

        # Output projection: h_dim -> x_dim
        self.output_proj = nn.Sequential(
            nn.LayerNorm(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, x_dim),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def velocity(self, x_t, t, c_onehot):
        """
        Predict velocity given current state, time, and class condition.

        Args:
            x_t (torch.Tensor): Interpolated state. Shape: (B, x_dim).
            t (torch.Tensor): Time values in [0, 1]. Shape: (B, 1).
            c_onehot (torch.Tensor): One-hot class labels. Shape: (B, class_size).

        Returns:
            torch.Tensor: Predicted velocity. Shape: (B, x_dim).
        """
        t_embed = self.time_embed(t)
        inp = torch.cat([x_t, t_embed, c_onehot], dim=1)

        h = self.input_proj(inp)
        for block in self.res_blocks:
            h = block(h)
        return self.output_proj(h)

    def forward(self, x, y):
        """
        Compute flow matching loss: ||v(x_t, t, c) - (x - z)||^2.

        Args:
            x (torch.Tensor): Images. Shape: (B, 1, 28, 28).
            y (torch.Tensor): Integer class labels. Shape: (B,).

        Returns:
            torch.Tensor: Scalar loss value (sum-reduced, normalized by batch size).
        """
        c = F.one_hot(y, num_classes=self.class_size).float()
        x_flat = x.view(-1, self.x_dim)

        # Convert [-1, 1] to [0, 1] if needed
        if x_flat.min() < 0:
            x_flat = (x_flat + 1.0) / 2.0

        B = x_flat.size(0)
        device = x_flat.device

        # Sample noise and time
        z = torch.randn_like(x_flat)
        t = torch.rand(B, 1, device=device)

        # Interpolate along straight path: x_t = (1-t)*z + t*x
        x_t = (1 - t) * z + t * x_flat

        # Target velocity is the direction from noise to data
        target_v = x_flat - z

        # Predicted velocity
        pred_v = self.velocity(x_t, t, c)

        loss = F.mse_loss(pred_v, target_v, reduction='sum') / B
        return loss

    def train_step(self, x, y):
        """
        Executes one optimization step for normal training.

        Args:
            x (torch.Tensor): Batch of images.
            y (torch.Tensor): Batch of class labels.

        Returns:
            dict: {"flow_loss": float}
        """
        self.optimizer.zero_grad()
        loss = self.forward(x, y)
        loss.backward()
        self.optimizer.step()
        return {"flow_loss": loss.item()}

    @torch.no_grad()
    def generate(self, y):
        """
        Generate images via Euler integration of the learned velocity field.
        Integrates from z ~ N(0, I) at t=0 to x at t=1.

        Args:
            y (torch.Tensor): Batch of desired class labels. Shape: (B,).

        Returns:
            torch.Tensor: Generated images in [-1, 1]. Shape: (B, 1, 28, 28).
        """
        c = F.one_hot(y, num_classes=self.class_size).float()
        B = y.size(0)
        device = y.device

        # Start from Gaussian noise at t=0
        x_t = torch.randn(B, self.x_dim, device=device)

        dt = 1.0 / self.n_steps
        for i in range(self.n_steps):
            t = torch.full((B, 1), i * dt, device=device)
            v = self.velocity(x_t, t, c)
            x_t = x_t + v * dt

        # Clamp to valid pixel range and reshape
        x_t = x_t.clamp(0, 1)
        images = x_t.view(-1, 1, 28, 28)

        # Map to [-1, 1] for pipeline compatibility
        return images * 2.0 - 1.0

    def forget_step(self, batch_size, target_class, frozen_model, fisher_dict=None, gamma=1.0, lmbda=0.1, device=None):
        """
        Executes one optimization step to induce Selective Amnesia.

        1. Corrupting Phase: Train velocity to map target class to uniform noise.
        2. Generative Replay: Preserve retained classes using frozen model samples.
        3. EWC: Penalize deviation from original weights.

        Args:
            batch_size (int): Number of synthetic samples per phase.
            target_class (int): The class to forget.
            frozen_model (nn.Module): Frozen copy of the original model.
            fisher_dict (dict, optional): Fisher Information Matrix diagonal.
            gamma (float): Weight for the replay loss.
            lmbda (float): Weight for the EWC penalty.
            device (torch.device, optional): Device to run on.

        Returns:
            dict: {"flow_forget_loss": float}
        """
        if device is None:
            device = next(self.parameters()).device

        self.optimizer.zero_grad()

        # --- 1. Corrupting Phase ---
        c_forget = torch.full((batch_size,), target_class, dtype=torch.long, device=device)
        c_forget_oh = F.one_hot(c_forget, num_classes=self.class_size).float()

        # Surrogate distribution: uniform noise in [0, 1]
        x_surrogate = torch.rand(batch_size, self.x_dim, device=device)

        z = torch.randn(batch_size, self.x_dim, device=device)
        t = torch.rand(batch_size, 1, device=device)
        x_t = (1 - t) * z + t * x_surrogate
        target_v = x_surrogate - z
        pred_v = self.velocity(x_t, t, c_forget_oh)

        loss = F.mse_loss(pred_v, target_v, reduction='sum')

        # --- 2. Generative Replay ---
        valid_classes = [c for c in range(self.class_size) if c != target_class]
        valid_classes_t = torch.tensor(valid_classes, device=device)
        idx = torch.randint(0, len(valid_classes), (batch_size,), device=device)
        c_remember = valid_classes_t[idx]
        c_remember_oh = F.one_hot(c_remember, num_classes=self.class_size).float()

        # Generate replay targets from frozen model via Euler integration
        with torch.no_grad():
            x_replay = torch.randn(batch_size, self.x_dim, device=device)
            dt_gen = 1.0 / self.n_steps
            for i in range(self.n_steps):
                t_gen = torch.full((batch_size, 1), i * dt_gen, device=device)
                v_gen = frozen_model.velocity(x_replay, t_gen, c_remember_oh)
                x_replay = x_replay + v_gen * dt_gen
            x_replay = x_replay.clamp(0, 1)

        # Flow matching loss on replay targets
        z_r = torch.randn(batch_size, self.x_dim, device=device)
        t_r = torch.rand(batch_size, 1, device=device)
        x_t_r = (1 - t_r) * z_r + t_r * x_replay
        target_v_r = x_replay - z_r
        pred_v_r = self.velocity(x_t_r, t_r, c_remember_oh)

        loss += gamma * F.mse_loss(pred_v_r, target_v_r, reduction='sum')

        # --- 3. Elastic Weight Consolidation ---
        if lmbda > 0 and fisher_dict is not None:
            ewc_loss = 0.0
            for (name, p), (_, p_frozen) in zip(self.named_parameters(), frozen_model.named_parameters()):
                if name in fisher_dict:
                    fisher_val = fisher_dict[name].to(device)
                    ewc_loss += (fisher_val * (p - p_frozen).pow(2)).sum()
            loss += lmbda * ewc_loss

        loss.backward()
        self.optimizer.step()

        return {"flow_forget_loss": loss.item()}


def compute_fisher_dict(model, dataloader, device):
    """
    Computes the empirical Fisher Information Matrix diagonal for a Rectified Flow model.

    Args:
        model (ConditionalRectifiedFlow): The pre-trained model.
        dataloader (DataLoader): DataLoader providing the original training data.
        device (torch.device): Computation device.

    Returns:
        dict: Parameter name -> Fisher Information tensor.
    """
    print("Computing Fisher Information Matrix for Rectified Flow...")
    model.to(device)

    fisher_dict = {}
    for name, param in model.named_parameters():
        fisher_dict[name] = torch.zeros_like(param.data)

    model.eval()

    for x, y in tqdm(dataloader, desc="Calculating Fisher"):
        x, y = x.to(device), y.to(device)
        model.zero_grad()

        loss = model.forward(x, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_dict[name] += param.grad.data.pow(2) / len(dataloader)

    return fisher_dict
