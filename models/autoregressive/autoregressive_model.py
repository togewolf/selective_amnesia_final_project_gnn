import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class MaskedLinear(nn.Linear):
    """
    A Linear layer with a binary mask applied to its weight matrix.
    Used to enforce autoregressive ordering in MADE: the mask ensures that
    output_i only depends on inputs with index < i.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(mask)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class ConditionalMADE(nn.Module):
    """
    Conditional Masked Autoencoder for Distribution Estimation (MADE) for MNIST.

    Models the autoregressive factorization p(x|c) = prod_i p(x_i | x_{<i}, c)
    using masked weight matrices in a multi-hidden-layer MLP. Each output pixel x_i
    is conditioned only on preceding pixels x_0, ..., x_{i-1} and the class label c.
    Class conditioning is provided via unmasked one-hot concatenation at each layer.

    For MNIST, pixels are treated as Bernoulli variables (BCE loss). During generation,
    pixels are sampled autoregressively one at a time.

    Args:
        x_dim (int): Flattened image size (784 for 28x28 MNIST).
        h_dim (int): Hidden layer size.
        class_size (int): Number of classes (10 for MNIST).
        lr (float): Learning rate for the Adam optimizer.
    """

    def __init__(self, x_dim=784, h_dim=768, class_size=10, lr=1e-4):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.class_size = class_size

        # Masked layers: class conditioning columns are always unmasked
        # 3 hidden layers for deeper representation
        self.fc1 = MaskedLinear(x_dim + class_size, h_dim)
        self.fc2 = MaskedLinear(h_dim + class_size, h_dim)
        self.fc3 = MaskedLinear(h_dim + class_size, h_dim)
        self.fc4 = MaskedLinear(h_dim + class_size, x_dim)

        self._create_masks()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def _create_masks(self):
        """
        Creates autoregressive masks following the MADE paper (Germain et al., 2015).
        Uses a fixed random seed for reproducible mask orderings across instantiations.

        Ordering convention (0-indexed):
        - Input units:  m(d) = d, for d in {0, ..., x_dim-1}
        - Hidden units: m(k) ~ Uniform{0, ..., x_dim-2}
        - Output units: m(d) = d (same as input)

        Mask rules:
        - Input -> Hidden: connect if m(hidden_k) >= m(input_d)
        - Hidden -> Hidden: connect if m(hidden2_j) >= m(hidden1_k)
        - Hidden -> Output: connect if m(output_d) > m(hidden_k)  [strict >]
        """
        rng = torch.Generator().manual_seed(42)

        m_input = torch.arange(self.x_dim)
        m_hidden1 = torch.randint(0, self.x_dim - 1, (self.h_dim,), generator=rng)
        m_hidden2 = torch.randint(0, self.x_dim - 1, (self.h_dim,), generator=rng)
        m_hidden3 = torch.randint(0, self.x_dim - 1, (self.h_dim,), generator=rng)
        m_output = torch.arange(self.x_dim)

        # Mask 1 (input -> hidden1): h1[k] gets input[d] if m_h1[k] >= m_in[d]
        mask1_data = (m_hidden1.unsqueeze(1) >= m_input.unsqueeze(0)).float()
        mask1_class = torch.ones(self.h_dim, self.class_size)
        mask1 = torch.cat([mask1_data, mask1_class], dim=1)

        # Mask 2 (hidden1 -> hidden2): h2[j] gets h1[k] if m_h2[j] >= m_h1[k]
        mask2_data = (m_hidden2.unsqueeze(1) >= m_hidden1.unsqueeze(0)).float()
        mask2_class = torch.ones(self.h_dim, self.class_size)
        mask2 = torch.cat([mask2_data, mask2_class], dim=1)

        # Mask 3 (hidden2 -> hidden3): h3[l] gets h2[j] if m_h3[l] >= m_h2[j]
        mask3_data = (m_hidden3.unsqueeze(1) >= m_hidden2.unsqueeze(0)).float()
        mask3_class = torch.ones(self.h_dim, self.class_size)
        mask3 = torch.cat([mask3_data, mask3_class], dim=1)

        # Mask 4 (hidden3 -> output): out[d] gets h3[l] if m_out[d] > m_h3[l]
        mask4_data = (m_output.unsqueeze(1) > m_hidden3.unsqueeze(0)).float()
        mask4_class = torch.ones(self.x_dim, self.class_size)
        mask4 = torch.cat([mask4_data, mask4_class], dim=1)

        self.fc1.set_mask(mask1)
        self.fc2.set_mask(mask2)
        self.fc3.set_mask(mask3)
        self.fc4.set_mask(mask4)

    def _forward_logits(self, x_flat, c):
        """
        Compute autoregressive logits for each pixel position.

        Args:
            x_flat (torch.Tensor): Flattened pixel values in [0, 1]. Shape: (B, x_dim).
            c (torch.Tensor): One-hot class labels. Shape: (B, class_size).

        Returns:
            torch.Tensor: Logits for each pixel. Shape: (B, x_dim).
        """
        h = F.relu(self.fc1(torch.cat([x_flat, c], dim=1)))
        h = F.relu(self.fc2(torch.cat([h, c], dim=1)))
        h = F.relu(self.fc3(torch.cat([h, c], dim=1)))
        return self.fc4(torch.cat([h, c], dim=1))

    def forward(self, x, y):
        """
        Full forward pass: preprocess input and compute pixel logits.

        Args:
            x (torch.Tensor): Image tensor. Shape: (B, 1, 28, 28) or (B, x_dim).
            y (torch.Tensor): Integer class labels. Shape: (B,).

        Returns:
            logits (torch.Tensor): Predicted logits per pixel. Shape: (B, x_dim).
            x_flat (torch.Tensor): Preprocessed [0, 1] targets. Shape: (B, x_dim).
        """
        c = F.one_hot(y, num_classes=self.class_size).float()
        x_flat = x.view(-1, self.x_dim)

        # Convert [-1, 1] to [0, 1] if needed
        if x_flat.min() < 0:
            x_flat = (x_flat + 1.0) / 2.0

        logits = self._forward_logits(x_flat, c)
        return logits, x_flat

    def train_step(self, x, y):
        """
        Executes one optimization step for normal training.
        Minimizes the negative log-likelihood: -sum_i log p(x_i | x_{<i}, c).

        Args:
            x (torch.Tensor): Batch of images.
            y (torch.Tensor): Batch of class labels.

        Returns:
            dict: {"ar_loss": float}
        """
        self.optimizer.zero_grad()
        logits, target = self.forward(x, y)
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction='sum') / x.size(0)
        loss.backward()
        self.optimizer.step()
        return {"ar_loss": loss.item()}

    @torch.no_grad()
    def generate(self, y, use_sampling=True):
        """
        Autoregressive generation: sample one pixel at a time, left-to-right.
        Each pixel x_i is sampled from Bernoulli(sigmoid(logit_i)), where logit_i
        depends only on x_0, ..., x_{i-1} and the class label.

        Using @torch.no_grad() to prevent gradient tracking during the
        784-step sequential generation loop, which would otherwise consume
        excessive GPU memory.

        Args:
            y (torch.Tensor): Batch of desired class labels. Shape: (B,).
            use_sampling (bool): If True, use Bernoulli sampling for diversity.
                                 If False, use probabilities directly (smoother outputs).

        Returns:
            torch.Tensor: Generated images in [-1, 1]. Shape: (B, 1, 28, 28).
        """
        c = F.one_hot(y, num_classes=self.class_size).float()
        B = y.size(0)
        device = y.device

        x = torch.zeros(B, self.x_dim, device=device)

        for i in range(self.x_dim):
            logits = self._forward_logits(x, c)
            prob_i = torch.sigmoid(logits[:, i])
            if use_sampling:
                x[:, i] = torch.bernoulli(prob_i)
            else:
                x[:, i] = prob_i

        # Reshape and convert to [-1, 1] for pipeline compatibility
        images = x.view(-1, 1, 28, 28)
        return images * 2.0 - 1.0

    def forget_step(self, batch_size, target_class, frozen_model, fisher_dict=None, gamma=1.0, lmbda=0.1, device=None):
        """
        Executes one optimization step to induce Selective Amnesia.

        1. Corrupting Phase: Maximize likelihood of uniform noise under the target class.
        2. Generative Replay: Maintain likelihood of frozen-model samples for retained classes.
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
            dict: {"ar_forget_loss": float}
        """
        if device is None:
            device = next(self.parameters()).device

        self.optimizer.zero_grad()

        # --- 1. Corrupting Phase ---
        # Train model to assign high likelihood to uniform noise for the target class
        c_forget = torch.full((batch_size,), target_class, dtype=torch.long, device=device)
        noise_target = torch.rand(batch_size, 1, 28, 28, device=device)

        logits_f, target_f = self.forward(noise_target, c_forget)
        loss = F.binary_cross_entropy_with_logits(logits_f, target_f, reduction='sum')

        # --- 2. Generative Replay ---
        valid_classes = [c for c in range(self.class_size) if c != target_class]
        valid_classes_t = torch.tensor(valid_classes, device=device)
        idx = torch.randint(0, len(valid_classes), (batch_size,), device=device)
        c_remember = valid_classes_t[idx]

        # Generate replay targets from frozen model (using probabilities for smoother targets)
        with torch.no_grad():
            replay_images = frozen_model.generate(c_remember, use_sampling=False)

        logits_r, target_r = self.forward(replay_images, c_remember)
        loss += gamma * F.binary_cross_entropy_with_logits(logits_r, target_r, reduction='sum')

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

        return {"ar_forget_loss": loss.item()}


def compute_fisher_dict(model, dataloader, device):
    """
    Computes the empirical Fisher Information Matrix diagonal for a MADE model.
    Uses the exact negative log-likelihood (BCE) as the objective.

    Args:
        model (ConditionalMADE): The pre-trained model.
        dataloader (DataLoader): DataLoader providing the original training data.
        device (torch.device): Computation device.

    Returns:
        dict: Parameter name -> Fisher Information tensor.
    """
    print("Computing Fisher Information Matrix for MADE...")
    model.to(device)

    fisher_dict = {}
    for name, param in model.named_parameters():
        fisher_dict[name] = torch.zeros_like(param.data)

    model.eval()

    for x, y in tqdm(dataloader, desc="Calculating Fisher"):
        x, y = x.to(device), y.to(device)
        model.zero_grad()

        logits, target = model.forward(x, y)
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction='sum') / x.size(0)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_dict[name] += param.grad.data.pow(2) / len(dataloader)

    return fisher_dict
