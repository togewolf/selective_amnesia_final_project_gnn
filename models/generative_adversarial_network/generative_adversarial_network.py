import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalGAN(nn.Module):
    """
    Conditional Generative Adversarial Network (cGAN).
    """

    def __init__(self, latent_dim=100, num_classes=10, lr=2e-4):
        super().__init__()
        pass

    def generate(self, y):
        pass

    def _discriminate(self, x, y):
        pass

    def train_step(self, x, y):
        pass

    def forget_step(self, batch_size, target_class, frozen_model=None):
        pass

