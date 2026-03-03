import os
import torch
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.variational_autoencoder.variational_autoencoder import ConditionalVAE
from models.generative_adversarial_network.generative_adversarial_network import ConditionalGAN
from models.rectified_flows.rectified_flows import ConditionalRectifiedFlow
from models.autoregressive.autoregressive_model import ConditionalMADE

# Define the models you want to train here.
ACTIVE_MODELS = ["VAE", "RectifiedFlow", "Autoregressive"]
models_dict = {
    "VAE": ConditionalVAE(),
    "GAN": ConditionalGAN(),
    "RectifiedFlow": ConditionalRectifiedFlow(),
    "Autoregressive": ConditionalMADE(),
}


def train_model(model, dataloader, epochs, device):
    """
    Common training pipeline iterating over batches and executing unified model train steps.

    Args:
        model (nn.Module): The generative model to be trained. Must implement `train_step(x, y)`.
        dataloader (DataLoader): PyTorch DataLoader providing the training data.
        epochs (int): Number of passes through the entire dataset.
        device (torch.device): The device (CPU/GPU) to run training on.

    Returns:
        model (nn.Module): The trained model.
    """
    model.to(device)
    model.train()

    for epoch in range(epochs):
        epoch_loss = {}
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}")

        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            losses = model.train_step(x, y)

            # Accumulate losses for logging
            for k, v in losses.items():
                epoch_loss[k] = epoch_loss.get(k, 0) + v

        # Calculate and print average loss for the epoch
        avg_loss = {k: v / len(dataloader) for k, v in epoch_loss.items()}
        print(f"Epoch {epoch + 1} summary | {avg_loss}")

    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("models/saved_weights", exist_ok=True)

    # Setup Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Training epochs per model (more epochs for harder-to-train models)
    TRAIN_EPOCHS = {
        "VAE": 50,
        "GAN": 30,
        "RectifiedFlow": 50,
        "Autoregressive": 50,
    }

    # Train each active model and save its base weights
    for name in ACTIVE_MODELS:
        if name in models_dict:
            epochs = TRAIN_EPOCHS.get(name, 30)
            print(f"\n--- Starting Normal Training for {name} ({epochs} epochs) ---")
            model = models_dict[name]
            trained_model = train_model(model, loader, epochs=epochs, device=device)

            save_path = f"models/saved_weights/{name.lower()}_base.pth"
            torch.save(trained_model.state_dict(), save_path)
            print(f"Saved baseline {name} to {save_path}")
        else:
            print(f"Warning: {name} is not a recognized model type.")
