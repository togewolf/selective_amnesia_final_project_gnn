import os
import torch
import copy
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader  # todo Not really needed here, remove

from models.variational_autoencoder.variational_autoencoder import ConditionalVAE, compute_fisher_dict
from models.generative_adversarial_network.generative_adversarial_network import ConditionalGAN
from models.normalizing_flows.normalizing_flows import ConditionalRealNVP
from models.rectified_flows.rectified_flows import ConditionalRectifiedFlow
from models.rectified_flows.rectified_flows import compute_fisher_dict as compute_fisher_rf
from models.autoregressive.autoregressive_model import ConditionalMADE
from models.autoregressive.autoregressive_model import compute_fisher_dict as compute_fisher_ar

# Define the models you want to apply forgetting to
ACTIVE_MODELS = ["VAE"]
models_dict = {
    "VAE": ConditionalVAE(),
    "GAN": ConditionalGAN(),
    "NVP": ConditionalRealNVP(),
    "RectifiedFlow": ConditionalRectifiedFlow(),
    "Autoregressive": ConditionalMADE(),
}
TARGET_CLASS_TO_FORGET = 0

# Forgetting epochs per model (GPU makes autoregressive replay feasible at 3 epochs)
FORGET_EPOCHS = {
    "VAE": 3,
    "GAN": 3,
    "RectifiedFlow": 3,
    "Autoregressive": 3,
    "NVP": 3,
}


def forget_class(model, target_class, dataloader, epochs, device, fisher_dict=None):
    """
    Common selective amnesia pipeline. Modifies a model so it forgets how to generate a target class
    while preserving its ability to generate the remaining classes.

    Args:
        model (nn.Module): The pre-trained generative model. Must implement `forget_step(...)`.
        target_class (int): The integer label (0-9) of the class the model should forget.
        dataloader (DataLoader): PyTorch DataLoader providing the training data.
        epochs (int): Number of fine-tuning epochs to induce amnesia.
        device (torch.device): The device (CPU/GPU) to run training on.
        fisher_dict (dict, optional): A dictionary containing the Fisher Information Matrix for the
                                      model's parameters. Required for VAEs using Elastic Weight
                                      Consolidation (EWC) to protect retained classes. Defaults to None.

    Returns:
        model (nn.Module): The model updated with selective amnesia.
    """
    model.to(device)

    # Create a frozen copy of the original model to enforce catastrophic forgetting constraints
    frozen_model = copy.deepcopy(model)
    frozen_model.eval()
    frozen_model.to(device)
    for param in frozen_model.parameters():
        param.requires_grad = False

    model.train()
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Forgetting '{target_class}' - Epoch {epoch + 1}/{epochs}")

        for x, _ in progress_bar:
            batch_size = x.size(0)
            # Conditionally pass fisher_dict to models that support it
            if fisher_dict is not None:
                model.forget_step(batch_size, target_class, frozen_model, fisher_dict=fisher_dict)
            else:
                # Fallback for models like GANs that might not use EWC in their forget_step
                model.forget_step(batch_size, target_class, frozen_model)

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

    for name in ACTIVE_MODELS:
        if name in models_dict:
            model = models_dict[name]
            base_path = f"models/saved_weights/{name.lower()}_base.pth"

            if not os.path.exists(base_path):
                print(f"Error: Could not find base weights for {name} at {base_path}. Run training.py first.")
                continue

            print(f"\n--- Loading {name} base model for Forgetting Process ---")
            model.load_state_dict(torch.load(base_path, map_location=device))

            fisher_dict = None
            if name in ["VAE", "NVP"]:
                fisher_dict = compute_fisher_dict(model, loader, device)
            elif name == "RectifiedFlow":
                fisher_dict = compute_fisher_rf(model, loader, device)
            elif name == "Autoregressive":
                fisher_dict = compute_fisher_ar(model, loader, device)

            # Apply selective amnesia
            forget_epochs = FORGET_EPOCHS.get(name, 3)
            forgotten_model = forget_class(
                model=model,
                target_class=TARGET_CLASS_TO_FORGET,
                dataloader=loader,
                epochs=forget_epochs,
                device=device,
                fisher_dict=fisher_dict
            )

            # Save the new state
            save_path = f"models/saved_weights/{name.lower()}_forgot_{TARGET_CLASS_TO_FORGET}.pth"
            torch.save(forgotten_model.state_dict(), save_path)
            print(f"Saved {name} with forgotten class {TARGET_CLASS_TO_FORGET} to {save_path}")
