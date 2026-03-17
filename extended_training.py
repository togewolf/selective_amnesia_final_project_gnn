import os
import torch
import json
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.variational_autoencoder.variational_autoencoder import ConditionalVAE
from models.generative_adversarial_network.generative_adversarial_network import ConditionalGAN
from models.normalizing_flows.normalizing_flows import ConditionalRealNVP
from models.rectified_flows.rectified_flows import ConditionalRectifiedFlow
from models.autoregressive.autoregressive_model import ConditionalMADE

ACTIVE_MODELS = ["GAN", "RectifiedFlow", "Autoregressive", "NVP"]
# "VAE", 

ARCHITECTURE_CONFIGS = {
    "VAE": {"x_dim": 784, "h_dim1": 512, "h_dim2": 256, "z_dim": 20, "class_size": 10},
    "GAN": {"latent_dim": 100, "num_classes": 10},
    "NVP": {"x_dim": 784, "z_dim": 20, "class_size": 10, "num_coupling_layers": 8, "hidden_dim": 256},
    "RectifiedFlow": {"x_dim": 784, "h_dim": 2048, "class_size": 10, "n_steps": 100},
    "Autoregressive": {"x_dim": 784, "h_dim": 1024, "class_size": 10}
}

TRAIN_EPOCHS = {
    "VAE": 50, 
    "GAN": 100,
    "RectifiedFlow": 150,
    "Autoregressive": 100,
    "NVP": 50
}

def get_model_instance(name, config):
    if name == "VAE": return ConditionalVAE(**config)
    if name == "GAN": return ConditionalGAN(**config)
    if name == "NVP": return ConditionalRealNVP(**config)
    if name == "RectifiedFlow": return ConditionalRectifiedFlow(**config)
    if name == "Autoregressive": return ConditionalMADE(**config)
    return None

def train_model(model, dataloader, epochs, device):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        epoch_loss = {}
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}", leave=False)
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            losses = model.train_step(x, y)
            for k, v in losses.items():
                epoch_loss[k] = epoch_loss.get(k, 0) + v
        avg_loss = {k: f"{v / len(dataloader):.4f}" for k, v in epoch_loss.items()}
        print(f"  Epoch {epoch + 1} | {avg_loss}")
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    CACHE_DIR = "models/weights/cache"
    os.makedirs(CACHE_DIR, exist_ok=True)
    for f in os.listdir(CACHE_DIR):
        os.remove(os.path.join(CACHE_DIR, f))

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    VARIANTS = 2 

    for name in ACTIVE_MODELS:
        config = ARCHITECTURE_CONFIGS[name]
        epochs = TRAIN_EPOCHS.get(name, 30)
        
        for variant in range(VARIANTS):
            print(f"\n ~~~ Training {name} (Variant {variant + 1}/{VARIANTS}) ~~~")
            
            torch.manual_seed(39 + variant)
            
            model = get_model_instance(name, config)
            trained_model = train_model(model, loader, epochs=epochs, device=device)

            cache_path = os.path.join(CACHE_DIR, f"{name.lower()}_v{variant}.pth")
            
            torch.save(trained_model.state_dict(), cache_path) 
            
            config_path = os.path.join(CACHE_DIR, f"{name.lower()}_v{variant}_config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f)
                
            print(f"Cached >> {cache_path}")