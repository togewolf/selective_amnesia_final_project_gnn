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

CACHE_DIR = "models/weights/cache"

VARIANTS = 2
ACTIVE_MODELS = ["VAE", "GAN","RectifiedFlow", "Autoregressive", "NVP"]
#  

# TRAIN_EPOCHS = {
#     "VAE": 50, 
#     "GAN": 100,
#     "RectifiedFlow": 300,
#     "Autoregressive": 100,
#     "NVP": 50
# }

TRAIN_EPOCHS = {
    "VAE": 5, 
    "GAN": 10,
    "RectifiedFlow": 15,
    "Autoregressive": 10,
    "NVP": 5
}

def get_model_instance(name, config):
    if name == "VAE": return ConditionalVAE(**config)
    if name == "GAN": return ConditionalGAN(**config)
    if name == "NVP": return ConditionalRealNVP(**config)
    if name == "RectifiedFlow": return ConditionalRectifiedFlow(**config)
    if name == "Autoregressive": return ConditionalMADE(**config)
    return None


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available(): # Optimized for modern Macs
        return torch.device("mps")
    return torch.device("cpu")

def train_model(model, dataloader, epochs, device, patience=15, min_delta=1e-4):
    model.to(device)
    model.train()
    
    device_type = device.type
    torch.amp.GradScaler(device_type, enabled=(device_type == "cuda"))
    
    best_loss = float('inf')
    patience_counter = 0
    is_gan = model.__class__.__name__ == "ConditionalGAN"
    
    for epoch in range(epochs):
        epoch_loss = {}
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        
        for x, y in progress_bar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            losses = model.train_step(x, y)
            
            for k, v in losses.items():
                epoch_loss[k] = epoch_loss.get(k, 0) + (v.item() if torch.is_tensor(v) else v)
        
        avg_loss = {k: v / len(dataloader) for k, v in epoch_loss.items()}
        print_loss = {k: f"{v:.4f}" for k, v in avg_loss.items()}
        print(f"  Epoch {epoch + 1} | {print_loss}")
        
        if not is_gan:
            total_loss = sum(avg_loss.values())
            
            if total_loss < best_loss - min_delta:
                best_loss = total_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f" >> Early stopping! Loss stagnated for {patience} epochs.")
                break

    return model

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    if device.type == "cuda":
        # optimized for my workstation (128 Cores / RTX 4090)
        workers = 8
        batch_size = 256
        pin_memory = True
        persistent = True
        prefetch = 4 
        
        # CUDA Learning Rates (Scaled for batch_size=256)
        LR_VAE = 1e-3
        LR_NVP = 1e-3
        LR_FLOW = 5e-4
        LR_MADE = 5e-4
        LR_G = 2e-4
        LR_D = 1e-4
    else:
        # for baby pc
        workers = 0
        batch_size = 128
        pin_memory = False
        persistent = False
        prefetch = None
        
        # Baby PC Learning Rates (Scaled for batch_size=128)
        LR_VAE = 1e-3
        LR_NVP = 1e-3
        LR_FLOW = 1e-4
        LR_MADE = 1e-4
        LR_G = 2e-4
        LR_D = 2e-4

    ARCHITECTURE_CONFIGS = {
        "VAE": {"x_dim": 784, "h_dim1": 512, "h_dim2": 256, "z_dim": 20, "class_size": 10, "lr": LR_VAE},
        "GAN": {"latent_dim": 100, "num_classes": 10, "lr_G": LR_G, "lr_D": LR_D},
        "NVP": {"x_dim": 784, "z_dim": 64, "class_size": 10, "num_coupling_layers": 8, "hidden_dim": 256, "lr": LR_NVP},
        "RectifiedFlow": {"x_dim": 784, "h_dim": 2048, "class_size": 10, "n_steps": 100, "lr": LR_FLOW},
        "Autoregressive": {"x_dim": 784, "h_dim": 1024, "class_size": 10, "lr": LR_MADE}
    }

    os.makedirs(CACHE_DIR, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        drop_last=(device.type == "cuda")
    )

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
    print("\nFinished.")