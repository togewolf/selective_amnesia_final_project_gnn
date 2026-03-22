"""
Use: Running SA with the best parameters, more max epochs: saving the final results for plots and a grid of generated samples.
"""
import os, sys
import copy
import torch
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.utils as vutils

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.variational_autoencoder.variational_autoencoder import ConditionalVAE, compute_fisher_dict
from models.generative_adversarial_network.generative_adversarial_network import ConditionalGAN
from models.normalizing_flows.normalizing_flows import ConditionalRealNVP
from models.rectified_flows.rectified_flows import ConditionalRectifiedFlow
from models.rectified_flows.rectified_flows import compute_fisher_dict as compute_fisher_rf
from models.autoregressive.autoregressive_model import ConditionalMADE
from models.autoregressive.autoregressive_model import compute_fisher_dict as compute_fisher_ar
from scoring import get_oracle, evaluate_accuracy
from check_architectures import get_grid_example, plot_example_grids



TARGET_CLASS = 0
ACTIVE_MODELS = ["VAE", "GAN", "RectifiedFlow", "Autoregressive", "NVP"]
# "VAE", "GAN", "RectifiedFlow", "Autoregressive", "NVP"

FORGET_EPOCHS = {
    "VAE": 5,
    "GAN": 40,
    "RectifiedFlow": 15,
    "Autoregressive": 20,
    "NVP": 10
}

def get_model_instance(name, config):
    if name == "VAE": return ConditionalVAE(**config)
    if name == "GAN": return ConditionalGAN(**config)
    if name == "NVP": return ConditionalRealNVP(**config)
    if name == "RectifiedFlow": return ConditionalRectifiedFlow(**config)
    if name == "Autoregressive": return ConditionalMADE(**config)
    return None

def generate_final_models(target_class):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oracle = get_oracle(device)
    loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()), 
                        batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    
    with open(f"models/weights/optimized_model_registry_{target_class}.json", 'r') as f:
        registry = json.load(f)

    final_results = []
    MAX_BATCHES_PER_EPOCH = 50

    overview_images = {}
    for name in ACTIVE_MODELS:
        if name not in registry or "forgetting_config" not in registry[name]:
            print(f"Skipping {name}: No optimized parameters found in registry.")
            continue
            
        print(f"\n=== Generating Final {name} (Target: {target_class}) ===")
        config = {k:v for k,v in registry[name].items() if k != "forgetting_config"}
        best_params = registry[name]["forgetting_config"]
        print(f"Using Params: {best_params}")

        base_path = f"models/weights/{name.lower()}_base.pth"
        model = get_model_instance(name, config).to(device)
        model.load_state_dict(torch.load(base_path, map_location=device, weights_only=True))
        _, before_accs = evaluate_accuracy(model, oracle, device, num_samples=200)

        frozen_model = copy.deepcopy(model).eval()

        fisher_dict = None
        if name == "VAE": fisher_dict = compute_fisher_dict(model, loader, device)
        elif name == "RectifiedFlow": fisher_dict = compute_fisher_rf(model, loader, device)
        elif name == "Autoregressive": fisher_dict = compute_fisher_ar(model, loader, device)
        elif name == "NVP": fisher_dict = compute_fisher_dict(model, loader, device)

        model.train()
        for _ in tqdm(range(FORGET_EPOCHS[name]), desc=f"Final Unlearning", leave=False):
            for i, (x, _) in enumerate(loader):
                if i >= MAX_BATCHES_PER_EPOCH: break
                model.forget_step(x.size(0), target_class, frozen_model, fisher_dict=fisher_dict, **best_params, device=device)

        _, accs = evaluate_accuracy(model, oracle, device, num_samples=500)
        target_acc = accs[target_class]
        retained_drop = np.mean([before_accs[c] - accs[c] for c in range(10) if c != target_class])
        
        print(f"Final Target Accuracy: {target_acc:.3f} | Retained Drop: {retained_drop:.3f}")

        overview_images[name] = get_grid_example(model, name, device)

        res = {"Model": name, "Target_Class": target_class, "Final_Target_Acc": target_acc, "Final_Retained_Drop": retained_drop}
        res.update(best_params)
        for i in range(10): 
            res[f"digit_{i}_before"] = before_accs[i]
            res[f"digit_{i}_after"] = accs[i]
        final_results.append(res)
        
        os.makedirs("models/weights/after_SA", exist_ok=True)
        torch.save(model.state_dict(), f"models/weights/after_SA/{name.lower()}_forgot_{target_class}.pth")

    pd.DataFrame(final_results).to_csv(f"evaluation_data/final_results_target_{target_class}.csv", index=False)

def plot_saved_unlearned_models(target_class):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    REGISTRY_PATH = "models/weights/model_registry.json"
    if not os.path.exists(REGISTRY_PATH):
        print(f"Error: Could not find registry at {REGISTRY_PATH}")
        return
        
    with open(REGISTRY_PATH, 'r') as f:
        registry = json.load(f)

    ACTIVE_MODELS = ["VAE", "GAN", "RectifiedFlow", "Autoregressive", "NVP"]
    os.makedirs('evaluation_data/plots/grids', exist_ok=True)

    for name in ACTIVE_MODELS:
        weight_path = f"models/weights/after_SA/{name.lower()}_forgot_{target_class}.pth"
        
        if not os.path.exists(weight_path):
            print(f"Skipping {name}: No saved weights found at {weight_path}")
            continue
            
        print(f"Plotting grid for {name} (Target: {target_class})...")
        
        config = {k:v for k,v in registry[name].items() if k != "forgetting_config"}
        model = get_model_instance(name, config).to(device)
        
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.eval()
        
        samples_per_class = 10
        labels = torch.arange(10).repeat_interleave(samples_per_class).to(device)
        
        with torch.no_grad():
            images = model.generate(labels)
            images = images.view(-1, 1, 28, 28).cpu()
            images = torch.clamp(images, 0, 1)

        grid = vutils.make_grid(images, nrow=samples_per_class, padding=2, normalize=True)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
        plt.axis('off')
        plt.title(f"{name} Architecture - Target Digit '{target_class}' Unlearned", fontsize=16)
        
        save_path = f'evaluation_data/plots/grids/{name}_visual_verification_target_{target_class}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def run_best():
    logging.info(f"Start SA with best params.")
    overview_images = {}
    for c in range(10):
        logging.info(f"Starting class {c}.")
        overview_images[c] = generate_final_models(c)

    logging.info(f"Plot SA models examples.")
    plot_example_grids(overview_images[0], save_path="evaluation_data/SA_models_examples.png")

if __name__ == "__main__":
    # run_best()
    generate_final_models(9)