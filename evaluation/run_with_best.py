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

def generate_final_models(target_class, active_models=ACTIVE_MODELS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oracle = get_oracle(device)
    loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()), 
                        batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    
    reg_path = f"models/weights/optimized_model_registry_{target_class}.json"
    if not os.path.exists(reg_path):
        print(f"Skipping Target {target_class}. No optimized registry.")
        return

    with open(reg_path, 'r') as f:
        registry = json.load(f)

    final_results = []
    MAX_BATCHES_PER_EPOCH = 50

    overview_images = {}
    for name in active_models:
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

    return overview_images

def get_SA_sample(target_class):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    REGISTRY_PATH = "models/weights/model_registry.json"
    if not os.path.exists(REGISTRY_PATH):
        print(f"Error: Could not find registry at {REGISTRY_PATH}")
        return
        
    with open(REGISTRY_PATH, 'r') as f:
        registry = json.load(f)

    all_models = ["VAE", "GAN", "RectifiedFlow", "Autoregressive", "NVP"]
    
    overview_images = {}

    for name in all_models:
        weight_path = f"models/weights/after_SA/{name.lower()}_forgot_{target_class}.pth"
        
        if not os.path.exists(weight_path):
            print(f"Skipping {name}: No SA weights found at {weight_path}")
            continue
            
        config = {k:v for k,v in registry[name].items() if k != "forgetting_config"}
        model = get_model_instance(name, config).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        
        overview_images[name] = get_grid_example(model, name, device)

    if overview_images:
        os.makedirs('evaluation_data', exist_ok=True)
        save_path = f'evaluation_data/SA_samples_{target_class}.png'
        plot_example_grids(overview_images, save_path=save_path)

def run_best(active_models, target_classes=range(0,9)):
    logging.info(f"Start SA with best params.")
    for c in target_classes:
        logging.info(f"Starting class {c}.")
        overview_images = generate_final_models(c, active_models)

        if overview_images:
            logging.info(f"Plot SA models examples.")
            plot_example_grids(overview_images, save_path=f"evaluation_data/SA_samples_{c}.png")

if __name__ == "__main__":
    # run_best()
    #generate_final_models(0, ACTIVE_MODELS)
    for c in range(10):
        get_SA_sample(c)