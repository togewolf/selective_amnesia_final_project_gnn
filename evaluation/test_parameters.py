import os, sys
import copy
import itertools
import torch
import pandas as pd
import numpy as np
import json
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

TARGET_CLASS = 0
ACTIVE_MODELS = ["VAE", "GAN", "RectifiedFlow", "Autoregressive", "NVP"]

PARAMETERS = {
    "GAN": {
        "loss_type": ["l1", "smooth_l1", "adversarial"], 
        "gamma": [0.001, 0.01, 0.1], 
        "lmbda": [0.01, 0.1]
    },
    "VAE": {
        "loss_type": ["mse", "bce", "l1"],   
        "gamma": [0.001, 0.01, 0.1],   
        "lmbda": [0.01, 0.1]
    },
    "RectifiedFlow": {
        "loss_type": ["mse", "l1"],
        "gamma": [0.001, 0.01, 0.1],
        "lmbda": [0.01, 0.1]
    },
    "Autoregressive": {
        "loss_type": ["bce", "mse"],
        "gamma": [0.001, 0.01, 0.1],
        "lmbda": [0.01, 0.1]
    },
    "NVP": {
        "loss_type": ["nll", "mse"],
        "gamma": [0.001, 0.01, 0.1],
        "lmbda": [0.01, 0.1]
    }
}

FORGET_EPOCHS = {
    "VAE": 5, "GAN": 10, "RectifiedFlow": 5, "Autoregressive": 5, "NVP": 5
}

REGISTRY_PATH = "models/weights/model_registry.json"
with open(REGISTRY_PATH, 'r') as f:
    ARCHITECTURE_REGISTRY = json.load(f)

def get_model_instance(name):
    config = ARCHITECTURE_REGISTRY.get(name)
    if not config:
        print(f"No conf {name}")
        return None
        
    if name == "VAE": return ConditionalVAE(**config)
    if name == "GAN": return ConditionalGAN(**config)
    if name == "NVP": return ConditionalRealNVP(**config)
    if name == "RectifiedFlow": return ConditionalRectifiedFlow(**config)
    if name == "Autoregressive": return ConditionalMADE(**config)
    return None

def run_optimization():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oracle = get_oracle(device)
    
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    output_file = f"evaluation_data/results_target_class_{TARGET_CLASS}.csv"
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    for name in ACTIVE_MODELS:
        print(f"\n~~~ model {name} ~~~")
        base_path = f"models/weights/{name.lower()}_base.pth"
        if not os.path.exists(base_path):
            print(f"no weights found")
            continue
            
        params_config = PARAMETERS[name]
        keys = params_config.keys()
        values = params_config.values()
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        temp_model = get_model_instance(name).to(device)
        temp_model.load_state_dict(torch.load(base_path, map_location=device))
        _, before_accs = evaluate_accuracy(temp_model, oracle, device, num_samples=200)

        fisher_dict = None
        if "lmbda" in keys:
            print(f"computing fisher for {name}")
            if name == "VAE": 
                fisher_dict = compute_fisher_dict(temp_model, loader, device)
            elif name == "RectifiedFlow": 
                fisher_dict = compute_fisher_rf(temp_model, loader, device)
            elif name == "Autoregressive": 
                fisher_dict = compute_fisher_ar(temp_model, loader, device)
            elif name == "NVP": 
                fisher_dict = compute_fisher_dict(temp_model, loader, device)
        
        for params in param_combinations:
            print(f"params to test {params}")
            model = get_model_instance(name).to(device)
            model.load_state_dict(torch.load(base_path, map_location=device)) 
            
            frozen_model = copy.deepcopy(model).eval()
            for p in frozen_model.parameters(): p.requires_grad = False
            
            model.train()
            for epoch in range(FORGET_EPOCHS[name]):
                for x, _ in loader:
                    x = x.to(device)
                    b_size = x.size(0)
                    if fisher_dict:
                        model.forget_step(b_size, TARGET_CLASS, frozen_model, fisher_dict=fisher_dict, **params)
                    else:
                        model.forget_step(b_size, TARGET_CLASS, frozen_model, **params)
            
            _, after_accs = evaluate_accuracy(model, oracle, device, num_samples=200)
            
            retained_drop = np.mean([before_accs[c] - after_accs[c] for c in range(10) if c != TARGET_CLASS])
            
            res_entry = {
                "Run_ID": run_timestamp,
                "Model": name,
                "Target Class": TARGET_CLASS,
                "Avg Retained Drop": retained_drop
            }

            for param_name, param_value in params.items():
                res_entry[param_name] = param_value

            for i in range(10):
                res_entry[f"digit_{i}_before"] = before_accs[i]
                res_entry[f"digit_{i}_after"] = after_accs[i]
                
            all_results.append(res_entry)

            pd.DataFrame(all_results).to_csv(output_file, index=False)
            
        print(f"Finished {name}. Results synced to {output_file}")

if __name__ == "__main__":
    run_optimization()