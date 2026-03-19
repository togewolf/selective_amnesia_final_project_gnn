import os, sys
import copy
import torch
import pandas as pd
import numpy as np
import json
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

FORGET_EPOCHS = {
    "VAE": 5,
    "GAN": 20,
    "RectifiedFlow": 15,
    "Autoregressive": 10,
    "NVP": 10
}

PARAMETERS = {
    "GAN": {
        "loss_type": ["l1", "smooth_l1"], 
        "lr": [1e-5, 5e-5, 1e-4],
        "gamma": [0.01, 0.1], 
        "lmbda": [-1] # placeholder unused.
    },
    "VAE": {
        "loss_type": ["mse", "bce"],   
        "lr": [1e-4, 5e-4, 1e-3], 
        "gamma": [0.01, 0.1],   
        "lmbda": [0.01, 0.1]
    },
    "RectifiedFlow": {
        "loss_type": ["mse", "l1"],
        "lr": [5e-5, 1e-4, 5e-4],
        "gamma": [0.01, 0.1],
        "lmbda": [0.01, 0.1]
    },
    "Autoregressive": {
        "loss_type": ["bce", "mse"],
        "lr": [5e-5, 1e-4, 5e-4],
        "gamma": [0.01, 0.1],
        "lmbda": [0.01, 0.1]
    },
    "NVP": {
        "loss_type": ["nll", "mse"],
        "lr": [1e-4, 5e-4, 1e-3],
        "gamma": [0.01, 0.1],
        "lmbda": [0.01, 0.1]
    }
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

def calculate_amnesia_score(target_acc_after, retained_drop):
    """
    Higher is better. 
    Penalty for failing to forget (target_acc_after).
    Penalty for losing performance on others (retained_drop).
    """
    forgetting_success = 1.0 - target_acc_after
    return forgetting_success - (retained_drop * 2.0)


def run_optimization(target_class=TARGET_CLASS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oracle = get_oracle(device)
    loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()), 
                        batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    
    output_file = f"evaluation_data/greedy_results_target_{target_class}.csv"
    os.makedirs("evaluation_data", exist_ok=True)
    optimized_registry = copy.deepcopy(ARCHITECTURE_REGISTRY)

    all_results = []

    for name in ACTIVE_MODELS:
        print(f"\n--- Optimizing {name} ---")
        base_path = f"models/weights/{name.lower()}_base.pth"
        if not os.path.exists(base_path):
            print(f"Skipping {name}: Base weights not found at {base_path}")
            continue
        
        temp_model = get_model_instance(name).to(device)
        temp_model.load_state_dict(torch.load(base_path, map_location=device, weights_only=True))
        _, before_accs = evaluate_accuracy(temp_model, oracle, device, num_samples=200)

        fisher_dict = None
        if name == "VAE":
            fisher_dict = compute_fisher_dict(temp_model, loader, device)
        elif name == "RectifiedFlow":
            fisher_dict = compute_fisher_rf(temp_model, loader, device)
        elif name == "Autoregressive":
            fisher_dict = compute_fisher_ar(temp_model, loader, device)
        elif name == "NVP":
            fisher_dict = compute_fisher_dict(temp_model, loader, device)

        def evaluate_params(current_params, fisher_dict_in):
            model = get_model_instance(name).to(device)
            model.load_state_dict(torch.load(base_path, map_location=device, weights_only=True))
            frozen_model = copy.deepcopy(model).eval()
            
            model.train()
            for _ in range(FORGET_EPOCHS[name]):
                for x, _ in loader:
                    model.forget_step(x.size(0), target_class, frozen_model, fisher_dict=fisher_dict_in, **current_params, device=device)
            
            _, accs = evaluate_accuracy(model, oracle, device, num_samples=200)
            target_acc = accs[target_class]
            retained_drop = np.mean([before_accs[c] - accs[c] for c in range(10) if c != target_class])
            score = calculate_amnesia_score(target_acc, retained_drop)
            
            return score, accs, retained_drop

        best_params = {
            "loss_type": PARAMETERS[name]["loss_type"][0],
            "lr": PARAMETERS[name]["lr"][len(PARAMETERS[name]["lr"])//2],
            "gamma": PARAMETERS[name]["gamma"][len(PARAMETERS[name]["gamma"])//2], 
            "lmbda": PARAMETERS[name]["lmbda"][len(PARAMETERS[name]["lmbda"])//2]
        }

        for param_to_tune in ["loss_type", "lr", "gamma", "lmbda"]:
            if param_to_tune not in PARAMETERS[name]: continue
            
            best_val_for_step = None
            max_step_score = -float('inf')
            
            for val in PARAMETERS[name][param_to_tune]:
                test_params = copy.deepcopy(best_params)
                test_params[param_to_tune] = val
                
                score, accs, drop = evaluate_params(test_params, fisher_dict)
                print(f"{name}={val} >> Score: {score:.3f} (Target Acc: {accs[target_class]:.2f}, Drop: {drop:.3f})")
                
                if score > max_step_score:
                    max_step_score = score
                    best_val_for_step = val
                
                res = {
                    "Model": name, 
                    "score": score, 
                    "Avg Retained Drop": drop,
                    **test_params
                }
                for i in range(10): 
                    res[f"digit_{i}_before"] = before_accs[i]
                    res[f"digit_{i}_after"] = accs[i]
                all_results.append(res)

            best_params[param_to_tune] = best_val_for_step

        pd.DataFrame(all_results).to_csv(output_file, index=False)

        if name in optimized_registry:
            optimized_registry[name]["forgetting_config"] = best_params
        with open("models/weights/optimized_model_registry.json", "w") as f:
            json.dump(optimized_registry, f, indent=4)
        print(f"Finished {name}.")
    optimized_registry = copy.deepcopy(ARCHITECTURE_REGISTRY)

    

def run_all_target_classes():
    for c in range(0,9):
        run_optimization(target_class=c)

if __name__ == "__main__":
    run_optimization()