"""
Use: Running SA with various parameters.
"""
import os, sys
import copy
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
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
# "VAE", "GAN", "RectifiedFlow", "Autoregressive", "NVP"

# FORGET_EPOCHS = {
#     "VAE": 2,
#     "GAN": 5,
#     "RectifiedFlow": 5,
#     "Autoregressive": 5,
#     "NVP": 5
# }
FORGET_EPOCHS = {
    "VAE": 5,
    "GAN": 40,
    "RectifiedFlow": 15,
    "Autoregressive": 20,
    "NVP": 10
}

PARAMETERS = {
    "GAN": {
        "loss_type": ["l1", "smooth_l1"], 
        "lr": [1e-5, 5e-5, 1e-4],
        "gamma": [0.0001,0.001,0.01, 0.1],
        "lmbda": ["-"] # placeholder unused.
    },
    "VAE": {
        "loss_type": ["mse", "bce"],   
        "lr": [1e-4, 5e-4, 1e-3], 
        "gamma": [0.0001,0.001,0.01, 0.1], 
        "lmbda": [0.0001,0.001,0.01, 0.1]
    },
    "RectifiedFlow": {
        "loss_type": ["mse", "l1"],
        "lr": [5e-5, 1e-4, 5e-4],
        "gamma": [0.0001,0.001,0.01, 0.1],
        "lmbda": [0.0001,0.001,0.01, 0.1]
    },
    "Autoregressive": {
        "loss_type": ["bce", "mse"],
        "lr": [5e-5, 1e-4, 5e-4],
        "gamma": [0.0001,0.001,0.01, 0.1],
        "lmbda": [0.0001,0.001,0.01, 0.1]
    },
    "NVP": {
        "loss_type": ["nll", "mse"],
        "lr": [1e-4, 5e-4, 1e-3],
        "gamma": [0.0001,0.001,0.01, 0.1],
        "lmbda": [0.0001,0.001,0.01, 0.1]
    }
}

REGISTRY_PATH = "models/weights/model_registry.json"
with open(REGISTRY_PATH, 'r') as f:
    ARCHITECTURE_REGISTRY = json.load(f)

def get_model_instance(name):
    config = ARCHITECTURE_REGISTRY.get(name)
    if not config:
        logging.debug(f"No conf {name}")
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

def deep_update(base, saved):
    for key, value in saved.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base

def run_optimization(target_class=TARGET_CLASS, active_models=ACTIVE_MODELS, custom_params=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oracle = get_oracle(device)
    loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()), 
                        batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    new_best = []
    
    os.makedirs("evaluation_data", exist_ok=True)
    normal_csv = f"evaluation_data/results_target_{target_class}.csv"
    final_csv = f"evaluation_data/final_results_target_{target_class}.csv"
    
    if os.path.exists(normal_csv):
        normal_df = pd.read_csv(normal_csv)
    else:
        normal_df = pd.DataFrame()

    if os.path.exists(final_csv):
        final_df = pd.read_csv(final_csv)
    else:
        final_df = pd.DataFrame()
    
    registry_path = f"models/weights/optimized_model_registry_{target_class}.json"
    optimized_registry = copy.deepcopy(ARCHITECTURE_REGISTRY)

    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            saved_data = json.load(f)
        optimized_registry = deep_update(optimized_registry, saved_data)

    MAX_BATCHES_PER_EPOCH = 20

    for name in active_models:
        logging.debug(f"\n--- Optimizing {name} ---")
        
        if custom_params is not None and name not in custom_params:
            logging.debug(f"Skipping {name}: Not found in custom_params dictionary.")
            continue
            
        base_path = f"models/weights/{name.lower()}_base.pth"
        if not os.path.exists(base_path):
            logging.debug(f"Skipping {name}: Base weights not found at {base_path}")
            continue
        
        temp_model = get_model_instance(name).to(device)
        temp_model.load_state_dict(torch.load(base_path, map_location=device, weights_only=True))
        _, before_accs = evaluate_accuracy(temp_model, oracle, device, num_samples=200)

        fisher_dict = None
        if name == "VAE": fisher_dict = compute_fisher_dict(temp_model, loader, device)
        elif name == "RectifiedFlow": fisher_dict = compute_fisher_rf(temp_model, loader, device)
        elif name == "Autoregressive": fisher_dict = compute_fisher_ar(temp_model, loader, device)
        elif name == "NVP": fisher_dict = compute_fisher_dict(temp_model, loader, device)

        def evaluate_params(current_params, fisher_dict_in, param_name, param_val):
            model = get_model_instance(name).to(device)
            model.load_state_dict(torch.load(base_path, map_location=device, weights_only=True))
            frozen_model = copy.deepcopy(model).eval()
            
            model.train()
            desc_str = f"Testing {param_name}={param_val}" if param_name else "Testing Targeted Params"
            for _ in tqdm(range(FORGET_EPOCHS[name]), desc=desc_str, leave=False):
                for i, (x, _) in enumerate(loader):
                    if i >= MAX_BATCHES_PER_EPOCH: break 
                    model.forget_step(x.size(0), target_class, frozen_model, fisher_dict=fisher_dict_in, **current_params, device=device)
            
            _, accs = evaluate_accuracy(model, oracle, device, num_samples=200)
            target_acc = accs[target_class]
            retained_drop = np.mean([before_accs[c] - accs[c] for c in range(10) if c != target_class])
            score = calculate_amnesia_score(target_acc, retained_drop)
            
            return score, accs, retained_drop

        if custom_params is not None:
            test_params = custom_params[name]
            logging.debug(f"Evaluating targeted parameters: {test_params}")
            
            already_in_normal = False
            if not normal_df.empty and name in normal_df['Model'].values:
                model_subset = normal_df[normal_df['Model'] == name]
                
                for _, row in model_subset.iterrows():
                    match = True
                    for k, v in test_params.items():
                        try:
                            if float(row[k]) != float(v): match = False
                        except ValueError:
                            if str(row[k]) != str(v): match = False
                            
                    if match:
                        already_in_normal = True
                        logging.debug(f"  -> Found exact parameter match in {normal_csv}. Skipping training.")
                        score = row['score']
                        target_acc = row[f'digit_{target_class}_after']
                        drop = row['Avg Retained Drop']
                        accs = {i: row[f'digit_{i}_after'] for i in range(10)}
                        break

            if not already_in_normal:
                score, accs, drop = evaluate_params(test_params, fisher_dict, None, None)
                target_acc = accs[target_class]
                logging.debug(f"  -> Result Score: {score:.3f} (Target Acc: {target_acc:.2f}, Drop: {drop:.3f})")
                
                res = {"Model": name, "score": score, "Avg Retained Drop": drop, **test_params}
                for i in range(10): 
                    res[f"digit_{i}_before"] = before_accs[i]
                    res[f"digit_{i}_after"] = accs[i]
                
                normal_df = pd.concat([normal_df, pd.DataFrame([res])], ignore_index=True)
                normal_df.to_csv(normal_csv, index=False)
                logging.debug(f"  -> Safely appended new run to {normal_csv}")

            if not final_df.empty and name in final_df['Model'].values:
                final_idx = final_df[final_df['Model'] == name].index[0]
                old_target_acc = final_df.at[final_idx, 'Final_Target_Acc']
                old_drop = final_df.at[final_idx, 'Final_Retained_Drop']
                old_score = calculate_amnesia_score(old_target_acc, old_drop)
                
                if score > old_score:
                    logging.debug(f"  Updating Final Records for {name}...")
                    new_best.append(name)
                    
                    final_df.at[final_idx, 'Final_Target_Acc'] = target_acc
                    final_df.at[final_idx, 'Final_Retained_Drop'] = drop
                    for k, v in test_params.items():
                        final_df.at[final_idx, k] = v
                        
                    for i in range(10):
                        final_df.at[final_idx, f'digit_{i}_before'] = before_accs[i]
                        final_df.at[final_idx, f'digit_{i}_after'] = accs[i]
                        
                    final_df.to_csv(final_csv, index=False)
                    
                    if name in optimized_registry:
                        optimized_registry[name]["forgetting_config"] = test_params
                    with open(f"models/weights/optimized_model_registry_{target_class}.json", "w") as f:
                        json.dump(optimized_registry, f, indent=4)
                else:
                    logging.debug(f"  -> Targeted score ({score:.3f}) did not beat Final score ({old_score:.3f}). Final CSV untouched.")

        else:
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
                    
                    score, accs, drop = evaluate_params(test_params, fisher_dict, param_to_tune, val)
                    logging.debug(f"  {param_to_tune}={val} >> Score: {score:.3f} (Target Acc: {accs[target_class]:.2f}, Drop: {drop:.3f})")
                    
                    if score > max_step_score:
                        max_step_score = score
                        best_val_for_step = val
                    
                    res = {"Model": name, "score": score, "Avg Retained Drop": drop, **test_params}
                    for i in range(10): 
                        res[f"digit_{i}_before"] = before_accs[i]
                        res[f"digit_{i}_after"] = accs[i]
                    
                    normal_df = pd.concat([normal_df, pd.DataFrame([res])], ignore_index=True)

                best_params[param_to_tune] = best_val_for_step

            normal_df.to_csv(normal_csv, index=False)

            if name in optimized_registry:
                optimized_registry[name]["forgetting_config"] = best_params
            with open(f"models/weights/optimized_model_registry_{target_class}.json", "w") as f:
                json.dump(optimized_registry, f, indent=4)
            logging.debug(f"Finished {name}.")
            
    optimized_registry = copy.deepcopy(ARCHITECTURE_REGISTRY)
    return new_best

def run_all_target_classes(active_models, target_classes=range(0,9)):
    logging.info(f"Start parameter test.")
    for c in target_classes:
        logging.info(f"Start class {c}.")
        new_best = run_optimization(target_class=c, active_models=active_models)

def run_specific_params(params, target_classes=range(0,9)):
    logging.info(f"Start targeted parameter test.")

    active_models = [key for key in params]

    for c in target_classes:
        logging.info(f"Start class {c}.")
        new_best = run_optimization(target_class=c, active_models=active_models, custom_params=params)

if __name__ == "__main__":

    params = {
        "GAN": {
        "loss_type": ["l1"], 
        "lr": [5e-4, 1e-3],
        "gamma": [0.001],
        "lmbda": ["-"]
        },
    }
    target_classes = 0

    new_best = run_specific_params(params, target_classes)
    if len(new_best) != 0:
        logging.info(f"Maybe run new best for {new_best}")