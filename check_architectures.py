"""
Use: Verify model quality and save best architecture and weights for parameter optimization.
"""

import os
import glob
import torch
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from models.variational_autoencoder.variational_autoencoder import ConditionalVAE
from models.generative_adversarial_network.generative_adversarial_network import ConditionalGAN
from models.normalizing_flows.normalizing_flows import ConditionalRealNVP
from models.rectified_flows.rectified_flows import ConditionalRectifiedFlow
from models.autoregressive.autoregressive_model import ConditionalMADE

from scoring import get_oracle, evaluate_accuracy

ACTIVE_MODELS = ["VAE", "GAN", "RectifiedFlow", "Autoregressive", "NVP"]
CACHE_DIR = "models/weights/cache"
FINAL_DIR = "models/weights"
EVAL_DIR = "evaluation_data"

def get_model_instance(name, config):
    if name == "VAE": return ConditionalVAE(**config)
    if name == "GAN": return ConditionalGAN(**config)
    if name == "NVP": return ConditionalRealNVP(**config)
    if name == "RectifiedFlow": return ConditionalRectifiedFlow(**config)
    if name == "Autoregressive": return ConditionalMADE(**config)
    return None

def get_grid_example(model, model_name, device):
    model.eval()
    
    num_classes = 10
    samples_per_class = 10
    
    labels = torch.arange(num_classes).repeat_interleave(samples_per_class).to(device)
    
    with torch.no_grad():
        images = model.generate(labels)
        
        images = images.view(-1, 1, 28, 28).cpu()
        images = torch.clamp(images, 0, 1)
        
        grid = vutils.make_grid(images, nrow=samples_per_class, padding=2, normalize=True)
        img_np = grid.permute(1, 2, 0).numpy()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
        plt.axis('off')
        plt.title(f"{model_name} - Quality Verification")
        plt.close()
    return img_np

def plot_example_grids(overview_images, save_path=os.path.join(EVAL_DIR, "base_models_examples.png")):
    fig = plt.figure(figsize=(18, 12))
        
    gs = fig.add_gridspec(2, 6)
    
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])
    
    ax4 = fig.add_subplot(gs[1, 1:3])
    ax5 = fig.add_subplot(gs[1, 3:5])
    
    axes = [ax1, ax2, ax3, ax4, ax5]
    
    for ax, (model_name, img_data) in zip(axes, overview_images.items()):
        ax.imshow(img_data, cmap='gray')
        ax.axis('off')
        ax.set_title(f"{model_name} Output", fontsize=16, fontweight='bold', pad=10)
    
    for i in range(len(overview_images), 5):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def pick_best_and_save():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(FINAL_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    # clean old best weights
    # filelist = [ f for f in os.listdir(FINAL_DIR) if f.endswith(".pth") ]
    # for f in filelist:
    #     if f != "oracle.pth":
    #         os.remove(os.path.join(FINAL_DIR, f))

    if not os.path.exists("models/weights/oracle.pth"):
        raise LookupError("No oracle.pth, run train_oracle.py first.")
    oracle = get_oracle(device)
    registry = {}
    REGISTRY_PATH = os.path.join(FINAL_DIR, "model_registry.json")
    
    overview_images = {}

    for name in ACTIVE_MODELS:
        variant_files = glob.glob(os.path.join(CACHE_DIR, f"{name.lower()}_v*.pth"))
        
        best_score = -1.0
        best_file = None
        best_config = None
        best_accs = None
        
        for file_path in variant_files:
            config_path = file_path.replace(".pth", "_config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            model = get_model_instance(name, config).to(device)
            model.load_state_dict(torch.load(file_path, map_location=device, weights_only=True)) 
            model.eval()
            
            _, class_accs = evaluate_accuracy(model, oracle, device, num_samples=500)
            
            # lowest scoring digit accuracy is the bottleneck, so we want to maximize it
            min_acc = min(class_accs.values())
            # high average acc
            mean_acc = np.mean(list(class_accs.values()))
            
            # metric to balance min and mean acc
            score = mean_acc * min_acc
            
            print(f"  {os.path.basename(file_path)} >> mean acc: {mean_acc:.3f} | min acc: {min_acc:.3f}")
            
            if score > best_score:
                best_score = score
                best_file = file_path
                best_config = config
                best_accs = class_accs
                
        if best_file:
            # if model still bad (< 80% acc), warning
            if min(best_accs.values()) < 0.80:
                print(f"!!! {name} Best run has a class with acc {min(best_accs.values()):.2f} >> increase epochs?")
            
            best_model = get_model_instance(name, best_config).to(device)
            best_model.load_state_dict(torch.load(best_file, map_location=device, weights_only=True))
            overview_images[name] = get_grid_example(best_model, name, device)

            final_path = os.path.join(FINAL_DIR, f"{name.lower()}_base.pth")
            shutil.copy(best_file, final_path)
            
            registry[name] = best_config
        else:
            print(f"no cache {name}.")

    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=4)
    print(f"\nFinished.")

    if overview_images:
        plot_example_grids(overview_images)

if __name__ == "__main__":
    pick_best_and_save()