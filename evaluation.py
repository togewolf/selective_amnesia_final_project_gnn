import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fixes "Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.variational_autoencoder.variational_autoencoder import ConditionalVAE
from models.generative_adversarial_network.generative_adversarial_network import ConditionalGAN
from scoring import get_oracle, evaluate_accuracy

# Define the models you want to evaluate
ACTIVE_MODELS = ["VAE"]
models_dict = {
    "VAE": ConditionalVAE(),
    "GAN": ConditionalGAN()
    # todo: add further models here
}
TARGET_CLASS_FORGOTTEN = 0


def plot_class_comparisons(before_accs, after_accs, model_name, target_class):
    """
    Creates and saves a bar chart comparing generation accuracy per class
    before and after the forgetting process.

    Args:
        before_accs (dict): Accuracy per class for the baseline model.
        after_accs (dict): Accuracy per class for the model with amnesia.
        model_name (str): The name of the model (e.g., 'GAN', 'VAE') for the title.
        target_class (int): The class that was forgotten, to highlight it in the chart.
    """
    classes = list(range(10))
    before = [before_accs[c] for c in classes]
    after = [after_accs[c] for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, before, width, label='Before Forgetting (Base)')
    bar_after = ax.bar(x + width / 2, after, width, label='After Forgetting')

    # Highlight the target forgotten class in red
    bar_after[target_class].set_color('red')

    ax.set_ylabel('Oracle Generation Accuracy')
    ax.set_title(f'{model_name}: Impact of Selective Amnesia (Target: Class {target_class})')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"evaluation_data/{model_name}_forget_{target_class}_chart.png")
    plt.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("evaluation_data", exist_ok=True)

    # Setup Dataset (Needed solely to train the Oracle if it doesn't exist yet)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Retrieve the Oracle classifier
    oracle = get_oracle(device, loader, epochs=2)

    results = []
    # 10 Samples for each class to form a nice visual 10x10 grid (100 total images)
    grid_y = torch.arange(10).repeat_interleave(10).to(device)

    for name in ACTIVE_MODELS:
        if name not in models_dict:
            continue

        print(f"\n--- Evaluating {name} ---")
        # Cleanly instantiate fresh model architectures
        base_model = models_dict[name].to(device)

        if name == "VAE":
            forgotten_model = ConditionalVAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=20, class_size=10).to(device)
        elif name == "GAN":
            forgotten_model = ConditionalGAN(latent_dim=100, num_classes=10).to(device)

        base_path = f"models/saved_weights/{name.lower()}_base.pth"
        forget_path = f"models/saved_weights/{name.lower()}_forgot_{TARGET_CLASS_FORGOTTEN}.pth"

        if not os.path.exists(base_path) or not os.path.exists(forget_path):
            print(f"Missing weights for {name}. Please run training.py and forgetting.py first.")
            continue

        # Load weights
        base_model.load_state_dict(torch.load(base_path, map_location=device))
        forgotten_model.load_state_dict(torch.load(forget_path, map_location=device))

        # 1. Evaluate Accuracy Before and After
        _, before_c_accs = evaluate_accuracy(base_model, oracle, device, num_samples=100)
        _, after_c_accs = evaluate_accuracy(forgotten_model, oracle, device, num_samples=100)

        # 2. Generate Image Grids
        with torch.no_grad():
            base_imgs = base_model.generate(grid_y) * 0.5 + 0.5
            forget_imgs = forgotten_model.generate(grid_y) * 0.5 + 0.5
            save_image(base_imgs, f"evaluation_data/{name}_base_grid.png", nrow=10)
            save_image(forget_imgs, f"evaluation_data/{name}_forgot_{TARGET_CLASS_FORGOTTEN}_grid.png", nrow=10)

        # 3. Create Plots
        plot_class_comparisons(before_c_accs, after_c_accs, name, TARGET_CLASS_FORGOTTEN)

        # 4. Calculate Final Metrics
        retained_classes = [c for c in range(10) if c != TARGET_CLASS_FORGOTTEN]
        catastrophic_forgetting = np.mean([before_c_accs[c] - after_c_accs[c] for c in retained_classes])

        results.append({
            "Model": name,
            "Target Class": TARGET_CLASS_FORGOTTEN,
            "Target Acc (Before)": before_c_accs[TARGET_CLASS_FORGOTTEN],
            "Forgetting Precision (Target Acc After, Lower=Better)": after_c_accs[TARGET_CLASS_FORGOTTEN],
            "Catastrophic Forgetting (Retained Acc Drop, Lower=Better)": catastrophic_forgetting
        })

    if results:
        df = pd.DataFrame(results)
        df.to_csv("evaluation_data/evaluation_results.csv", index=False)
        print("\nEvaluation Results saved to CSV:")
        print(df.to_string())