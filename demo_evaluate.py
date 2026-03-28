"""
Demo: Evaluate Pretrained Selective Amnesia Models
===================================================
Choose an architecture and a forgotten digit, then visualize:
  1. A grid of generated samples from the SA model (all 10 classes)
  2. Oracle classification accuracy per class (before vs. after SA)

Usage:
    python demo_evaluate.py                                    # interactive menu
    python demo_evaluate.py --model VAE --target 3             # direct CLI
    python demo_evaluate.py --model VAE --target 3 --samples 5 # fewer samples
    python demo_evaluate.py --all                              # all 50 combinations
    python demo_evaluate.py --all --samples 20                 # all, 20 per class

Requirements:
    pip install torch torchvision matplotlib
"""

import os
import sys
import json
import argparse

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# ── path setup ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.variational_autoencoder.variational_autoencoder import ConditionalVAE
from models.generative_adversarial_network.generative_adversarial_network import ConditionalGAN
from models.normalizing_flows.normalizing_flows import ConditionalRealNVP
from models.rectified_flows.rectified_flows import ConditionalRectifiedFlow
from models.autoregressive.autoregressive_model import ConditionalMADE
from scoring import MNISTOracle

# ── constants ───────────────────────────────────────────────
WEIGHTS_DIR = os.path.join(ROOT, "GNN weights")
REGISTRY_PATH = os.path.join(WEIGHTS_DIR, "model_registry.json")
ORACLE_PATH = os.path.join(WEIGHTS_DIR, "oracle.pth")
SA_DIR = os.path.join(WEIGHTS_DIR, "after_SA")
OUTPUT_DIR = os.path.join(ROOT, "demo_outputs")

MODEL_NAMES = ["VAE", "GAN", "RectifiedFlow", "Autoregressive", "NVP"]

# keys in model_registry.json that are NOT constructor args
NON_INIT_KEYS = {"lr", "lr_G", "lr_D"}

FILENAME_MAP = {
    "VAE": "vae",
    "GAN": "gan",
    "RectifiedFlow": "rectifiedflow",
    "Autoregressive": "autoregressive",
    "NVP": "nvp",
}


# ── helpers ─────────────────────────────────────────────────
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_registry():
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def build_model(name, config):
    """Instantiate a model from its registry config (strips non-init keys)."""
    init_cfg = {k: v for k, v in config.items() if k not in NON_INIT_KEYS}
    constructors = {
        "VAE": ConditionalVAE,
        "GAN": ConditionalGAN,
        "NVP": ConditionalRealNVP,
        "RectifiedFlow": ConditionalRectifiedFlow,
        "Autoregressive": ConditionalMADE,
    }
    return constructors[name](**init_cfg)


def load_weights(model, path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_oracle(device):
    oracle = MNISTOracle().to(device)
    oracle.load_state_dict(torch.load(ORACLE_PATH, map_location=device, weights_only=True))
    oracle.eval()
    return oracle


def generate_grid(model, device, num_per_class=10):
    """Generate num_per_class images for each digit 0-9, return (10*num_per_class, 1, 28, 28)."""
    all_imgs = []
    with torch.no_grad():
        for c in range(10):
            y = torch.full((num_per_class,), c, dtype=torch.long, device=device)
            imgs = model.generate(y)  # (N, 1, 28, 28) in [-1, 1]
            all_imgs.append(imgs)
    return torch.cat(all_imgs, dim=0)


def evaluate_per_class(model, oracle, device, num_samples=10):
    """Return dict {class_int: accuracy} using the Oracle."""
    model.eval()
    oracle.eval()
    accs = {}
    with torch.no_grad():
        for c in range(10):
            y = torch.full((num_samples,), c, dtype=torch.long, device=device)
            imgs = model.generate(y)
            preds = oracle(imgs).argmax(dim=1)
            accs[c] = (preds == y).sum().item() / num_samples
    return accs


# ── visualisation ───────────────────────────────────────────
def _draw_sample_grid(ax, imgs, title, target_class):
    """Draw a 10×10 sample grid (10 classes × 10 samples) on the given axis."""
    grid = vutils.make_grid(imgs * 0.5 + 0.5, nrow=10, padding=2, normalize=False)
    ax.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    # each cell is 28px + 2px padding on each side = 30px stride; center at 14px offset
    cell_stride = 28 + 2
    ax.set_yticks([14 + i * cell_stride for i in range(10)])
    yticklabels = []
    for i in range(10):
        label = f"{i} ◄" if i == target_class else str(i)
        yticklabels.append(label)
    ax.set_yticklabels(yticklabels, fontsize=9)
    ax.set_ylabel("digit class")


def plot_results(model_name, target_class, base_imgs, sa_imgs, base_accs, sa_accs, save_path):
    """Create a three-panel figure: base grid, SA grid, accuracy bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7),
                              gridspec_kw={"width_ratios": [1, 1, 1]})

    # — left: base model sample grid —
    _draw_sample_grid(axes[0], base_imgs,
                      f"Base Model (before SA)", target_class)

    # — center: SA model sample grid —
    _draw_sample_grid(axes[1], sa_imgs,
                      f"After Forgetting Digit {target_class}", target_class)

    # — right: accuracy comparison —
    x = list(range(10))
    w = 0.35
    bars_base = [base_accs.get(c, 0) for c in x]
    bars_sa = [sa_accs.get(c, 0) for c in x]

    axes[2].bar([c - w / 2 for c in x], bars_base, w, label="Base model", color="#4CAF50", alpha=0.85)
    axes[2].bar([c + w / 2 for c in x], bars_sa, w, label="After SA", color="#F44336", alpha=0.85)
    axes[2].axvline(target_class, color="blue", ls="--", lw=1.2, label=f"Target (digit {target_class})")
    axes[2].set_xlabel("Digit class")
    axes[2].set_ylabel("Oracle accuracy")
    axes[2].set_title("Per-class Oracle Accuracy", fontsize=12)
    axes[2].set_xticks(x)
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(fontsize=9)

    plt.suptitle(f"Selective Amnesia Evaluation — {model_name} forgetting digit {target_class}",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ── core routine ────────────────────────────────────────────
def run_demo(model_name, target_class, device, registry, oracle, num_samples=10, quiet=False):
    config = registry[model_name]
    tag = FILENAME_MAP[model_name]

    # paths
    base_path = os.path.join(WEIGHTS_DIR, f"{tag}_base.pth")
    sa_path = os.path.join(SA_DIR, f"{tag}_forgot_{target_class}.pth")

    for p, label in [(base_path, "base weights"), (sa_path, "SA weights")]:
        if not os.path.exists(p):
            print(f"  ✗ Missing {label}: {p}")
            return

    if not quiet:
        print(f"\n{'='*55}")
        print(f"  {model_name}  —  target digit {target_class}  ({num_samples} samples/class)")
        print(f"{'='*55}")

    # load base model, generate grid & evaluate
    base_model = build_model(model_name, config)
    load_weights(base_model, base_path, device)
    if not quiet:
        print("  Evaluating base model …")
    base_accs = evaluate_per_class(base_model, oracle, device, num_samples=num_samples)
    base_imgs = generate_grid(base_model, device, num_per_class=10)

    # load SA model, generate grid & evaluate
    sa_model = build_model(model_name, config)
    load_weights(sa_model, sa_path, device)
    if not quiet:
        print("  Evaluating SA model …")
    sa_accs = evaluate_per_class(sa_model, oracle, device, num_samples=num_samples)
    sa_imgs = generate_grid(sa_model, device, num_per_class=10)

    # print table
    if not quiet:
        print(f"\n  {'Class':>6}  {'Base':>8}  {'After SA':>8}  {'Drop':>8}")
        print(f"  {'-'*38}")
        for c in range(10):
            marker = " ◄ target" if c == target_class else ""
            drop = sa_accs[c] - base_accs[c]
            print(f"  {c:>6}  {base_accs[c]:>8.3f}  {sa_accs[c]:>8.3f}  {drop:>+8.3f}{marker}")
        avg_base = sum(base_accs.values()) / 10
        avg_sa = sum(sa_accs.values()) / 10
        print(f"  {'-'*38}")
        print(f"  {'Avg':>6}  {avg_base:>8.3f}  {avg_sa:>8.3f}  {avg_sa - avg_base:>+8.3f}")

    # save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"{tag}_forgot_{target_class}.png")
    plot_results(model_name, target_class, base_imgs, sa_imgs, base_accs, sa_accs, save_path)
    return base_accs, sa_accs


# ── interactive menu ────────────────────────────────────────
def interactive_menu():
    print("\n╔══════════════════════════════════════════════╗")
    print("║   Selective Amnesia — Pretrained Model Demo  ║")
    print("╚══════════════════════════════════════════════╝\n")
    print("Available architectures:")
    for i, name in enumerate(MODEL_NAMES):
        print(f"  [{i}] {name}")
    while True:
        try:
            idx = int(input("\nSelect architecture [0-4]: "))
            if 0 <= idx < len(MODEL_NAMES):
                break
        except (ValueError, EOFError):
            pass
        print("  Invalid choice, try again.")
    model_name = MODEL_NAMES[idx]

    while True:
        try:
            target = int(input(f"Select digit to forget [0-9]: "))
            if 0 <= target <= 9:
                break
        except (ValueError, EOFError):
            pass
        print("  Invalid choice, try again.")
    return model_name, target


# ── main ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained Selective Amnesia models.")
    parser.add_argument("--model", type=str, choices=MODEL_NAMES,
                        help="Architecture name")
    parser.add_argument("--target", type=int, choices=range(10), metavar="[0-9]",
                        help="Target digit that was forgotten")
    parser.add_argument("--samples", type=int, default=10, metavar="N",
                        help="Images per class for Oracle evaluation (default: 10). "
                             "Lower values speed up evaluation on slower systems. "
                             "The visual sample grid always shows 10 per class.")
    parser.add_argument("--all", action="store_true",
                        help="Run evaluation for all 50 model-target combinations")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    registry = load_registry()
    oracle = load_oracle(device)

    if args.all:
        for name in MODEL_NAMES:
            for t in range(10):
                run_demo(name, t, device, registry, oracle,
                         num_samples=args.samples, quiet=True)
        print(f"\nAll results saved to {OUTPUT_DIR}/")
        return

    if args.model and args.target is not None:
        model_name, target = args.model, args.target
    else:
        model_name, target = interactive_menu()

    run_demo(model_name, target, device, registry, oracle, num_samples=args.samples)


if __name__ == "__main__":
    main()
