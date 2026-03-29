# Selective Amnesia: Targeted Unlearning in Different Conditional Generative Models

**Final Project — Generative Neural Networks for the Sciences (WS 2025/26)**

By **Lela Eigenrauch**, **Mehdy Shinwari**, and **Thomas Wolf**

---

## Overview

This project verifies and extends the [Selective Amnesia (SA)](https://arxiv.org/abs/2305.10120) framework by Heng & Soh beyond its original scope (VAEs and diffusion models) to five structurally diverse conditional generative architectures:

| Architecture | Type | Latent Bottleneck | 
|---|---|---|
| **Conditional VAE** | Latent variable model | Yes (20-dim) |
| **Conditional GAN** | Adversarial model | No | Implicit |
| **Hybrid RealNVP** | Normalizing Flow | Yes (20-dim, via frozen VAE) |
| **Conditional Rectified Flow** | ODE-based flow | No | 
| **Conditional MADE** | Autoregressive model | No |

All models are trained on the MNIST dataset. The SA framework induces selective forgetting of individual digit classes while preserving generation quality for the remaining nine classes.

---

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (recommended; CPU is supported but slow)

### Installation

```bash
git clone https://github.com/togewolf/selective_amnesia_final_project_gnn.git
cd selective_amnesia_final_project_gnn

# Pretrained Models, download manually for faster download speed 
wget "https://nx29079.your-storageshare.de/s/mMHN4yEskmRTyEz/download" -O gnn_weights.zip
unzip gnn_weights.zip -d "GNN weights"
rm gnn_weights.zip

python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

## Quick Demo

### Evaluating Pretrained Models

All 50 pretrained SA models (5 architectures, 10 target digits) are available  [here](https://nx29079.your-storageshare.de/s/mMHN4yEskmRTyEz/download) . Use the demo script to load any model, generate samples and compare Oracle accuracy before and after forgetting:

```bash
# Interactive mode: choose architecture and digit from a menu
python demo_evaluate.py

# The default sample amount is 10 for fast execution, for better accuracy set higher e.g. 200
python demo_evaluate.py --sample 200
```

For each evaluation the script:
1. Loads the base model and the corresponding SA model from `GNN weights/`
2. Generates 10 images per class using both models
3. Classifies them with the Oracle CNN
4. Prints a per-class accuracy table (before vs. after SA)
5. Saves a two-panel figure (sample grid + accuracy bar chart) to `demo_outputs/`

---

## How to Run

### Full Pipeline

```bash
python main.py
```

This executes the complete workflow end-to-end:

1. **Base model training** — trains 3 random-seed variants per architecture
2. **Architecture selection** — evaluates variants, picks the best per model
3. **SA hyperparameter optimization** — grid search over loss type, learning rate, γ (replay weight), and λ (EWC weight) for all 10 target classes
4. **Final SA execution** — runs forgetting with optimal parameters, saves 50 final models
5. **Plot generation** — creates publication figures

**Estimated total compute:** ~35 hours on an NVIDIA RTX 4090.

### Simple Workflow (Single Model, Single Class)

For quick experimentation with a single model and target class:

```bash
# 1. Train a base model
python simple_process/training_single.py

# 2. Apply Selective Amnesia (forgets class 0 by default)
python simple_process/forgetting.py

# 3. Evaluate and visualize
python simple_process/evaluation.py
```

### Extended training and evaluation process

Each stage can be run independently:

```bash
# Trains with different architecture parameters and saves multiple weights
python training.py

# Trains the oracle
python train_oracle.py

# Tests trained architectures and picks best performing weights
python check_architectures.py

# Tests best architecture for all models with different parameters
python test_parameters.py

# Takes the best parameters and run them with slightly higher max epochs, plots example images
python run_with_best.py

# Plots results for paper
python paper_plots.py
```

---

## Model Architectures

### Conditional VAE
Dense encoder/decoder MLP with a 20-dimensional latent bottleneck. Class conditioning via one-hot concatenation at both encoder and decoder inputs. Training loss: ELBO (BCE reconstruction + KL divergence). This is the baseline architecture, functionally equivalent to the one used in the original SA paper.

### Conditional GAN
Generator maps 100-dim noise + class embedding → 784-dim image. Discriminator classifies real/fake conditioned on class label. Trained with BCEWithLogits loss and label smoothing. Lacks a tractable likelihood, so EWC cannot be applied (λ = 0).

### Hybrid RealNVP (Normalizing Flow)
Two-stage architecture: (1) an unconditional VAE compresses MNIST into a 20-dim latent space, (2) a conditional RealNVP with affine coupling layers models the class-conditional prior over that latent space. SA is applied only to the flow component; the VAE decoder remains frozen. This separation enables "latent space amnesia". The model forgets where to locate a class in latent space but retains the ability to render valid digits.

### Conditional Rectified Flow
Learns a velocity field v(x_t, t, c) that transports Gaussian noise to data along straight-line ODE trajectories. Implemented as a residual MLP (8 blocks, hidden dim 2048) with sinusoidal time embeddings and one-hot class conditioning. 

### Conditional MADE (Autoregressive)
Masked Autoencoder for Distribution Estimation with 5 hidden layers (dim 1024). Enforces autoregressive ordering via binary weight masks. Class conditioning via unmasked one-hot columns at every layer. 

---

## Evaluation

### Oracle Classifier

An independently trained CNN classifier is used to evaluate the success of forgetting the target clase and general model performance (retained drop). 

### Metrics

| Metric | Definition | Ideal Value |
|---|---|---|
| **Target Accuracy** | Oracle accuracy on generated target-class images after SA | 0.0 (complete forgetting) |
| **Retained Drop** | Average accuracy drop on non-target classes after SA | 0.0 (no catastrophic forgetting) |
| **Amnesia Score** | `(1 - target_acc) - 2 × retained_drop` | Higher is better |

