# Contains functions useful to score images / for evaluation

import os
import torch
import torch.nn as nn
from torchmetrics.image.fid import FrechetInceptionDistance


class MNISTOracle(nn.Module):
    """
    A standard Convolutional Neural Network (CNN) acting as an 'Oracle' classifier.

    The Oracle is trained on the standard, real MNIST training dataset to high accuracy.
    Once trained, its weights are frozen. During evaluation, we feed the images produced
    by our generative models into this Oracle.

    - If we ask the GAN to generate a '3' (y=3), and the Oracle classifies the output as a '3',
      we have high 'Generation Precision'.
    - If we apply Selective Amnesia to make the GAN forget '3', and ask it to generate a '3',
      a successful amnesia process will result in the Oracle classifying the output as
      something else (or noise), leading to a low accuracy for that specific class.
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 5 * 5, 128), nn.ReLU(), nn.Linear(128, 10)
        )
        # Is a sklearn.ensemble.RandomForestClassifier better?

    def forward(self, x):
        return self.model(x)


def get_oracle(device, dataloader=None, epochs=2):
    """
    Loads or trains the Oracle classifier.

    Args:
        device (torch.device): The device (CPU/GPU) to map the model to.
        dataloader (DataLoader, optional): PyTorch DataLoader for the MNIST dataset.
                                           Required if the oracle hasn't been trained yet.
        epochs (int): Number of epochs to train the oracle if no saved weights are found.

    Returns:
        oracle (MNISTOracle): The trained and frozen oracle model.
    """
    os.makedirs("models/weights", exist_ok=True)
    oracle_path = "models/weights/oracle.pth"
    oracle = MNISTOracle().to(device)

    if os.path.exists(oracle_path):
        oracle.load_state_dict(torch.load(oracle_path, map_location=device))
        oracle.eval()
        return oracle

    if dataloader is None or epochs == 0:
        print("Warning: Oracle weights not found and no dataloader provided. Returning untrained Oracle.")
        return oracle

    print("Training Evaluation Oracle...")
    optimizer = torch.optim.Adam(oracle.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    oracle.train()

    for epoch in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(oracle(x), y)
            loss.backward()
            optimizer.step()

    torch.save(oracle.state_dict(), oracle_path)
    oracle.eval()
    return oracle


def calculate_fid(real_images, fake_images):
    """
    Calculates the Frechet Inception Distance (FID) between real and generated images.

    Args:
        real_images (torch.Tensor): A batch of real images from the dataset. Shape: (B, 1, 28, 28).
        fake_images (torch.Tensor): A batch of generated images. Shape: (B, 1, 28, 28).

    Returns:
        float: The computed FID score. Lower is better.
    """

    def preprocess(imgs):
        # Inception needs 3 channels, so we repeat the grayscale channel
        imgs = imgs.repeat(1, 3, 1, 1)
        # Normalize to [0, 255] uint8 for torchmetrics FID
        return ((imgs * 0.5 + 0.5) * 255).clamp(0, 255).to(torch.uint8)

    fid = FrechetInceptionDistance(feature=64)
    fid.update(preprocess(real_images), real=True)
    fid.update(preprocess(fake_images), real=False)
    return fid.compute().item()


def evaluate_accuracy(model, oracle, device, num_samples=100):
    """
    Evaluates how accurately a generative model produces specific classes using the Oracle.

    Args:
        model (nn.Module): The generative model to evaluate.
        oracle (nn.Module): The pre-trained Oracle classifier.
        device (torch.device): Device to perform computations on.
        num_samples (int): Number of images to generate per class for the evaluation.

    Returns:
        avg_acc (float): The overall accuracy across all 10 classes.
        class_accs (dict): A dictionary mapping class integers (0-9) to their specific accuracy.
    """
    model.eval()
    oracle.eval()
    class_accs = {}

    with torch.no_grad():
        for c in range(10):
            # Ask the generator to create `num_samples` images of class `c`
            y_target = torch.full((num_samples,), c, dtype=torch.long, device=device)
            generated_images = model.generate(y_target)

            # Ask the Oracle what class it thinks the generated images are
            predictions = oracle(generated_images).argmax(dim=1)

            # Calculate accuracy: How often did the Oracle agree with our requested class?
            class_accs[c] = (predictions == y_target).sum().item() / num_samples

    avg_acc = sum(class_accs.values()) / 10.0
    return avg_acc, class_accs
