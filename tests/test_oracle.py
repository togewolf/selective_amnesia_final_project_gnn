import unittest
import torch
import os
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Add project root to sys.path so 'import scoring' works from anywhere
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scoring import get_oracle


class TestOracleAccuracy(unittest.TestCase):
    """
    Test suite for validating the pre-trained MNIST Oracle classifier's performance.
    """

    def setUp(self):
        # Shift the current working directory to the project root for the duration of the test.
        # This ensures that hardcoded relative paths in scoring.py (like "models/saved_weights/oracle.pth")
        # resolve correctly regardless of how the test is executed.
        self.original_cwd = os.getcwd()
        os.chdir(PROJECT_ROOT)

    def tearDown(self):
        # Restore the original working directory after the test finishes
        os.chdir(self.original_cwd)

    def test_real_oracle_accuracy(self):
        """
        Tests the classification accuracy of the actual saved Oracle on the real MNIST test set.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        oracle_path = "models/saved_weights/oracle.pth"

        # 1. Ensure the oracle weights actually exist before proceeding
        self.assertTrue(
            os.path.exists(oracle_path),
            f"Oracle weights not found at '{os.path.abspath(oracle_path)}'. Please run evaluation.py first."
        )

        # 2. Prepare the standard MNIST transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # 3. Load the full MNIST TEST dataset (10,000 unseen images)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        # 4. Load the Oracle (passing None/0 guarantees no fallback training occurs)
        oracle = get_oracle(device, dataloader=None, epochs=0)
        oracle.eval()

        # 5. Evaluate the Oracle
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = oracle(x)
                predictions = outputs.argmax(dim=1)

                correct += (predictions == y).sum().item()
                total += y.size(0)

        accuracy = correct / total

        print(f"\n[Test] Oracle Accuracy on Full Real MNIST Test Set: {accuracy * 100:.2f}%")

        # 6. Assertions
        # A properly trained CNN on MNIST should easily exceed 95% accuracy.
        self.assertGreater(
            accuracy,
            0.95,
            f"Oracle accuracy is too low ({accuracy * 100:.2f}%). The saved weights might be corrupted or undertrained."
        )


if __name__ == '__main__':
    unittest.main()