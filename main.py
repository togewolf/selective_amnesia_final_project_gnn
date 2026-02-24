# Doesn't do anything important

import torch
import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    # print(torch.__version__)
    # print(torch.version.cuda)
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()
