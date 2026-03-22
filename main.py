import torch

from extended_training import run as training
from check_architectures import pick_best_and_save
from evaluation.test_parameters import run_all_target_classes
from evaluation.run_with_best import run_best

import logging

logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("evaluation_data/SA.log"),
        logging.StreamHandler()
    ]
)

def main():
    # print(torch.__version__)
    # print(torch.version.cuda)
    # print(torch.cuda.is_available())

    # base models
    # training()
    # pick_best_and_save()

    ACTIVE_MODELS = ["VAE", "GAN", "RectifiedFlow", "Autoregressive", "NVP"]

    # SA
    logging.info(f"Start")
    run_all_target_classes(ACTIVE_MODELS)
    run_best(ACTIVE_MODELS)
    logging.info(f"Stop.")


if __name__ == "__main__":
    main()