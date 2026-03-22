import torch

from extended_training import run as training
from check_architectures import pick_best_and_save
from evaluation.test_parameters import run_all_target_classes
from evaluation.run_with_best import run_best
from evaluation.paper_plots import plot_all

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
    run_all_target_classes(["GAN"],target_classes=range(2)) # had to repeat because accidentally mapped 0-> 8 (too similar)
    run_all_target_classes(ACTIVE_MODELS,target_classes=range(2,10)) # continue after fix with remaining
    run_best(ACTIVE_MODELS, target_classes=range(10)) # get final SA models and sample imgs
    plot_all(target_classes=range(10)) # plot for paper
    logging.info(f"Finished.")


if __name__ == "__main__":
    main()