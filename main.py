import torch

from extended_training import run as training
from check_architectures import pick_best_and_save
from evaluation.test_parameters import run_all_target_classes

def main():
    # print(torch.__version__)
    # print(torch.version.cuda)
    # print(torch.cuda.is_available())

    # training()
    # pick_best_and_save()
    run_all_target_classes()


if __name__ == "__main__":
    main()