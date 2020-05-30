import argparse

import torch

from clustre.helpers.datasets import cifar10_testloader, mnist_testloader
from clustre.helpers.metrics import (
    classification_report,
    classification_report_fgsm,
    classification_report_pgd,
)
from clustre.models import (
    cifar10_cnn,
    cifar10_wide_resnet34_10,
    mnist_cnn,
    mnist_resnet18,
)


def __main__():
    model_structures = {
        "mnist_cnn": [mnist_cnn, mnist_testloader],
        "mnist": [mnist_resnet18, mnist_testloader],
        "cifar10_cnn": [cifar10_cnn, cifar10_testloader],
        "cifar10_wide_resnet34_10": [
            cifar10_wide_resnet34_10,
            cifar10_testloader,
        ],
    }

    parser = argparse.ArgumentParser(
        description="Evaluate models against attacked perturbations"
    )

    parser.add_argument("arch", type=str)
    parser.add_argument("state_dict", type=str)
    args = parser.parse_args()

    arch = args.arch
    state_path = args.state_dict

    model, testloader = model_structures[arch]
    state = torch.load(state_path)
    model.load_state_dict(state)

    print(f"Unattacked {arch}")
    print(classification_report(model, testloader, device="cuda"))

    print(f"FGSM attacked {arch}")
    print(classification_report_fgsm(model, testloader, device="cuda"))

    print(f"PGD attacked {arch}")
    print(classification_report_pgd(model, testloader, device="cuda"))


if __name__ == "__main__":
    __main__()
