# %%
import logging
import os

import torch
from clustre.adversarial_training import free_training
from clustre.helpers.datasets import (
    cifar10_testloader,
    cifar10_trainloader,
    cifar10_trainset,
    mnist_testloader,
    mnist_trainloader,
    mnist_trainset,
)
from clustre.helpers.metrics import (
    classification_report,
    classification_report_fgsm,
    classification_report_pgd,
)
from clustre.models import cifar10_wide_resnet34_10
from clustre.models.state_dicts import cifar10_wide_resnet34_10_state
from torch.utils.data import DataLoader

# %%
LOG_FILENAME = os.path.abspath(__file__)[:-3] + "_log.txt"
SCRIPT_PATH = os.path.dirname(__file__)
FORMAT = "%(message)s"
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, format=FORMAT)
log = logging.getLogger()

# %%
cifar10_trainloader_droplast = DataLoader(
    cifar10_trainset, batch_size=64, shuffle=True, drop_last=True
)

# %%
cifar10_wide_resnet34_10.load_state_dict(cifar10_wide_resnet34_10_state)

models = {
    "CIFAR-10 Wide ResNet-34 10": [
        cifar10_wide_resnet34_10,
        cifar10_trainloader_droplast,
        cifar10_testloader,
    ],
}

# %%
for model_name, (model, trainloader, testloader) in models.items():
    logging.info(f"Training {model_name}")
    new_model = free_training(
        model, trainloader, n_epoches=40, device="cuda", log=log
    )
    torch.save(
        model.state_dict(),
        os.path.join(SCRIPT_PATH, f"Free {model_name}.model"),
    )

    logging.info(f"Unattacked {model_name}")
    logging.info(classification_report(model, testloader, device="cuda"))

    logging.info(f"FGSM attacked {model_name}")
    logging.info(classification_report_fgsm(model, testloader, device="cuda"))

    logging.info(f"PGD attacked {model_name}")
    logging.info(classification_report_pgd(model, testloader, device="cuda"))
