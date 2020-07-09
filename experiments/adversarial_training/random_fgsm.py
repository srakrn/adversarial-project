# %%
import logging
import os

import torch

from clustre.adversarial_training import fgsm_training
from clustre.helpers.datasets import (
    cifar10_testloader,
    cifar10_trainloader,
    mnist_testloader,
    mnist_trainloader,
)
from clustre.helpers.metrics import (
    classification_report,
    classification_report_fgsm,
    classification_report_pgd,
)
from clustre.models import mnist_resnet18
from clustre.models.state_dicts import mnist_resnet18_state

# %%
LOG_FILENAME = os.path.abspath(__file__)[:-3] + "_log.txt"
SCRIPT_PATH = os.path.dirname(__file__)
FORMAT = "%(message)s"
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, format=FORMAT)
log = logging.getLogger()

# %%
mnist_resnet18.load_state_dict(mnist_resnet18_state)

models = {
    "MNIST ResNet18": [mnist_resnet18, mnist_trainloader, mnist_testloader],
}


# %%
for model_name, (model, trainloader, testloader) in models.items():
    logging.info(f"Training {model_name}")
    new_model = fgsm_training(
        model, trainloader, device="cuda", log=log, n_epoches=40, random=True
    )
    torch.save(
        model.state_dict(),
        os.path.join(SCRIPT_PATH, f"FGSM {model_name}.model"),
    )

    logging.info(f"Unattacked {model_name}")
    logging.info(classification_report(model, testloader, device="cuda"))

    logging.info(f"FGSM attacked {model_name}")
    logging.info(classification_report_fgsm(model, testloader, device="cuda"))

    logging.info(f"PGD attacked {model_name}")
    logging.info(classification_report_pgd(model, testloader, device="cuda"))


# %%
