# %%
import logging
import os
import sys

import torch
from clustre.adversarial_training import pgd_training
from clustre.helpers.datasets import (cifar10_testloader, cifar10_trainloader,
                                      mnist_testloader, mnist_trainloader)
from clustre.helpers.metrics import (classification_report,
                                     classification_report_fgsm,
                                     classification_report_pgd)
from clustre.models import (cifar10_cnn, cifar10_resnet, cifar10_wideresnet,
                            mnist_cnn, mnist_fcnn, mnist_resnet)
from clustre.models.state_dicts import (cifar10_cnn_state,
                                        cifar10_resnet_state, mnist_cnn_state,
                                        mnist_resnet_state)
from torch import nn, optim

# %%
LOG_FILENAME = os.path.abspath(__file__)[:-3] + "_log.txt"
SCRIPT_PATH = os.path.dirname(__file__)
FORMAT = "%(message)s"
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, format=FORMAT)
log = logging.getLogger()

# %%
mnist_cnn.load_state_dict(mnist_cnn_state)
mnist_resnet.load_state_dict(mnist_resnet_state)
cifar10_cnn.load_state_dict(cifar10_cnn_state)
cifar10_resnet.load_state_dict(cifar10_resnet_state)

models = {
    "CIFAR-10 ResNet": [
        cifar10_resnet,
        cifar10_trainloader,
        cifar10_testloader,
    ],
}

# %%
for model_name, (model, trainloader, testloader) in models.items():
    logging.info(f"Training {model_name}")
    new_model = pgd_training(
        model, trainloader, n_epoches=40, device="cuda", log=log
    )
    torch.save(
        model.state_dict(),
        os.path.join(SCRIPT_PATH, f"PGD {model_name}.model"),
    )

    logging.info(f"Unattacked {model_name}")
    logging.info(classification_report(model, testloader, device="cuda"))

    logging.info(f"FGSM attacked {model_name}")
    logging.info(classification_report_fgsm(model, testloader, device="cuda"))

    logging.info(f"PGD attacked {model_name}")
    logging.info(classification_report_pgd(model, testloader, device="cuda"))
