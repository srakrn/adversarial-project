# %%
import logging
import os
import sys

import torch
from torch import nn, optim

from clustre.helpers.datasets import (
    cifar10_testloader,
    cifar10_trainloader,
    mnist_testloader,
    mnist_trainloader,
)
from clustre.helpers.metrics import classification_report, classification_report_pgd
from clustre.models import (
    cifar10_cnn,
    cifar10_resnet,
    cifar10_wideresnet,
    mnist_cnn,
    mnist_fcnn,
    mnist_resnet,
)
from clustre.models.state_dicts import (
    cifar10_cnn_state,
    cifar10_resnet_state,
    cifar10_wideresnet_state,
    mnist_cnn_state,
    mnist_fcnn_state,
    mnist_resnet_state,
)

# %%
LOG_FILENAME = os.path.abspath(__file__)[:-3] + "_log.txt"
FORMAT = "%(message)s"
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, format=FORMAT)

# %%
mnist_fcnn.load_state_dict(mnist_fcnn_state)
mnist_cnn.load_state_dict(mnist_cnn_state)
mnist_resnet.load_state_dict(mnist_resnet_state)
cifar10_cnn.load_state_dict(cifar10_cnn_state)
cifar10_resnet.load_state_dict(cifar10_resnet_state)
cifar10_wideresnet.load_state_dict(cifar10_wideresnet_state)

models = {
    "MNIST FCNN": [mnist_fcnn, mnist_trainloader, mnist_testloader],
    "MNIST CNN": [mnist_cnn, mnist_trainloader, mnist_testloader],
    "MNIST ResNet": [mnist_resnet, mnist_trainloader, mnist_testloader],
    "CIFAR-10 CNN": [cifar10_cnn, cifar10_trainloader, cifar10_testloader],
    "CIFAR-10 ResNet": [cifar10_resnet, cifar10_trainloader, cifar10_testloader],
    "CIFAR-10 Wide ResNet": [
        cifar10_wideresnet,
        cifar10_trainloader,
        cifar10_testloader,
    ],
}

# %%
for model_name, (model, _, testloader) in models.items():
    logging.info(f"Unattacked {model_name}")
    logging.info(classification_report(model, testloader, device="cuda"))

# %%
for model_name, (model, _, testloader) in models.items():
    logging.info(f"PGD attacked {model_name}")
    logging.info(classification_report_pgd(model, testloader, device="cuda"))
