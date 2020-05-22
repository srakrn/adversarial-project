# %%
import logging
import os
import sys

import torch
from torch import nn, optim

from clustre.adversarial_training import cluster_training
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
DEVICE = "cuda:0"
LOG_FILENAME = os.path.abspath(__file__)[:-3] + "_log.txt"
SCRIPT_PATH = os.path.dirname(__file__)
FORMAT = "%(message)s"
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, format=FORMAT)
log = logging.getLogger()

# %%
mnist_fcnn.load_state_dict(mnist_fcnn_state)
mnist_cnn.load_state_dict(mnist_cnn_state)
mnist_resnet.load_state_dict(mnist_resnet_state)
cifar10_cnn.load_state_dict(cifar10_cnn_state)
cifar10_resnet.load_state_dict(cifar10_resnet_state)
cifar10_wideresnet.load_state_dict(cifar10_wideresnet_state)

models = {
    "MNIST ResNet": [mnist_resnet, mnist_trainloader, mnist_testloader],
}

params = [
    {"n_clusters": 10},
    {"n_clusters": 50},
    {"n_clusters": 100},
    {"n_clusters": 500},
    {"n_clusters": 1000},
    {"n_clusters": 5000},
    {"n_clusters": 10},
    {"n_clusters": 50},
    {"n_clusters": 100},
    {"n_clusters": 500},
    {"n_clusters": 1000},
    {"n_clusters": 5000},
    {"n_clusters": 10},
    {"n_clusters": 50},
    {"n_clusters": 100},
    {"n_clusters": 500},
    {"n_clusters": 1000},
    {"n_clusters": 5000},
]

new_models = {}

# %%
for model_name, (model, trainloader, testloader) in models.items():
    for p in params:
        logging.info(f"Training {model_name}")
        logging.info(f"Params {p}")
        new_model = cluster_training(
            model, trainloader, device=DEVICE, log=log, **p
        )
        torch.save(
            model.state_dict(),
            os.path.join(SCRIPT_PATH, f"Cluster{model_name}.model"),
        )

        logging.info(f"FGSM attacked {model_name}")
        logging.info(
            classification_report_fgsm(new_model, testloader, device=DEVICE)
        )

        logging.info(f"PGD attacked {model_name}")
        logging.info(
            classification_report_pgd(new_model, testloader, device=DEVICE)
        )
