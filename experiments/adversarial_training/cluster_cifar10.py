# %%
import logging
import os
import sys

import torch
from clustre.adversarial_training import cluster_training
from clustre.helpers.datasets import cifar10_testloader, cifar10_trainloader
from clustre.helpers.metrics import (classification_report,
                                     classification_report_fgsm,
                                     classification_report_pgd)
from clustre.models import cifar10_cnn, cifar10_resnet
from clustre.models.state_dicts import cifar10_cnn_state, cifar10_resnet_state
from torch import nn, optim

# %%
DEVICE = "cuda:0"
LOG_FILENAME = os.path.abspath(__file__)[:-3] + "_log.txt"
SCRIPT_PATH = os.path.dirname(__file__)
FORMAT = "%(message)s"
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, format=FORMAT)
log = logging.getLogger()

# %%
cifar10_cnn.load_state_dict(cifar10_cnn_state)
cifar10_resnet.load_state_dict(cifar10_resnet_state)

models = {
    "CIFAR-10 CNN": [
        cifar10_cnn,
        cifar10_cnn_state,
        cifar10_trainloader,
        cifar10_testloader,
    ],
    "CIFAR-10 ResNet": [
        cifar10_resnet,
        cifar10_resnet_state,
        cifar10_trainloader,
        cifar10_testloader,
    ],
}

params = [
    {
        "n_clusters": 2000,
        "cluster_with": "fgsm_input",
        "method": "kmcuda",
        "n_init": 3,
    },
    {
        "n_clusters": 5000,
        "cluster_with": "fgsm_input",
        "method": "kmcuda",
        "n_init": 3,
    },
    {
        "n_clusters": 2000,
        "cluster_with": "fgsm_perturb",
        "method": "kmcuda",
        "n_init": 3,
    },
    {
        "n_clusters": 5000,
        "cluster_with": "fgsm_perturb",
        "method": "kmcuda",
        "n_init": 3,
    },
    {
        "n_clusters": 2000,
        "cluster_with": "original_data",
        "method": "kmcuda",
        "n_init": 3,
    },
    {
        "n_clusters": 5000,
        "cluster_with": "original_data",
        "method": "kmcuda",
        "n_init": 3,
    },
]

global_param = {"n_epoches": 40}

new_models = {}

# %%
for model_name, (model, state, trainloader, testloader) in models.items():
    for p in params:
        model.load_state_dict(state)
        p = {**p, **global_param}
        logging.info(f"Training {model_name}")
        logging.info(f"Params {p}")
        new_model = cluster_training(
            model, trainloader, device=DEVICE, log=log, **p
        )
        torch.save(
            model.state_dict(),
            os.path.join(SCRIPT_PATH, f"Cluster{model_name}.model"),
        )

        logging.info(f"Unattacked {model_name}")
        logging.info(
            classification_report(new_model, testloader, device=DEVICE)
        )

        logging.info(f"FGSM attacked {model_name}")
        logging.info(
            classification_report_fgsm(new_model, testloader, device=DEVICE)
        )

        logging.info(f"PGD attacked {model_name}")
        logging.info(
            classification_report_pgd(new_model, testloader, device=DEVICE)
        )
