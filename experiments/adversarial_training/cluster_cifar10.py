# %%
import logging
import os
import sys

import torch
from clustre.adversarial_training import cluster_training
from clustre.helpers.datasets import cifar10_testloader, cifar10_trainloader
from clustre.helpers.metrics import (
    classification_report,
    classification_report_fgsm,
    classification_report_pgd,
)
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

global_param = {"n_init": 3, "n_epoches": 40}

new_models = {}

# %%
for model_name, (model, state, trainloader, testloader) in models.items():
    for n_clusters in [300, 500, 1000, 2000, 3000, 5000, 7000, 10000, 20000]:
        for cluster_with in ["original_data", "fgsm_perturb", "fgsm_input"]:
            model.load_state_dict(state)
            logging.info(f"Training {model_name}")
            logging.info(
                "n_cluster = {}, cluster_with = {}".format(
                    n_clusters, cluster_with
                )
            )
            new_model = cluster_training(
                model,
                trainloader,
                device=DEVICE,
                log=log,
                n_clusters=n_clusters,
                cluster_with=cluster_with,
                **global_param,
            )
            torch.save(
                model.state_dict(),
                os.path.join(SCRIPT_PATH, f"Cluster {model_name}.model"),
            )

            logging.info(f"Unattacked {model_name}")
            logging.info(
                classification_report(new_model, testloader, device=DEVICE)
            )

            logging.info(f"FGSM attacked {model_name}")
            logging.info(
                classification_report_fgsm(
                    new_model, testloader, device=DEVICE
                )
            )

            logging.info(f"PGD attacked {model_name}")
            logging.info(
                classification_report_pgd(new_model, testloader, device=DEVICE)
            )
