# %%
import logging
import os
import sys

import torch
from torch import nn, optim

from clustre.attacking.on_single_point.attack import fgsm  # isort:skip
from helpers.cifar10_helpers import cifar10_cnn_model, cifar10_trainset  # isort:skip

logging.basicConfig(
    filename=f"logs/{os.path.basename(__file__)}.log",
    filemode="a",
    level="INFO",
    format="%(process)d-%(levelname)s-%(asctime)s-%(message)s",
)

# %%
OUTPUT_PATH = "perturbs/on_single_point/cifar10/cnn_attack_fgsm_trainset.pt"

# %%
criterion = nn.CrossEntropyLoss()
logging.info("Started running")
perturbs = fgsm(cifar10_cnn_model, criterion, cifar10_trainset, verbose=True)
logging.info("Ended running")
#  %%
torch.save(perturbs, OUTPUT_PATH)
