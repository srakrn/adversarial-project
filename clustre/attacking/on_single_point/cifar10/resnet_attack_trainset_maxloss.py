# %%
import logging
import os
import sys

import torch
from torch import nn, optim

from clustre.attacking.on_single_point.attack import maxloss  # isort:skip
from clustre.helpers.cifar10_helpers import (
    cifar10_resnet_model,
    cifar10_testset,
)  # isort:skip

logging.basicConfig(
    filename=f"logs/{os.path.basename(__file__)}.log",
    filemode="a",
    level="INFO",
    format="%(process)d-%(levelname)s-%(asctime)s-%(message)s",
)

# %%
OUTPUT_PATH = "perturbs/on_single_point/cifar10/resnet_attack_fgsm_testset.pt"

# %%
criterion = nn.CrossEntropyLoss()
logging.info("Started running")
perturbs = maxloss(cifar10_resnet_model, criterion, cifar10_testset, verbose=True)
logging.info("Ended running")
#  %%
torch.save(perturbs, OUTPUT_PATH)
