# %%
import logging
import os
import sys

import torch
from torch import nn, optim

from clustre.helpers.cifar10_helpers import (  # isort:skip
    cifar10_resnet_model,
    testloader,
)

from clustre.attacking.on_single_point.attack import fgsm  # isort:skip

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
perturbs = fgsm(cifar10_resnet_model, criterion, testloader, verbose=True)
logging.info("Ended running")
#  %%
torch.save(perturbs, OUTPUT_PATH)
