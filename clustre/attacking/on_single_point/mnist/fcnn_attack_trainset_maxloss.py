# %%
import logging
import os
import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from clustre.attacking.on_single_point.attack import maxloss
from helpers.mnist_helpers import mnist_fcnn_model, mnist_trainset

logging.basicConfig(
    filename=f"logs/{os.path.basename(__file__)}.log",
    filemode="a",
    level="INFO",
    format="%(process)d-%(levelname)s-%(asctime)s-%(message)s",
)

# %%
OUTPUT_PATH = "perturbs/on_single_point/mnist/fcnn_maxloss_perturbs_trainset.pt"

# %%
criterion = nn.CrossEntropyLoss()
logging.info("Started running")
perturbs = maxloss(mnist_fcnn_model, criterion, mnist_trainset, verbose=True)
logging.info("Ended running")
#  %%
# torch.save(perturbs, OUTPUT_PATH)
