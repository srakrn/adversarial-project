# %%
import os

import torch
from torch import nn, optim

from attack import fgsm
from cifar10_helpers import cifar10_resnet_model, cifar10_trainset

# %%
OUTPUT_PATH = "perturbs/on_single_point/cifar10/resnet_fgsm_perturbs_trainset.pt"

# %%
criterion = nn.CrossEntropyLoss()
perturbs = fgsm(cifar10_resnet_model, criterion, cifar10_trainset, verbose=True)
#  %%
torch.save(perturbs, OUTPUT_PATH)
