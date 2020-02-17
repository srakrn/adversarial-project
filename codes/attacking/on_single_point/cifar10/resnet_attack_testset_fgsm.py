# %%
import os

import torch
from torch import nn, optim

from attack import fgsm
from cifar10_helpers import cifar10_resnet_model, cifar10_testset

# %%
OUTPUT_PATH = "perturbs/on_single_point/cifar10/resnet_fgsm_perturbs_testset.pt"

# %%
criterion = nn.CrossEntropyLoss()
perturbs = fgsm(cifar10_resnet_model, criterion, cifar10_testset, verbose=True)
#  %%
torch.save(perturbs, OUTPUT_PATH)
