# %%
import os

import torch
from torch import nn, optim

from attack import fgsm
from mnist_helpers import mnist_resnet_model, mnist_trainset

# %%
OUTPUT_PATH = "perturbs/on_single_point/mnist/resnet_fgsm_perturbs_trainset.pt"

# %%
criterion = nn.CrossEntropyLoss()
perturbs = fgsm(mnist_resnet_model, criterion, mnist_trainset, verbose=True)
#  %%
torch.save(perturbs, OUTPUT_PATH)
