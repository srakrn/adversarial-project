# %%
import os

import torch
from torch import nn, optim

from attack import fgsm
from mnist_helpers import mnist_fcnn_model, mnist_trainset

# %%
OUTPUT_PATH = "perturbs/on_single_point/mnist/fcnn_fgsm_perturbs_trainset.pt"

# %%
criterion = nn.CrossEntropyLoss()
perturbs = fgsm(mnist_fcnn_model, criterion, mnist_trainset, verbose=True)
#  %%
torch.save(perturbs, OUTPUT_PATH)
