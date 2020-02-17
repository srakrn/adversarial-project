# %%
import os

import torch
from torch import nn, optim

from attack import maxloss
from mnist_helpers import mnist_cnn_model, mnist_trainset

# %%
OUTPUT_PATH = "perturbs/on_single_point/mnist/cnn_maxloss_perturbs_trainset.pt"

# %%
criterion = nn.CrossEntropyLoss()
perturbs = maxloss(mnist_cnn_model, criterion, mnist_trainset, verbose=True)
#  %%
torch.save(perturbs, OUTPUT_PATH)
