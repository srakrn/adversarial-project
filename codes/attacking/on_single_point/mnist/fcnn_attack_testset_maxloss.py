# %%
import os

import torch
from torch import nn, optim

from attack import maxloss
from mnist_helpers import mnist_fcnn_model, mnist_testset

# %%
OUTPUT_PATH = "perturbs/on_single_point/mnist/fcnn_maxloss_perturbs_testset.pt"

# %%
criterion = nn.CrossEntropyLoss()
perturbs = maxloss(mnist_fcnn_model, criterion, mnist_testset, verbose=True)
#  %%
torch.save(perturbs, OUTPUT_PATH)
