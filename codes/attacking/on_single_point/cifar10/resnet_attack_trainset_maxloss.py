# %%
import os
import sys

import torch
from torch import nn, optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attack import maxloss  # isort:skip
from cifar10_helpers import cifar10_resnet_model, cifar10_trainset  # isort:skip

# %%
OUTPUT_PATH = "perturbs/on_single_point/cifar10/resnet_maxloss_perturbs_trainset.pt"

# %%
criterion = nn.CrossEntropyLoss()
perturbs = maxloss(cifar10_resnet_model, criterion, cifar10_trainset, verbose=True)
#  %%
torch.save(perturbs, OUTPUT_PATH)
