# %%
import os
import sys

import torch
from torch import nn, optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attack import fgsm  # isort:skip
from mnist_helpers import mnist_resnet_model, mnist_trainset  # isort:skip

# %%
OUTPUT_PATH = "perturbs/on_single_point/mnist/resnet_fgsm_perturbs_trainset.pt"

# %%
criterion = nn.CrossEntropyLoss()
perturbs = fgsm(mnist_resnet_model, criterion, mnist_trainset, verbose=True)
#  %%
torch.save(perturbs, OUTPUT_PATH)
