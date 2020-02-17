# %%
import os

import torch
from torch import nn, optim

from attack import fgsm
from mnist_helpers import mnist_cnn_model, mnist_testset

# %%
OUTPUT_PATH = "perturbs/on_single_point/mnist/cnn_fgsm_perturbs_testset.pt"

# %%
criterion = nn.CrossEntropyLoss()
perturbs = fgsm(mnist_cnn_model, criterion, mnist_testset, verbose=True)
#  %%
torch.save(perturbs, OUTPUT_PATH)
