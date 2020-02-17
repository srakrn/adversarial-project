# %%
import os
import sys

import torch
from torch import nn, optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attack import fgsm  # isort:skip
from mnist_helpers import mnist_cnn_model, mnist_testset  # isort:skip

# %%
OUTPUT_PATH = "perturbs/on_single_point/mnist/cnn_fgsm_perturbs_testset.pt"

# %%
criterion = nn.CrossEntropyLoss()
perturbs = fgsm(mnist_cnn_model, criterion, mnist_testset, verbose=True)
#  %%
torch.save(perturbs, OUTPUT_PATH)
