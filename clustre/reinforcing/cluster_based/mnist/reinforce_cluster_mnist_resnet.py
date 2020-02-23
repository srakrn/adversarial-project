# %%
import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from clustre.helpers import mnist_helpers
from clustre.reinforcing.cluster_based import helpers as reinforce_helpers
from clustre.reinforcing.cluster_based import reinforce

logging.basicConfig(
    filename=f"logs/{os.path.basename(__file__)}.log",
    filemode="a",
    level="INFO",
    format="%(process)d-%(levelname)s-%(asctime)s-%(message)s",
)

# %%
model = mnist_helpers.mnist_resnet_model

# %%
trainset_fgsm_perturbs = torch.load(
    "perturbs/on_single_point/mnist/resnet_fgsm_perturbs_trainset.pt"
)
testset_fgsm_perturbs = torch.load(
    "perturbs/on_single_point/mnist/resnet_fgsm_perturbs_testset.pt"
)
trainset_pgd_perturbs = torch.load(
    "perturbs/on_single_point/mnist/resnet_pgd_perturbs_trainset.pt"
)
testset_pgd_perturbs = torch.load(
    "perturbs/on_single_point/mnist/resnet_pgd_perturbs_testset.pt"
)

# %%
epsilon = 0.2
k = 100

# %%
reinforce_helpers.accuracy_unattacked(model, mnist_helpers.testloader)
reinforce_helpers.accuracy_attacked(
    model, mnist_helpers.testloader, testset_pgd_perturbs
)

# %%
train_target, train_perturb, train_km = reinforce.calculate_k_perturbs(
    model,
    mnist_helpers.mnist_trainset,
    trainset_pgd_perturbs.detach().numpy(),
    k,
    verbose=True,
)

ad = reinforce.AdversarialDataset(
    mnist_helpers.mnist_trainset, train_target, train_perturb
)
adversarialloader = DataLoader(ad, batch_size=16, shuffle=True)

trainloader = DataLoader(mnist_helpers.mnist_trainset, batch_size=32, shuffle=False)
logging.info(f"Started reinforcing on {reinforce.get_time()}")
reinforced_model = reinforce.k_reinforce(model, trainloader, adversarialloader)
logging.info(f"Finished reinforcing on {reinforce.get_time()}")
