# %%
import argparse
import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from clustre.attacking.on_single_point import attack
from clustre.helpers import mnist_helpers
from clustre.reinforcing import helpers as reinforce_helpers
from clustre.reinforcing.fgsm_based import reinforce

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description='Reinforce using cluster-based method')
parser.add_argument('--eps', type=float, default=0.2, help='Epsilon')

# PARAMETERS
args = parser.parse_args()

EPSILON = args.eps

logging.basicConfig(
    filename=f"logs/{os.path.basename(__file__)}.log",
    filemode="a",
    level="INFO",
    format="%(process)d-%(levelname)s-%(asctime)s-%(message)s",
)

# %%
model = mnist_helpers.mnist_cnn_model

# %%
testset_fgsm_perturbs = torch.load(
    "perturbs/on_single_point/mnist/fcnn_fgsm_perturbs_testset.pt"
)
testset_pgd_perturbs = torch.load(
    "perturbs/on_single_point/mnist/fcnn_pgd_perturbs_testset.pt"
)

# %%
logging.info(f"EPSILON = {EPSILON}")

# %%
reinforce_helpers.accuracy_unattacked(model, mnist_helpers.testloader, desc="Accuracy")
reinforce_helpers.accuracy_attacked(
    model, mnist_helpers.testloader, testset_pgd_perturbs, desc="PGD accuracy"
)
reinforce_helpers.accuracy_attacked(
    model, mnist_helpers.testloader, testset_fgsm_perturbs, desc="FGSM accuracy"
)

# %%
trainloader = DataLoader(mnist_helpers.mnist_trainset, batch_size=32)

logging.info(f"Started reinforcing on {reinforce.get_time()}")
reinforced_model = reinforce.fgsm_reinforce(model, trainloader, cuda=True)
logging.info(f"Finished reinforcing on {reinforce.get_time()}")

# %%
pn = os.path.basename(__file__).split(".")[0]
torch.save(reinforced_model.state_dict(), f"models/reinforced/{pn}.model")

# %%
new_testset_pgd_perturbs = attack.pgd(reinforced_model, nn.CrossEntropyLoss(), mnist_helpers.testloader, EPSILON)
new_testset_fgsm_perturbs = attack.fgsm(reinforced_model, nn.CrossEntropyLoss(), mnist_helpers.testloader, EPSILON)

# %%
reinforce_helpers.accuracy_unattacked(reinforced_model, mnist_helpers.testloader, desc="Accuracy")
reinforce_helpers.accuracy_attacked(
    reinforced_model, mnist_helpers.testloader, testset_pgd_perturbs, desc="PGD accuracy"
)
reinforce_helpers.accuracy_attacked(
    reinforced_model, mnist_helpers.testloader, testset_fgsm_perturbs, desc="FGSM accuracy"
)

# %%
reinforce_helpers.accuracy_unattacked(reinforced_model, mnist_helpers.testloader, desc="Accuracy on new perturbs")
reinforce_helpers.accuracy_attacked(
    reinforced_model, mnist_helpers.testloader, new_testset_pgd_perturbs, desc="PGD accuracy on new perturbs"
)
reinforce_helpers.accuracy_attacked(
    reinforced_model, mnist_helpers.testloader, new_testset_fgsm_perturbs, desc="FGSM accuracy on new perturbs"
)
