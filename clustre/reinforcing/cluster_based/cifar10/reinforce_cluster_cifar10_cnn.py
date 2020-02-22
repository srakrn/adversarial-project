# %%
import logging
import os
import sys

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cifar10_helpers  # isort:skip
import reinforce  # isort:skip

logging.basicConfig(
    filename=f"logs/{os.path.basename(__file__)}.log",
    filemode="a",
    level="INFO",
    format="%(process)d-%(levelname)s-%(asctime)s-%(message)s",
)

# %%
model = cifar10_helpers.cifar10_cnn_model

# %%
trainset_perturbs = torch.load(
    "perturbs/on_single_point/cifar10/cnn_attack_fgsm_trainset.pt"
)
testset_perturbs = torch.load(
    "perturbs/on_single_point/cifar10/cnn_attack_fgsm_testset.pt"
)

# %%
epsilon = 0.2

# %%
y_test = []
y_pred = []
for image, label in cifar10_helpers.testloader:
    y_test.append(label.item())
    y_pred.append(model(image).argmax(axis=1).item())
print("Original model report:")
print(classification_report(y_test, y_pred))
logging.info(classification_report(y_test, y_pred))

# %%
y_test = []
y_pred = []
for (image, label), perturb in zip(cifar10_helpers.testloader, testset_perturbs):
    y_test.append(label.item())
    y_pred.append(
        model(image + 0.2 * perturb.reshape(1, 3, 32, 32)).argmax(axis=1).item()
    )
print("Adversarial on original model report:")
print(classification_report(y_test, y_pred))
logging.info(classification_report(y_test, y_pred))


# %%
k = 100

# %%
train_target, train_perturb, train_km = reinforce.calculate_k_perturbs(
    model, cifar10_helpers.cifar10_trainset, trainset_perturbs.detach().numpy(), k
)

# %%
ad = reinforce.AdversarialDataset(
    cifar10_helpers.cifar10_trainset, train_target, train_perturb
)
adversarialloader = DataLoader(ad, batch_size=16, shuffle=True)

# %%
logging.info(f"Started reinforcing on {reinforce.get_time()}")
reinforced_model = reinforce.k_reinforce(
    model, cifar10_helpers.trainloader, adversarialloader, n_epoches=20, adversarial_weight=5
)
logging.info(f"Finished reinforcing on {reinforce.get_time()}")

# %%
y_test = []
y_pred = []
for image, label in cifar10_helpers.testloader:
    y_test.append(label.item())
    y_pred.append(model(image).argmax(axis=1).item())
print("Reinforced model report:")
print(classification_report(y_test, y_pred))
logging.info(classification_report(y_test, y_pred))


# %%
y_test = []
y_pred = []
for (image, label), perturb in zip(cifar10_helpers.testloader, testset_perturbs):
    y_test.append(label.item())
    y_pred.append(
        model(image + 0.2 * perturb.reshape(1, 3, 32, 32)).argmax(axis=1).item()
    )
print("Adversarial on reinforced model report:")
print(classification_report(y_test, y_pred))
logging.info(classification_report(y_test, y_pred))
