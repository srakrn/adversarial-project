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

from clustre.helpers import helpers, mnist_helpers
from clustre.reinforcing.fgsm_based import reinforce

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

logging.basicConfig(
    filename=f"logs/{os.path.basename(__file__)}.log",
    filemode="a",
    level="INFO",
    format="%(process)d-%(levelname)s-%(asctime)s-%(message)s",
)

testset_perturbs = torch.load(
    "perturbs/on_single_point/mnist/fcnn_fgsm_perturbs_testset.pt"
)

trainloader = DataLoader(mnist_helpers.mnist_trainset, batch_size=32)
model = mnist_helpers.mnist_resnet_model
model.eval()

# %%
epsilon = 0.2

# %%
y_test = []
y_pred = []
for image, label in mnist_helpers.testloader:
    y_test.append(label.item())
    y_pred.append(model(image).argmax(axis=1).item())
print("Original model report:")
print(classification_report(y_test, y_pred))
logging.info(classification_report(y_test, y_pred))

# %%
y_test = []
y_pred = []
for (image, label), perturb in zip(mnist_helpers.testloader, testset_perturbs):
    perturb = perturb.to("cpu")
    y_test.append(label.item())
    y_pred.append(
        model(image + 0.2 * perturb.reshape(1, 1, 28, 28)).argmax(axis=1).item()
    )
print("Adversarial on original model report:")
print(classification_report(y_test, y_pred))
logging.info(classification_report(y_test, y_pred))
# %%
model.train()
logging.info(f"Started reinforcing on {reinforce.get_time()}")
reinforced_model = reinforce.fgsm_reinforce(model, trainloader)
logging.info(f"Finished reinforcing on {reinforce.get_time()}")
model.eval()

# %%
y_test = []
y_pred = []
for image, label in mnist_helpers.testloader:
    y_test.append(label.item())
    y_pred.append(model(image).argmax(axis=1).item())
print("Reinforced model report:")
print(classification_report(y_test, y_pred))
logging.info(classification_report(y_test, y_pred))


# %%
y_test = []
y_pred = []
for (image, label), perturb in zip(mnist_helpers.testloader, testset_perturbs):
    perturb = perturb.to("cpu")
    y_test.append(label.item())
    y_pred.append(
        model(image + 0.2 * perturb.reshape(1, 1, 28, 28)).argmax(axis=1).item()
    )
print("Adversarial on reinforced model report:")
print(classification_report(y_test, y_pred))
logging.info(classification_report(y_test, y_pred))
