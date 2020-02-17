# %%
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from partition import calculate_k_perturbs

# %%
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# %%
mnist_testset = datasets.MNIST(
    root="mnist", train=False, download=True, transform=transform
)

#  %%
fcnn_perturbs = torch.load("perturbs/on_single_point/fcnn_on_single_point.pt")
fcnn_perturbs = fcnn_perturbs.detach().numpy()
fcnn_perturbs = fcnn_perturbs.reshape(-1, 28 * 28)
fcnn_perturbs.shape

#  %%
torch.manual_seed(0)

mnist_trainset = datasets.MNIST(
    root="mnist", train=True, download=True, transform=transform
)

mnist_testset = datasets.MNIST(
    root="mnist", train=False, download=True, transform=transform
)
trainloader = DataLoader(mnist_trainset, batch_size=100, shuffle=False)
testloader = DataLoader(mnist_testset, batch_size=100, shuffle=False)


class MnistFcnn(nn.Module):
    def __init__(self):
        super(MnistFcnn, self).__init__()
        self.linear1 = nn.Linear(784, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.reshape(-1, 28 * 28)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


model = MnistFcnn()
mnist_state = torch.load("models/mnist_fcnn.model")
model.load_state_dict(mnist_state)

#  %%
ks = range(1, 101)
k_result = [
    calculate_k_perturbs(
        model,
        fcnn_perturbs,
        mnist_testset,
        i,
        n_epoches=500,
        verbose=True,
        log="perturbs/partitioned/fcnn/on_perturb_gradientdesc.log",
    )
    for i in ks
]

#  %%
with open("perturbs/partitioned/fcnn/on_perturb.pkl", "wb") as f:
    pickle.dump(k_result, f)
