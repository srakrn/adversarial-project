#%%
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

#%%
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

#%%
mnist_testset = datasets.MNIST(
    root="mnist", train=False, download=True, transform=transform
)

# %%
fcnn_perturbs = torch.load("perturbs/on_single_point/fcnn_on_single_point.pt")
fcnn_perturbs = fcnn_perturbs.detach().numpy()
fcnn_perturbs = fcnn_perturbs.reshape(-1, 28 * 28)
fcnn_perturbs.shape

# %%
fcnn_adver_pred = np.load(
    "models/classification_results/on_single_point/cnn_model/cnn_testset_adver.npy"
)
fcnn_adver_pred.shape

# %%
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

# %%
def calculate_k_perturbs(
    model, perturbs, training_set, k, n_epoches=20, verbose=False, log=False
):
    km = KMeans(n_clusters=k)
    km_clusters = km.fit_predict(perturbs.reshape(len(perturbs), -1))
    print(f"Training {k} perturbs")

    k_points = []
    k_perturbs = []
    losses = []

    for i in set(km_clusters):
        if log:
            log_f = open(log, "a")
        idx = np.where(km_clusters == i)[0]
        data = [training_set[j] for j in idx]
        trainloader = DataLoader(data, batch_size=len(data), shuffle=False)

        perturb = torch.zeros([1, 28 * 28], requires_grad=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD([perturb], lr=0.03)

        if verbose:
            print(f"\tTraining #{i+1} perturb")
            print(f"\tThis set of perturbation will attack {len(data)} data points.")

        for e in range(n_epoches):
            running_loss = 0
            for images, labels in trainloader:
                images = images.reshape(-1, 28 * 28)
                optimizer.zero_grad()
                output = model(images + perturb)
                loss = -1 * criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                perturb.data.clamp(-1, 1)
        if verbose:
            print(f"\tTraining loss: {-1 * running_loss/len(trainloader)}")
        losses.append(-1 * running_loss / len(trainloader))
    if log:
        t = time.strftime("%H:%M:%S", time.localtime())
        log_f.write(f"{t},{k},")
        log_f.write(",".join([f"{i:.5f}" for i in losses]))
        log_f.write("\n")
    k_points.append(idx)
    k_perturbs.append(perturb.detach().numpy())
    return [k_points, k_perturbs, km]


# %%
ks = range(1, 101)
k_result = [
    calculate_k_perturbs(
        model,
        fcnn_perturbs,
        mnist_testset,
        i,
        n_epoches=500,
        verbose=True,
        log="perturbs/clustered/fcnn/on_perturb_gradientdesc.log",
    )
    for i in ks
]

# %%
with open("perturbs/clustered/fcnn/on_perturb_gradientdesc.pkl", "wb") as f:
    pickle.dump(k_result, f)
