# %%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# %%
torch.manual_seed(0)

# %%
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# %%
mnist_testset = datasets.MNIST(
    root="mnist", train=False, download=True, transform=transform
)

testloader = DataLoader(mnist_testset, batch_size=10000, shuffle=False)

X_test, y_test = next(iter(testloader))
# %%
with open("perturbs/partitioned/fcnn/on_perturb_minibatch.pkl", "rb") as f:
    k_perturb = pickle.load(f)

# %%
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
density = 0.2

# %%
y_pred = model(
    X_test + density * torch.Tensor(k_perturb[0][1][0].reshape(-1, 1, 28, 28))
).argmax(axis=1)
print(classification_report(y_test, y_pred))

# %%
def get_n_perturbs(n_attacks, idx):
    attack_target, attack_perturb, _ = k_perturb[n_attacks - 1]

    for i in range(len(attack_target)):
        if idx in attack_target[i]:
            return attack_perturb[i]


# %%
n = 100

# %%
if n == 0:
    y_pred = model(X_test).argmax(axis=1)
else:
    n_perturbs = [get_n_perturbs(n, i) for i in range(10000)]
    n_perturbs = torch.Tensor(n_perturbs).reshape(-1, 1, 28, 28)
    y_pred = model(X_test + density * torch.Tensor(n_perturbs)).argmax(axis=1)
print(classification_report(y_test, y_pred))


# %%
ns = [0, 1, 2, 3, 4, 5, 7, 10, 20, 30, 40, 50, 70, 100]
score = []
for n in ns:
    if n == 0:
        y_pred = model(X_test).argmax(axis=1)
    else:
        n_perturbs = [get_n_perturbs(n, i) for i in range(10000)]
        n_perturbs = torch.Tensor(n_perturbs).reshape(-1, 1, 28, 28)
        y_pred = model(X_test + density * torch.Tensor(n_perturbs)).argmax(axis=1)
    score.append(accuracy_score(y_test, y_pred))
plt.plot(ns, score, ".-")
plt.show()

# %%
