#%%
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report, confusion_matrix

#%%
torch.manual_seed(0)

#%%
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
mnist_testset = datasets.MNIST(
    root="mnist", train=False, download=True, transform=transform
)
testloader = DataLoader(mnist_testset, batch_size=10000, shuffle=False)

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

# %%
model = MnistFcnn()
mnist_state = torch.load("models/weak_learner/on_single_point/fcnn/fcnn_perturbs.model")
model.load_state_dict(mnist_state)

# %%
x_test, y_test = next(iter(testloader))
y_pred = model(x_test).argmax(dim=1).numpy().astype(int)

# %%
report = classification_report(y_test, y_pred)
print("Classification report of model")
print(report)

# %%