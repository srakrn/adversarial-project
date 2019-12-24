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
class MnistCnn(nn.Module):
    def __init__(self):
        super(MnistCnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# %%
model = MnistCnn()
mnist_state = torch.load("models/weak_learner/on_single_point/cnn/fcnn_perturbs.model")
model.load_state_dict(mnist_state)

# %%
x_test, y_test = next(iter(testloader))
y_pred = model(x_test).argmax(dim=1).numpy().astype(int)

# %%
report = classification_report(y_test, y_pred)
print("Classification report of model")
print(report)