# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import classification_report
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# %%
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# %%
testset = datasets.CIFAR10(
    root="cifar10", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

# %%
class CifarCnn(nn.Module):
    def __init__(self):
        super(CifarCnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = CifarCnn()
state = torch.load("models/cifar10_cnn_96.model")
model.load_state_dict(state)

# %%
y_test = []
y_pred = []
for images, labels in testloader:
    outputs = model(images).argmax(axis=1)
    y_test += labels.tolist()
    y_pred += outputs.tolist()

# %%
print(classification_report(y_test, y_pred))
