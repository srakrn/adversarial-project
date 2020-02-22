# %%
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

# %%
CNN_PATH = "models/cifar10_cnn_20.model"
RESNET_PATH = "models/cifar10_resnet18_90.model"

# %%
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
cifar10_trainset = datasets.CIFAR10(
    root="cifar10", train=True, download=True, transform=transform
)
cifar10_testset = datasets.CIFAR10(
    root="cifar10", train=False, download=True, transform=transform
)
trainloader = DataLoader(cifar10_trainset, batch_size=1, shuffle=False)
testloader = DataLoader(cifar10_testset, batch_size=1, shuffle=False)

#  %%
class Cifar10Cnn(nn.Module):
    def __init__(self):
        super(Cifar10Cnn, self).__init__()
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


#  %%
cifar10_cnn_model = Cifar10Cnn()
cifar10_cnn_state = torch.load(CNN_PATH)
cifar10_cnn_model.load_state_dict(cifar10_cnn_state)

# %%
cifar10_resnet_model = resnet18()
cifar10_resnet_model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
cifar10_resnet_state = torch.load(RESNET_PATH)
cifar10_resnet_model.load_state_dict(cifar10_resnet_state)
