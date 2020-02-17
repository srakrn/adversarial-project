# %%
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

import attack

# %%
FCNN_PATH = "models/mnist_fcnn.model"
CNN_PATH = "models/mnist_cnn.model"
RESNET_PATH = "models/mnist_resnet18.model"

# %%
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
mnist_trainset = datasets.MNIST(
    root="mnist", train=True, download=True, transform=transform
)
mnist_testset = datasets.MNIST(
    root="mnist", train=False, download=True, transform=transform
)
trainloader = DataLoader(mnist_trainset, batch_size=1, shuffle=False)
testloader = DataLoader(mnist_testset, batch_size=1, shuffle=False)

#  %%
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


#  %%
mnist_fcnn_model = MnistFcnn()
mnist_fcnn_state = torch.load(FCNN_PATH)
mnist_fcnn_model.load_state_dict(mnist_fcnn_state)

#  %%
mnist_cnn_model = MnistCnn()
mnist_cnn_state = torch.load(CNN_PATH)
mnist_cnn_model.load_state_dict(mnist_cnn_state)

# %%
mnist_resnet_model = resnet18()
mnist_resnet_model.conv1 = nn.Conv2d(
    1, 64, kernel_size=7, stride=2, padding=3, bias=False
)
mnist_resnet_model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
mnist_resnet_state = torch.load(RESNET_PATH)
mnist_resnet_model.load_state_dict(mnist_resnet_state)
