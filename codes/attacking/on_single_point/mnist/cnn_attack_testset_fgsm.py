# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# %%
MODEL_PATH = "models/mnist_cnn.model"
OUTPUT_PATH = "perturbs/on_single_point/mnist/cnn_fgsm_perturbs_testset.pt"

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
model = MnistCnn()
mnist_state = torch.load(MODEL_PATH)
model.load_state_dict(mnist_state)

#  %%
if os.path.exists(OUTPUT_PATH):
    print("Loading pre-existed perturbations")
    perturbs = torch.load(OUTPUT_PATH)
    perturbs = list(perturbs)
else:
    print("Creating new set of perturbation")
    perturbs = []

# %%
criterion = nn.CrossEntropyLoss()
for i, (attack_image, attack_label) in enumerate(testloader):
    print("Image:", i + 1)

    model.zero_grad()
    attack_image.requires_grad = True

    output = model(attack_image)
    loss = -criterion(output, attack_label)
    loss.backward()

    perturb = attack_image.grad.data.sign()

    perturbs.append(perturb)

#  %%
perturbs = torch.stack(perturbs)
torch.save(perturbs, OUTPUT_PATH)
