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
MODEL_PATH = "models/mnist_fcnn.model"
OUTPUT_PATH = "perturbs/on_single_point/mnist/fcnn_fgsm_perturbs_testset.pt"

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


#  %%
model = MnistFcnn()
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
