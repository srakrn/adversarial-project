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
from torchvision.models import resnet18

# %%
MODEL_PATH = "models/mnist_resnet18.model"
OUTPUT_PATH = "perturbs/on_single_point/mnist/resnet_fgsm_perturbs_trainset.pt"

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
model = resnet18()
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
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
model.eval()

for i, (attack_image, attack_label) in enumerate(trainloader):
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
