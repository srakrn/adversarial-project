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
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
mnist_trainset = datasets.MNIST(
    root="mnist", train=True, download=True, transform=transform
)
mnist_testset = datasets.MNIST(
    root="mnist", train=False, download=True, transform=transform
)
trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
testloader = DataLoader(mnist_testset, batch_size=1, shuffle=True)

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
mnist_state = torch.load("models/mnist_fcnn.model")
model.load_state_dict(mnist_state)

#  %%
if os.path.exists("perturbs/on_single_point/fcnn_trainset_on_single_point.pt"):
    print("Loading pre-existed perturbations")
    perturbs = torch.load("perturbs/on_single_point/fcnn_trainset_on_single_point.pt")
    perturbs = list(perturbs)
else:
    print("Creating new set of perturbation")
    perturbs = []
densities = [-0.05, 0.05]

# %%
criterion = nn.CrossEntropyLoss()
for i, (attack_image, attack_label) in enumerate(mnist_trainset):
    print("Image:", i + 1)
    feeding_attack_image = attack_image.reshape(1, -1)
    feeding_attack_label = torch.tensor([attack_label])
    #  Fetch one attack image

    #  Create a random array of perturbation
    perturb = torch.zeros([1, 28 * 28], requires_grad=True)

    #  Epsilon defines the maximum density (-e, e). It should be
    #  in the range of the training set's scaled value.
    epsilon = 1

    adversarial_optimizer = optim.Adam([perturb], lr=0.1)

    #  Train the adversarial noise, maximising the loss
    epochs = 500
    for e in range(epochs):
        running_loss = 0
        adversarial_optimizer.zero_grad()

        output = model(feeding_attack_image + perturb)
        loss = -criterion(output, feeding_attack_label)
        loss.backward()
        adversarial_optimizer.step()
        running_loss += loss.item()
        perturb.data.clamp_(-epsilon, epsilon)
    print("\tNoise loss:", -1 * loss.item())

    #  Save the perturbations
    perturbs.append(perturb)

#  %%
perturbs = torch.stack(perturbs)
torch.save(perturbs, "perturbs/on_single_point/fcnn_trainset_on_single_point.pt")
