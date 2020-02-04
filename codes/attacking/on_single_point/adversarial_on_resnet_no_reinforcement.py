#%%
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

#%%
torch.manual_seed(0)

#%%
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
testloader = DataLoader(mnist_testset, batch_size=10, shuffle=True)

# %%
model = resnet18(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
mnist_state = torch.load("models/mnist_resnet18.model")
model.load_state_dict(mnist_state)
model = model.to('cuda')

# %%
if os.path.exists("perturbs/resnet_on_single_point.model"):
    print("Loading pre-existed perturbations")
    perturbs = torch.load("perturbs/resnet_on_single_point.model")
    perturbs = list(perturbs)
else:
    print("Creating new set of perturbation")
    perturbs = []
densities = [-0.05, 0.05]

#%%
criterion = nn.CrossEntropyLoss()
for i, (attack_image, attack_label) in enumerate(testloader): 
    attack_image = attack_image.to('cuda')
    attack_label = attack_label.to('cuda')

    # Create a random array of perturbation
    perturb = torch.zeros([10, 1, 28, 28], requires_grad=True, device='cuda')

    # Epsilon defines the maximum density (-e, e). It should be
    # in the range of the training set's scaled value.
    epsilon = 1
    adversarial_optimizer = optim.SGD([perturb], lr=0.1)

    # Train the adversarial noise, maximising the loss
    # Allow the maximum of 3 attempts
    for _ in range(3):
        epochs = 1000
        for e in range(epochs):
            running_loss = 0
            adversarial_optimizer.zero_grad()

            output = model(attack_image + perturb)
            loss = -criterion(output, attack_label)
            loss.backward()
            adversarial_optimizer.step()
            running_loss += loss.item()
            perturb.data.clamp_(-epsilon, epsilon)
        print(running_loss)
        if running_loss < -15:
            break

    # Save the perturbations
    perturb = perturb.to('cpu')
    perturbs.append(perturb)

# %%
perturbs = torch.stack(perturbs)
purturbs = perturbs.reshape(-1, 28, 28)
torch.save(perturbs, "perturbs/resnet_on_single_point.model")