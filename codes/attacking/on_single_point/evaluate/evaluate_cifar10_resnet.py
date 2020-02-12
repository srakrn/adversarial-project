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
model = models.resnet18()
model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
state = torch.load("models/cifar10_resnet18.model")
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


# %%
