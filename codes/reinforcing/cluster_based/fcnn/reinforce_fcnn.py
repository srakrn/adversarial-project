# %%
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import reinforce  # isort:skip

# %%
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# %%
mnist_trainset = datasets.MNIST(
    root="mnist", train=True, download=True, transform=transform
)
mnist_testset = datasets.MNIST(
    root="mnist", train=False, download=True, transform=transform
)


# %%
trainloader = DataLoader(mnist_trainset, batch_size=1, shuffle=False)
full_trainloader = DataLoader(
    mnist_trainset, batch_size=len(mnist_trainset), shuffle=False
)
testloader = DataLoader(mnist_testset, batch_size=1, shuffle=False)

# %%
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


# %%
model = MnistFcnn()
mnist_state = torch.load("models/mnist_fcnn.model")
model.load_state_dict(mnist_state)

# %%
trainset_perturbs = torch.load(
    "perturbs/on_single_point/mnist/fcnn_fgsm_perturbs_trainset.pt"
)
testset_perturbs = torch.load(
    "perturbs/on_single_point/mnist/fcnn_fgsm_perturbs_testset.pt"
)

# %%
epsilon = 0.2

# %%
y_test = []
y_pred = []
for image, label in testloader:
    y_test.append(label.item())
    y_pred.append(model(image).argmax(axis=1).item())
print("Original model report:")
print(classification_report(y_test, y_pred))

# %%
y_test = []
y_pred = []
for (image, label), perturb in zip(testloader, testset_perturbs):
    y_test.append(label.item())
    y_pred.append(
        model(image + 0.2 * perturb.reshape(1, 1, 28, 28)).argmax(axis=1).item()
    )
print("Adversarial on original model report:")
print(classification_report(y_test, y_pred))


# %%
k = 100

# %%
train_target, train_perturb, train_km = reinforce.calculate_k_perturbs(
    model, mnist_trainset, trainset_perturbs.detach().numpy(), k
)
"""

# %%
test_target, test_perturb, test_km = reinforce.calculate_k_perturbs(
    model, mnist_testset, testset_perturbs.detach().numpy(), k, attack_method="maxloss"
)

# %%
y_test = []
y_pred = []
for i, (image, label) in enumerate(testloader):
    y_test.append(label.item())
    perturb = torch.tensor(reinforce.get_nth_perturb(i, test_target, test_perturb))
    y_pred.append(
        model(image + 0.2 * perturb.reshape(1, 1, 28, 28)).argmax(axis=1).item()
    )

print(classification_report(y_test, y_pred))
"""

# %%
ad = reinforce.AdversarialDataset(mnist_trainset, train_target, train_perturb)
adversarialloader = DataLoader(ad, batch_size=16, shuffle=True)

# %%
print(f"Started reinforcing on {reinforce.get_time()}")
reinforced_model = reinforce.k_reinforce(model, trainloader, adversarialloader)
print(f"Finished reinforcing on {reinforce.get_time()}")

# %%
y_test = []
y_pred = []
for image, label in testloader:
    y_test.append(label.item())
    y_pred.append(model(image).argmax(axis=1).item())
print("Reinforced model report:")
print(classification_report(y_test, y_pred))


# %%
y_test = []
y_pred = []
for (image, label), perturb in zip(testloader, trainset_perturbs):
    y_test.append(label.item())
    y_pred.append(
        model(image + 0.2 * perturb.reshape(1, 1, 28, 28)).argmax(axis=1).item()
    )

print(classification_report(y_test, y_pred))

"""

# %%
rem = MnistFcnn()

# %%
criterion = nn.CrossEntropyLoss(reduction="none")
optimizer = optim.Adam(rem.parameters())
epochs = 10
for e in range(epochs):
    running_loss = 0
    for (images, labels), (adver_images, adver_labels) in zip(
        trainloader, perturbloader
    ):
        X = torch.cat([images, adver_images], 0)
        y = torch.cat([labels, adver_labels], 0)
        w = torch.tensor(
            [
                1 if i < len(labels) else 2
                for i in range(len(labels) + len(adver_labels))
            ]
        ).float()
        optimizer.zero_grad()

        output = F.log_softmax(rem(X), dim=1)
        loss = torch.dot(criterion(output, y), w)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")


# %%
y_test = []
y_pred = []
for image, label in testloader:
    y_test.append(label.item())
    y_pred.append(model(image).argmax(axis=1).item())

print(classification_report(y_test, y_pred))


# %%
y_test = []
y_pred = []
for (image, label), perturb in zip(testloader, trainset_perturbs):
    y_test.append(label.item())
    y_pred.append(
        model(image + 0.2 * perturb.reshape(1, 1, 28, 28)).argmax(axis=1).item()
    )

print(classification_report(y_test, y_pred))
"""
