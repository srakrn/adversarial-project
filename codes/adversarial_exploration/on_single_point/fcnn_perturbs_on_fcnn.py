#%%
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report, confusion_matrix

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
trainloader = DataLoader(mnist_trainset, batch_size=60000, shuffle=False)
testloader = DataLoader(mnist_testset, batch_size=10000, shuffle=False)

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
perturbs = torch.load("perturbs/on_single_point/fcnn_on_single_point.pt")
perturbs = perturbs.reshape(10000, 28, 28)

# %%
density = 0.1
x_test, y_test = next(iter(testloader))
x_test = x_test.reshape(-1, 28, 28)
x_adver = x_test + density * perturbs
y_pred = model(x_test).argmax(dim=1).numpy().astype(int)
y_pred_adver = model(x_test + x_adver).argmax(dim=1).detach().numpy().astype(int)
y_pred_perturbs = model(perturbs).argmax(dim=1).detach().numpy().astype(int)

# %%
report = classification_report(y_test, y_pred)
print("Classification report of model")
print(report)

# %%
adver_report = classification_report(y_test, y_pred_adver)
print("Classification report of model (adversarial)")
print(adver_report)

# %%
perturbs_report = classification_report(y_test, y_pred_perturbs)
print("Classification report of model (only perturbs)")
print(perturbs_report)

# %%
clf_mat = confusion_matrix(y_test, y_pred)
print("Confusion matrix: True and Predicted")
print(clf_mat)

# %%
adver_clf_mat = confusion_matrix(y_test, y_pred_adver)
print("Confusion matrix: True and Adversarial Predicted")
print(adver_clf_mat)

# %%
print("Confusion matrix difference: Adversarial - Predicted")
print(adver_clf_mat - clf_mat)

# %%
print("Confusion matrix: True and Perturbations")
perturbs_clf_mat = confusion_matrix(y_test, y_pred_perturbs)
print(perturbs_clf_mat)

# %%
print("Confusion matrix difference: Adversarial - Perturbations")
print(adver_clf_mat - perturbs_clf_mat)


# %%
np.save("models/classification_results/on_single_point/fcnn_model/fcnn_testset.npy", y_pred)
np.save("models/classification_results/on_single_point/fcnn_model/fcnn_testset_adver.npy", y_pred_adver)
np.save("models/classification_results/on_single_point/fcnn_model/fcnn_testset_perturbs.npy", y_pred_perturbs)

# %%
