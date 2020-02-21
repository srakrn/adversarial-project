#%%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#%%
torch.manual_seed(0)

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

# %%
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


# %%
model = MnistCnn()
mnist_state = torch.load("models/mnist_cnn.model")
model.load_state_dict(mnist_state)

# %%
fcnn_perturbs = torch.load("perturbs/on_single_point/fcnn_on_single_point.pt")
cnn_perturbs = torch.load("perturbs/on_single_point/cnn_on_single_point.pt")

# %%
fig, axs = plt.subplots(2, 5, figsize=(10, 4))
for perturb, ax in zip(fcnn_perturbs, axs.ravel()):
    ax.imshow(perturb.detach().reshape(28, 28))
plt.suptitle("Adversarial perturbations (FCNN)")
plt.show()

# %%
fig, axs = plt.subplots(2, 5, figsize=(10, 4))
for perturb, ax in zip(cnn_perturbs, axs.ravel()):
    ax.imshow(perturb.detach().reshape(28, 28))
plt.suptitle("Adversarial perturbations (CNN)")
plt.show()

# %%
n_perturbs = 0
perturbs_array = "cnn_perturbs"
density = 0.2
perturb = eval(perturbs_array)[n_perturbs].reshape(1, 28, 28)

# %%
n_training_samples = 0
training_feature, training_target = mnist_trainset[n_training_samples]
print(training_target)

# %%
attacked_training_feature = training_feature + density * perturb
attacked_training_feature = attacked_training_feature.reshape(1, 1, 28, 28)

# %%
predicted_target = model(attacked_training_feature).argmax()

fig, axs = plt.subplots(1, 3, figsize=(9, 3))
axs[0].imshow(training_feature.numpy().reshape(28, 28))
axs[0].set_title("Training point #{}".format(n_training_samples))
axs[1].imshow(perturb.detach().numpy().reshape(28, 28))
axs[1].set_title("{}[{}]".format(perturbs_array, n_perturbs))
axs[2].imshow(attacked_training_feature.detach().numpy().reshape(28, 28))
axs[2].set_title("Adversarial image")
plt.suptitle("Predicted as {}".format(predicted_target))
plt.show()
# %%
x_legend = np.linspace(-1, 1, 101)
attacked_images = []
for i in x_legend:
    attacked_images.append(i * perturb + training_feature)
model_probs = []
for i in attacked_images:
    model_probs.append(model(i.view(-1, 1, 28, 28)).detach().numpy())
model_probs = np.array(model_probs).reshape(101, 10).T

for i, v in enumerate(model_probs):
    plt.plot(x_legend, v, label="{}".format(i))

plt.legend()
plt.show()

# %%
predicted_targets = (
    model(mnist_testset.data.float().reshape(-1, 1, 28, 28))
    .argmax(axis=1)
    .detach()
    .numpy()
)
actual_targets = mnist_testset.targets.numpy()
print("Model's confusion matrix")
cm = confusion_matrix(actual_targets, predicted_targets)
print(cm)

# %%
predicted_adversarial_targets = (
    model(
        mnist_testset.data.float().reshape(-1, 1, 28, 28)
        + (density * perturb).reshape(1, 1, 28, 28)
    )
    .argmax(axis=1)
    .detach()
    .numpy()
)
actual_targets = mnist_testset.targets.numpy()
print("Model's adversarial confusion matrix")
adversarial_cm = confusion_matrix(actual_targets, predicted_adversarial_targets)
print(adversarial_cm)

# %%
print("Adversarial and non-adversarial differences")
print(adversarial_cm - cm)

# %%
