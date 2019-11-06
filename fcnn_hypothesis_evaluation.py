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
from sklearn.metrics import confusion_matrix 

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
if os.path.exists("models/fcnn_perturbs.model"):
    print("Loading pre-existed perturbations")
    perturbs = torch.load("models/fcnn_perturbs.model")
    perturbs = list(perturbs)
else:
    print("Creating new set of perturbation")
    perturbs = []

# %%
fig, axs = plt.subplots(2, 5, figsize=(10, 4))
for perturb, ax in zip(perturbs, axs.ravel()):
    ax.imshow(perturb.detach().reshape(28, 28))
plt.suptitle("Adversarial perturbations")
plt.show()

# %%
n_perturbs = 0
density = 0.2
perturb = perturbs[n_perturbs].reshape(1, 28, 28)

# %%
n_training_samples = 145
training_feature, training_target = mnist_trainset[n_training_samples]
print(training_target)

# %%
attacked_training_feature = training_feature + density * perturb

# %%
fig, axs = plt.subplots(1, 3, figsize=(9, 3))
axs[0].imshow(training_feature.numpy().reshape(28, 28))
axs[1].imshow(perturb.detach().numpy().reshape(28, 28))
axs[2].imshow(attacked_training_feature.detach().numpy().reshape(28, 28))
plt.show()

predicted_target = model(attacked_training_feature)
print(predicted_target.argmax())

# %%
x_legend = np.linspace(-1, 1, 101)
attacked_images = []
for i in x_legend:
    attacked_images.append(i * perturb + training_feature)
model_probs = []
for i in attacked_images:
    model_probs.append(model(i.view(i.shape[0], -1)).detach().numpy())
model_probs = np.array(model_probs).reshape(101, 10).T

for i, v in enumerate(model_probs):
    plt.plot(x_legend, v, label="{}".format(i))

plt.legend()
plt.show()

# %%
predicted_targets = model(mnist_testset.data.float()).argmax(axis=1).detach().numpy()
actual_targets = mnist_testset.targets.numpy()
print("Model's confusion matrix")
cm = confusion_matrix(actual_targets, predicted_targets)
print(cm)

# %%
predicted_adversarial_targets = (
    model(mnist_testset.data.float() + density * perturb)
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
