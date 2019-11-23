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
testloader = DataLoader(mnist_testset, batch_size=1, shuffle=True)

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
nth_training_points = [
    1,
    21,
    34,
    3,
    6,
    8,
    5,
    16,
    25,
    7,
    10,
    12,
    2,
    9,
    20,
    0,
    11,
    35,
    13,
    18,
    32,
    15,
    29,
    38,
    17,
    31,
    41,
    4,
    19,
    22,
]

#%%
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

#%%
perturbs = []
for i in nth_training_points:
    print("Index:", i)
    # Fetch one attack image
    attack_image, attack_label = mnist_trainset[i]
    feeding_attack_image = attack_image.reshape(1, -1)
    feeding_attack_label = torch.tensor([attack_label])

    # Create a random array of perturbation
    perturb = torch.zeros([1, 28 * 28], requires_grad=True)

    # Epsilon defines the maximum density (-e, e). It should be
    # in the range of the training set's scaled value.
    epsilon = 1

    adversarial_optimizer = optim.SGD([perturb], lr=0.1)

    # Train the adversarial noise, maximising the loss
    epochs = 10000
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

    # Save the perturbations
    perturbs.append(perturb)

# %%
perturbs = torch.stack(perturbs)
torch.save(perturbs, "models/multiclass_fcnn_perturbs.model")

# %%
perturbs = torch.load("models/multiclass_fcnn_perturbs.model")
fig, axs = plt.subplots(10, 3, figsize=(3, 10))
for p, ax in zip(perturbs, axs.ravel()):
    ax.imshow(p.detach().numpy().reshape(28, 28))
    ax.axis('off')
plt.show()


# %%
model2 = MnistFcnn()
mnist_state = torch.load("models/mnist_fcnn.model")
model2.load_state_dict(mnist_state)

training_feature = mnist_testset[0][0]
x_legend = np.linspace(-1, 1, 101)
attacked_images = []
for i in x_legend:
    attacked_images.append(i * perturbs[0].to("cpu") + training_feature)
model_probs = []
for i in attacked_images:
    model_probs.append(model2(i.view(-1, 1, 28, 28)).detach().numpy())
model_probs = np.array(model_probs).reshape(101, 10).T

for i, v in enumerate(model_probs):
    plt.plot(x_legend, v, label="{}".format(i))

plt.legend()
plt.show()

# %%
