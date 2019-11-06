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
if os.path.exists("models/cnn_perturbs.model"):
    print("Loading pre-existed perturbations")
    perturbs = torch.load("models/cnn_perturbs.model")
    perturbs = list(perturbs)
else:
    print("Creating new set of perturbation")
    perturbs = []
densities = [-0.05, 0.05]

#%%
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

init_rounds = len(perturbs)
total_rounds = 10

#%%
# SET YOUR ATTACKING IMAGE HERE!
attack_image, attack_label = mnist_trainset[0]
feeding_attack_image = attack_image.reshape(1, 1, 28, 28)
feeding_attack_label = torch.tensor([attack_label])

#%%
for r in range(init_rounds, init_rounds + total_rounds):
    print("Round:", r + 1)
    # Fetch one attack image

    # Create a random array of perturbation
    perturb = torch.zeros([1, 1, 28, 28], requires_grad=True)

    # Epsilon defines the maximum density (-e, e). It should be
    # in the range of the training set's scaled value.
    epsilon = 1

    adversarial_optimizer = optim.SGD([perturb], lr=0.1)

    # Train the adversarial noise, maximising the loss
    epochs = 50000
    for e in range(epochs):
        running_loss = 0
        adversarial_optimizer.zero_grad()

        output = model(feeding_attack_image + perturb)
        loss = -criterion(output, feeding_attack_label)
        loss.backward()
        adversarial_optimizer.step()
        running_loss += loss.item()
        perturb.data.clamp_(-epsilon, epsilon)
        if (e + 1) % 10000 == 0:
            print("\tNoise loss:", -1 * loss.item())

    # Save the perturbations
    perturbs.append(perturb)

    # Train the model again with all perturbations added.
    epochs = 5
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            for perturb in perturbs:
                for density in densities:
                    optimizer.zero_grad()

                    output = F.log_softmax(model(images + density * perturb), dim=1)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
        else:
            print("\tTraining loss: {}".format(running_loss / len(trainloader)))
    torch.save(
        model.state_dict(), "models/mnist_cnn_reinforced_{}.model".format(r + 1)
    )

# %%
perturbs = torch.stack(perturbs)
torch.save(perturbs, "models/cnn_perturbs.model")

# %%
fig, axs = plt.subplots(2, 5, figsize=(10, 4))
for p, ax in zip(perturbs, axs.ravel()):
    ax.imshow(p.detach().numpy().reshape(28, 28))
plt.show()

# %%
print("Hello?")

# %%
