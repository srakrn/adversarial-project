# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#%%
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

#%%
mnist_trainset = datasets.MNIST(
    root="mnist", train=True, download=True, transform=transform
)
mnist_testset = datasets.MNIST(
    root="mnist", train=False, download=True, transform=transform
)

trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
testloader = DataLoader(mnist_testset, batch_size=1, shuffle=False)

#%%
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


#%%
model = MnistFcnn()

#%%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(model.parameters(), lr=0.03)

#%%
epochs = 10
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
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
n = 1000
density = 0.2

# %%
def calculate_k_perturbs(
    model, training_set, k, n_epoches=20, verbose=False, log=False
):
    loader = DataLoader(training_set, batch_size=len(training_set), shuffle=False)
    X, y = next(iter(loader))
    km = KMeans(n_clusters=k)
    km_clusters = km.fit_predict(X.reshape(len(X), -1))
    print(f"Training {k} perturbs")

    k_points = []
    k_perturbs = []
    losses = []

    for i in set(km_clusters):
        if log:
            log_f = open(log, "a")
        idx = np.where(km_clusters == i)[0]
        data = [training_set[j] for j in idx]
        trainloader = DataLoader(data, batch_size=len(data), shuffle=False)

        perturb = torch.zeros([1, 28 * 28], requires_grad=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD([perturb], lr=0.03)

        if verbose:
            print(f"\tTraining #{i+1} perturb")
            print(f"\tThis set of perturbation will attack {len(data)} data points.")

        for e in range(n_epoches):
            running_loss = 0
            for images, labels in trainloader:
                images = images.reshape(-1, 28 * 28)
                optimizer.zero_grad()
                output = model(images + perturb)
                loss = -1 * criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                perturb.data.clamp(-1, 1)
        if verbose:
            print(f"\tTraining loss: {-1 * running_loss/len(trainloader)}")
        losses.append(-1 * running_loss / len(trainloader))
        k_points.append(idx)
        k_perturbs.append(perturb.detach().numpy())
    if log:
        t = time.strftime("%H:%M:%S", time.localtime())
        log_f.write(f"{t},{k},")
        log_f.write(",".join([f"{i:.5f}" for i in losses]))
        log_f.write("\n")
    return [k_points, k_perturbs, km]


# %%
attack_targets, attack_perturbs, attack_clusters = calculate_k_perturbs(
    model, mnist_trainset, n, verbose=True
)

# %%
fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for ax, perturb in zip(axs.ravel(), attack_perturbs):
    ax.imshow(perturb.reshape(28, 28))
plt.show()

# %%
y_test = []
y_pred = []
for i, (image, label) in enumerate(testloader):
    for j, attack_target in enumerate(attack_targets):
        if i in attack_target:
            perturb = torch.tensor(attack_perturbs[j]).reshape(1, 1, 28, 28)
            if i % 100 == 0:
                print(i, j)
    x = image + density * perturb
    y_test.append(label.item())
    y_pred.append(model(x).argmax().item())

print(classification_report(y_test, y_pred))

# %%
