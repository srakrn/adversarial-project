# %%
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
y_test = []
y_pred = []
for image, label in testloader:
    y_test.append(label.item())
    y_pred.append(model(image).argmax(axis=1).item())

print(classification_report(y_test, y_pred))


# %%
trainset_perturbs = torch.load(
    "perturbs/on_single_point/mnist/fcnn_fgsm_perturbs_trainset.pt"
)
testset_perturbs = torch.load(
    "perturbs/on_single_point/mnist/fcnn_fgsm_perturbs_testset.pt"
)


# %%
y_test = []
y_pred = []
for (image, label), perturb in zip(testloader, trainset_perturbs):
    y_test.append(label.item())
    y_pred.append(
        model(image + 0.2 * perturb.reshape(1, 1, 28, 28)).argmax(axis=1).item()
    )

print(classification_report(y_test, y_pred))

# %%
plt.imshow(
    (image + 0.2 * perturb.reshape(1, 1, 28, 28)).reshape(28, 28).detach().numpy()
)
plt.show()

# %%
def calculate_k_perturbs(
    model, training_set, clusterer, k, n_epoches=20, verbose=0, log=False
):
    loader = DataLoader(training_set, batch_size=len(training_set), shuffle=False)
    X, y = next(iter(loader))
    km = KMeans(n_clusters=k, verbose=verbose, n_init=1)
    km_clusters = km.fit_predict(clusterer.reshape(len(clusterer), -1))
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
        optimizer = optim.Adagrad([perturb], lr=0.03)

        if verbose:
            print(f"Training #{i+1} perturb")
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
train_target, train_perturb, train_km = calculate_k_perturbs(
    model, mnist_trainset, trainset_perturbs.detach().numpy(), 100, verbose=2
)


# %%
test_target, test_perturb, test_km = calculate_k_perturbs(
    model, mnist_testset, testset_perturbs.detach().numpy(), 100, verbose=2
)


# %%
fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for ax, perturb in zip(axs.ravel(), train_perturb):
    ax.imshow(perturb.reshape(28, 28))
plt.show()


# %%
fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for ax, perturb in zip(axs.ravel(), test_perturb):
    ax.imshow(perturb.reshape(28, 28))
plt.show()


# %%
def get_nth_perturb(n, targets, perturbs):
    for i, j in zip(targets, perturbs):
        if i in targets:
            return j
    return None


# %%
y_test = []
y_pred = []
for i, (image, label) in enumerate(testloader):
    y_test.append(label.item())
    perturb = torch.tensor(get_nth_perturb(i, test_target, test_perturb))
    y_pred.append(
        model(image + 0.2 * perturb.reshape(1, 1, 28, 28)).argmax(axis=1).item()
    )

print(classification_report(y_test, y_pred))


# %%
class PerturbDataset(Dataset):
    def __init__(self, data, targets, perturbs, density=0.2):
        super().__init__()
        self.data = data
        self.targets = targets
        self.perturbs = perturbs
        self.density = density

    def __len__(self):
        return len(self.data)

    def _get_nth_perturb(self, n):
        for i, j in zip(self.targets, self.perturbs):
            if i in self.targets:
                return j
        return None

    def __getitem__(self, idx):
        X, y = self.data[idx]
        X += 0.2 * perturb.reshape(1, 28, 28)
        return X, y


# %%
pd = PerturbDataset(mnist_trainset, train_target, train_perturb)


# %%
perturbloader = DataLoader(pd, batch_size=16, shuffle=True)


# %%
criterion = nn.CrossEntropyLoss(reduction="none")
optimizer = optim.Adam(model.parameters())
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

        output = F.log_softmax(model(X), dim=1)
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
