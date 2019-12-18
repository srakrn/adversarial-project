#%%
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

#%%
torch.manual_seed(0)

#%%
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

#%%
class PerturbsDataset(Dataset):
    data = torch.load("perturbs/on_single_point/cnn_on_single_point.pt").reshape(-1, 1, 28, 28)
    labels = np.load("models/classification_results/on_single_point/cnn_model/cnn_testset_perturbs.npy")
    labels = torch.tensor(labels)

    def __init__(self):
        pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return [self.data[idx], self.labels[idx]]

#%%
perturbs_dataset = PerturbsDataset()
trainloader = DataLoader(perturbs_dataset, batch_size=64, shuffle=True)

mnist_testset = datasets.MNIST(
    root="mnist", train=False, download=True, transform=transform
)
testloader = DataLoader(mnist_testset, batch_size=10000, shuffle=False)
#%%
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


net = MnistCnn()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

#%%
epochs = 15
for e in range(epochs):
    print(e)
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()

        output = F.log_softmax(net(images), dim=1)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

#%%
torch.save(net.state_dict(), "models/weak_learner/on_single_point/mnist_cnn.model")

#%%
X_test, y_test = next(iter(testloader))

# %%
y_pred = net(X_test).argmax(dim=1).numpy()

# %%
print(classification_report(y_test, y_pred))

# %%
