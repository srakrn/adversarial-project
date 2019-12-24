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
import torchvision.models as models

#%%
torch.manual_seed(0)

#%%
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

#%%
class PerturbsDataset(Dataset):
    data = torch.load("perturbs/on_single_point/resnet_on_single_point.pt").reshape(-1, 1, 28, 28)
    labels = np.load("models/classification_results/on_single_point/resnet_model/resnet_testset_perturbs.npy")
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
model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
model.fc.weight.requires_grad = True

#%%
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

#%%
epochs = 15
for e in range(epochs):
    print(e)
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()

        output = F.log_softmax(model(images), dim=1)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")

#%%
torch.save(model.state_dict(), "models/weak_learner/on_single_point/resnet/resnet_perturbs.model")

#%%
X_test, y_test = next(iter(testloader))

# %%
y_pred = model(X_test).argmax(dim=1).numpy()

# %%
print(classification_report(y_test, y_pred))

# %%
