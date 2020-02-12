import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, utils

torch.manual_seed(0)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = datasets.CIFAR10(
    root="cifar10", train=True, download=True, transform=transform
)
testset = datasets.CIFAR10(
    root="cifar10", train=False, download=True, transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2
)

model = models.wide_resnet50_2(pretrained=True)
model.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
for param in model.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
writer = SummaryWriter("tensorboard/cifar10_wideresnet")


epochs = 100
testing_losses = []
model = model.to("cuda")
for e in range(epochs):
    training_loss = 0
    testing_loss = 0
    y_train = []
    y_test = []
    y_train_pred = []
    y_test_pred = []
    for images, labels in trainloader:
        img_grid = utils.make_grid(images)
        writer.add_image("CIFAR-10", img_grid)
        y_train += labels.tolist()
        images = images.to("cuda")
        labels = labels.to("cuda")
        optimizer.zero_grad()
        output = model(images)
        y_train_pred += output.argmax(axis=1).tolist()
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
    training_loss /= len(trainloader)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    writer.add_scalar("Loss/Train", training_loss, e + 1)
    writer.add_scalar("Accuracy/Train", train_accuracy, e + 1)

    with torch.no_grad():
        for images, labels in testloader:
            y_test += labels.tolist()
            images = images.to("cuda")
            labels = labels.to("cuda")
            output = model(images)
            y_test_pred += output.argmax(axis=1).tolist()
            testing_loss += loss.item()
        testing_loss /= len(testloader)
        testing_losses.append(testing_loss)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        writer.add_scalar("Loss/Test", testing_loss, e + 1)
        writer.add_scalar("Accuracy/Test", test_accuracy, e + 1)

    print(f"Epoch {e + 1}")
    print(f"Train loss: {training_loss}, accuracy: {train_accuracy}")
    print(f"Testing loss: {testing_loss}, accuracy: {test_accuracy}")

    torch.save(model.state_dict(), f"models/cifar10_wideresnet18_{e + 1}.model")
