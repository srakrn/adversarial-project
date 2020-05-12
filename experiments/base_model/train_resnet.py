import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from clustre.models import cifar10_cnn, cifar10_resnet, cifar10_wideresnet

N_EPOCHES = 10
LR = 1e-3

torch.manual_seed(0)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


trainset = datasets.CIFAR10(
    root="datasets/cifar10", train=True, download=True, transform=transform
)
testset = datasets.CIFAR10(
    root="datasets/cifar10", train=False, download=True, transform=transform
)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=True)

models = {
    "cifar10_cnn": cifar10_cnn,
    "cifar10_resnet": cifar10_resnet,
    "cifar10_wideresnet": cifar10_wideresnet,
}

for model_name, model in models.items():
    print("Model: {}".format(model_name))

    model = model.to("cuda")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    testing_losses = []

    for e in range(N_EPOCHES):
        training_loss = 0
        testing_loss = 0
        for images, labels in trainloader:
            images = images.to("cuda")
            labels = labels.to("cuda")
            images = images.reshape(-1, 3, 32, 32)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        training_loss /= len(trainloader)

        with torch.no_grad():
            for images, labels in testloader:
                images = images.to("cuda")
                labels = labels.to("cuda")
                images = images.reshape(-1, 3, 32, 32)
                output = model(images)
                testing_loss += loss.item()
            testing_loss /= len(testloader)
            testing_losses.append(testing_loss)
        print(f"Epoch: {e}\n\tTrain: {training_loss} Test: {testing_loss}")

        if testing_loss <= min(testing_losses):
            torch.save(model.state_dict(), "results/models/{}.model".format(model_name))
