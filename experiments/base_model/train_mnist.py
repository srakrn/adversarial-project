import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from clustre.helpers.datasets import mnist_testloader, mnist_trainloader
from clustre.models import (
    mnist_cnn,
    mnist_fcnn,
    mnist_resnet50,
    mnist_wideresnet34_10,
)

N_EPOCHES = 100
LR = 1e-3

torch.manual_seed(0)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

models = {
    # "mnist_fcnn": mnist_fcnn,
    # "mnist_cnn": mnist_cnn,
    # "mnist_resnet50": mnist_resnet50,
    "mnist_wideresnet34_10": mnist_wideresnet34_10
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
        for images, labels in mnist_trainloader:
            images = images.to("cuda")
            labels = labels.to("cuda")
            images = images.reshape(-1, 1, 28, 28)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        training_loss /= len(mnist_trainloader)

        with torch.no_grad():
            for images, labels in mnist_testloader:
                images = images.to("cuda")
                labels = labels.to("cuda")
                images = images.reshape(-1, 1, 28, 28)
                output = model(images)
                testing_loss += loss.item()
            testing_loss /= len(mnist_testloader)
            testing_losses.append(testing_loss)
        print(f"Epoch: {e}\n\tTrain: {training_loss} Test: {testing_loss}")

        if testing_loss <= min(testing_losses):
            torch.save(model.state_dict(), "{}.model".format(model_name))
