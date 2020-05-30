import torch
from clustre.helpers.datasets import cifar10_testloader, cifar10_trainloader
from clustre.models import cifar10_cnn, cifar10_wide_resnet34_10
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

N_EPOCHES = 40
LR = 1e-3

torch.manual_seed(0)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

models = {
    # "cifar10_cnn": cifar10_cnn,
    # "cifar10_resnet34": cifar10_resnet34,
    "cifar10_wide_resnet34_10": cifar10_wide_resnet34_10
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
        for images, labels in cifar10_trainloader:
            images = images.to("cuda")
            labels = labels.to("cuda")
            images = images.reshape(-1, 3, 32, 32)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        training_loss /= len(cifar10_trainloader)

        with torch.no_grad():
            for images, labels in cifar10_testloader:
                images = images.to("cuda")
                labels = labels.to("cuda")
                images = images.reshape(-1, 3, 32, 32)
                output = model(images)
                testing_loss += loss.item()
            testing_loss /= len(cifar10_testloader)
            testing_losses.append(testing_loss)
        print(f"Epoch: {e}\n\tTrain: {training_loss} Test: {testing_loss}")

        if testing_loss <= min(testing_losses):
            torch.save(model.state_dict(), "{}.model".format(model_name))
