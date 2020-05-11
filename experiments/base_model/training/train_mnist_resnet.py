import torch
import torchvision.models as models
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(0)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

mnist_trainset = datasets.MNIST(
    root="datasets/mnist", train=True, download=True, transform=transform
)

mnist_testset = datasets.MNIST(
    root="datasets/mnist", train=False, download=True, transform=transform
)

trainloader = DataLoader(mnist_trainset, batch_size=100, shuffle=True)
testloader = DataLoader(mnist_testset, batch_size=100, shuffle=True)

model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
for param in model.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 10
testing_losses = []
model = model.to("cuda")
for e in range(epochs):
    training_loss = 0
    testing_loss = 0
    for images, labels in trainloader:
        images = images.to("cuda")
        labels = labels.to("cuda")
        images = images.reshape(-1, 1, 28, 28)
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
            images = images.reshape(-1, 1, 28, 28)
            output = model(images)
            testing_loss += loss.item()
        testing_loss /= len(testloader)
        testing_losses.append(testing_loss)
    print(f"Epoch: {e}\n\tTrain: {training_loss} Test: {testing_loss}")

    if testing_loss <= min(testing_losses):
        torch.save(model.state_dict(), "results/models/mnist_resnet18.model")
