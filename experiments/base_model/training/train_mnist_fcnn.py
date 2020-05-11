import torch
import torch.nn.functional as F
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

trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)

testloader = DataLoader(mnist_testset, batch_size=1, shuffle=True)


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


model = MnistFcnn()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 5
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

torch.save(model.state_dict(), "models/mnist_fcnn.model")
