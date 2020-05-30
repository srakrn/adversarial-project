import torch.nn.functional as F
import torchvision.models as models
from torch import nn


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


class CifarCnn(nn.Module):
    def __init__(self):
        super(CifarCnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


mnist_fcnn = MnistFcnn()

mnist_cnn = MnistCnn()

mnist_resnet50 = models.resnet50(pretrained=True)
mnist_resnet50.conv1 = nn.Conv2d(
    1, 64, kernel_size=7, stride=2, padding=3, bias=False
)
mnist_resnet50.fc = nn.Linear(in_features=2048, out_features=10, bias=True)

mnist_wideresnet34_10 = wideresnet34_10()
mnist_wideresnet34_10.conv1 = nn.Conv2d(
    1, 64, kernel_size=7, stride=2, padding=3, bias=False
)
mnist_wideresnet34_10.fc = nn.Linear(
    in_features=2048, out_features=10, bias=True
)


cifar10_cnn = CifarCnn()

cifar10_resnet50 = models.resnet50(pretrained=True)
cifar10_resnet50.fc = nn.Linear(in_features=2048, out_features=10, bias=True)

cifar10_wideresnet34_10 = wideresnet34_10()
cifar10_wideresnet34_10.fc = nn.Linear(
    in_features=2048, out_features=10, bias=True
)
