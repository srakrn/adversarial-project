import os.path
import pathlib

import torch

current_path = pathlib.Path(__file__).parent.absolute()


mnist_fcnn_state = torch.load(os.path.join(current_path, "mnist_fcnn.model"))
mnist_cnn_state = torch.load(os.path.join(current_path, "mnist_cnn.model"))
mnist_resnet_state = torch.load(os.path.join(current_path, "mnist_resnet.model"))
cifar10_cnn_state = torch.load(os.path.join(current_path, "cifar10_cnn.model"))
cifar10_resnet_state = torch.load(os.path.join(current_path, "cifar10_resnet.model"))
cifar10_wideresnet_state = torch.load(
    os.path.join(current_path, "cifar10_wideresnet.model")
)
