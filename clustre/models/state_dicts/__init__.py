import os.path
import pathlib

import torch

current_path = pathlib.Path(__file__).parent.absolute()


mnist_cnn_state = torch.load(os.path.join(current_path, "mnist_cnn.model"))
mnist_resnet50_state = torch.load(
    os.path.join(current_path, "mnist_resnet50.model")
)
cifar10_cnn_state = torch.load(os.path.join(current_path, "cifar10_cnn.model"))
cifar10_wide_resnet34_10_state = torch.load(
    os.path.join(current_path, "cifar10_wide_resnet34_10.model")
)
