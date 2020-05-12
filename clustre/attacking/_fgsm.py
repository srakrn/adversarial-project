import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def fgsm(model, criterion, image, label, epsilon=0.03, cuda=False):
    if len(image.shape) == 3:
        image.unsqueeze_(0)
        label.unsqueeze_(0)

    model.eval()

    if cuda:
        model.to("cuda")
        image = image.to("cuda")
        label = label.to("cuda")
    else:
        model.to("cpu")
        image = image.to("cpu")
        label = label.to("cpu")

    image.requires_grad = True

    output = model(image)
    loss = criterion(output, label)
    loss.backward()

    perturb = image.grad.data.sign() * epsilon
    attack_image = torch.clamp(image + perturb, min=-1, max=1)

    return attack_image
