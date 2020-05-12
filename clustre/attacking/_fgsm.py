import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def fgsm_single_image(model, criterion, image, label, epsilon=0.03, cuda=False):
    if len(image.shape) == 3:
        image.unsqueeze_(0)
        label.unsqueeze_(0)

    model.eval()

    if cuda:
        model.to("cuda")
        image = image.to("cuda")
        label = label.to("cuda")

    image.requires_grad = True

    output = model(image)
    loss = criterion(output, label)
    loss.backward()

    perturb = image.grad.data[0].sign().unsqueeze_(0) * epsilon
    attack_image = torch.clamp(image + perturb, min=-1, max=1)

    return attack_image


def fgsm(model, criterion, images, labels, epsilon=0.03, cuda=False):
    return torch.cat(
        [
            fgsm_single_image(model, criterion, i, j, epsilon, cuda)
            for (i, j) in zip(images, labels)
        ]
    )
