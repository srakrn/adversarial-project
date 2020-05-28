import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def fgsm(model, criterion, image, label, epsilon=0.03, device=None):
    if device is not None:
        model.to(device)
        image = image.to(device)
        label = label.to(device)

    perturb = fgsm_perturbs(
        model, criterion, image, label, epsilon=epsilon, device=device
    )
    attack_image = torch.clamp(image + perturb, min=-1, max=1)

    return attack_image


def fgsm_perturbs(model, criterion, image, label, epsilon=0.03, device=None):
    if len(image.shape) == 3:
        image.unsqueeze_(0)
        label.unsqueeze_(0)

    model.eval()

    if device is not None:
        model.to(device)
        image = image.to(device)
        label = label.to(device)

    image.requires_grad = True

    output = model(image)
    loss = criterion(output, label)
    loss.backward()

    perturb = image.grad.data.sign() * epsilon
    return perturb
