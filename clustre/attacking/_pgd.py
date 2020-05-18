from math import inf

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def pgd(
    model,
    criterion,
    images,
    labels,
    epsilon=0.3,
    step_size=0.02,
    n_epoches=100,
    verbose=True,
    device=None,
):
    model.eval()

    if device is not None:
        model = model.to(device)
        images = images.to(device)
        labels = labels.to(device)

    original_images = images
    perturbs = pgd_perturbs(
        model,
        criterion,
        images,
        labels,
        epsilon,
        step_size,
        n_epoches,
        verbose,
        device,
    )
    return torch.clamp(original_images + perturbs, min=-1, max=1)


def pgd_perturbs(
    model,
    criterion,
    images,
    labels,
    epsilon=0.3,
    step_size=0.02,
    n_epoches=100,
    verbose=True,
    device=None,
):
    model.eval()

    if device is not None:
        model = model.to(device)
        images = images.to(device)
        labels = labels.to(device)

    original_images = images

    for e in range(n_epoches + 1):
        images = images.detach()
        images.requires_grad = True

        output = model(images)

        model.zero_grad()
        loss = criterion(output, labels)
        loss.backward()

        if e == n_epoches:
            return epsilon * torch.sign(images.grad)

        images = images + (step_size * torch.sign(images.grad))
        images = torch.max(
            torch.min(images, original_images + epsilon),
            original_images - epsilon,
        )
