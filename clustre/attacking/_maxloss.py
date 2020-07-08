import torch
from torch import optim


def maxloss(
    model,
    criterion,
    images,
    labels,
    epsilon=0.3,
    optim=optim.SGD,
    optim_params={"lr": 0.03},
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
    perturbs = maxloss_perturbs(
        model,
        criterion,
        images,
        labels,
        epsilon,
        optim,
        optim_params,
        n_epoches,
        verbose,
        device,
    )
    return torch.clamp(original_images + perturbs, min=-1, max=1)


def maxloss_perturbs(
    model,
    criterion,
    images,
    labels,
    epsilon=0.3,
    optim=optim.SGD,
    optim_params={"lr": 0.03},
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

    perturbs = torch.rand(images.shape, device=images.device) / 1000
    optimizer = optim([perturbs], **optim_params)

    for e in range(n_epoches + 1):
        images = images.detach()
        images.requires_grad = True

        optimizer.zero_grad()
        output = model(images)
        loss = -1 * criterion(output, labels)
        loss.backward()
        optimizer.step()

        if e == n_epoches:
            return torch.clamp(perturbs, -epsilon, epsilon)

        images = images + perturbs
        images = torch.max(
            torch.min(images, original_images + epsilon),
            original_images - epsilon,
        )
