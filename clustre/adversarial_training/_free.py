import math

import torch
from torch import nn, optim

from clustre.helpers import get_time


def free_training(
    model,
    trainloader,
    n_epoches=10,
    epsilon=0.3,
    hop_step=5,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.Adam,
    optimizer_params={},
    device=None,
    log=None,
):
    # Move to device if desired
    if device is not None:
        model.to(device)
    # Log starting time if desired
    if log is not None:
        log.info(f"Training started: {get_time()}")

    # Create an optimiser instance
    optimizer = optimizer(model.parameters(), **optimizer_params)
    delta = None

    # Iterate over e times of epoches
    for e in range(math.ceil(n_epoches / hop_step)):
        # Running loss, for reference
        running_loss = 0
        # Log epoches
        if log is not None:
            log.info(f"\t{get_time()}: Epoch {e+1}")
        # Iterate over minibatches of trainloader
        for i, (images, labels) in enumerate(trainloader):
            # Move tensors to device if desired
            if device is not None:
                images = images.to(device)
                labels = labels.to(device)

            # Initialise attacking images
            if delta is None:
                delta = torch.zeros(
                    images.shape, requires_grad=True, device=images.device
                )

            # Replay each minibatch for `hop_steps` to simulate PGD
            for i in range(hop_step):
                # Initialise attacking images
                attack_images = (images + delta).detach().requires_grad_()

                # Update model's parameter
                optimizer.zero_grad()
                output = model(attack_images)
                loss = criterion(output, labels)

                # Backprop
                loss.backward()
                optimizer.step()

                # Use gradients calculated for the minimisation step
                # to update delta
                grad = attack_images.grad.data.detach()
                delta = delta.detach() + epsilon * torch.sign(grad)
                delta.clamp_(min=-epsilon, max=epsilon)

            running_loss += loss.item()
        else:
            if log is not None:
                log.info(f"\tTraining loss: {running_loss/len(trainloader)}")
    if log is not None:
        log.info(f"Training ended: {get_time()}")

    return model
