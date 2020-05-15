import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from clustre.attacking import pgd
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
    """Adversarial Training For Free

    Parameters
    ----------
    model: torch.nn.model
        The model to be reinforced
    trainloader: torch.utils.data.DataLoader
        The DataLoader for the training set
    n_epoches: int
        The epoches to be trained
    epsilon: float
        Perturbation bound
    hop_step: int
        Hop step
    criterion: function
        Criterion function
    optimizer: class of torch.optim
        Optimiser to train the model
    optimizer_params: dict
        Parameters to be passed to the optimiser
    device: torch.device, str, or None
        Device to be used
    log: logger or None
        If logger, logs to the corresponding logger
    """

    # Move to device if desired
    if device is not None:
        model.to(device)
    # Log starting time if desired
    if log is not None:
        log.info(f"Training started: {get_time()}")

    # Create an optimiser instance
    optimizer = optimizer(model.parameters(), **optimizer_params)

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
            images.requires_grad = True

            # Initialise attacking images
            delta = torch.zeros(images.shape, requires_grad=True, device=images.device)

            # Replay each minibatch for `hop_steps` to simulate PGD
            for i in range(hop_step):
                # Initialise attacking images
                attack_images = images + delta

                # Update model's parameter
                optimizer.zero_grad()
                output = model(attack_images)
                loss = criterion(output, labels)

                # Backprop
                loss.backward()
                optimizer.step()

                # Use gradients calculated for the minimisation step to update delta
                perturbs = epsilon * delta.grad.data.sign()
                delta = torch.clamp(delta, min=-epsilon, max=epsilon).detach()
                delta.requires_grad = True

            running_loss += loss.item()
        else:
            if log is not None:
                log.info(f"\tTraining loss: {running_loss/len(trainloader)}")
    if log is not None:
        log.info(f"Training ended: {get_time()}")
    return model
