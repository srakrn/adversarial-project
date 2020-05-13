import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from clustre.attacking import pgd
from clustre.helpers.helpers import get_time


def pgd_reinforce(
    model,
    trainloader,
    n_epoches=10,
    epsilon=0.3,
    pgd_step_size=0.02,
    pgd_epoches=100,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.Adam,
    optimizer_params={},
    device=None,
    log=None,
):
    """Standard k-PGD Adversarial Training

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
    pgd_step_size: float
        PGD step size
    pgd_epoches: int
        PGD maximization iterations per minimisation step
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
    for e in range(n_epoches):
        # Running loss, for reference
        running_loss = 0
        # Iterate over minibatches of trainloader
        for i, (images, labels) in enumerate(trainloader):
            # Log epoches
            if log is not None:
                log.info(f"\t{get_time()}: Epoch {e+1} Minibatch {i+1}")
            # Move tensors to device if desired
            if device is not None:
                images = images.to(device)
                labels = labels.to(device)
            # Calculate perturbations
            adver_images = pgd(
                model,
                criterion,
                images,
                labels,
                epsilon,
                pgd_step_size,
                pgd_epoches,
                device=device,
            )
            optimizer.zero_grad()

            output = model(adver_images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            print(f"\tTraining loss: {running_loss/len(trainloader)}")
    log.info(f"Training ended: {get_time()}")
    return model
