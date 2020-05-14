import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from clustre.attacking import fgsm, pgd
from clustre.helpers import get_time


def fgsm_training(
    model,
    trainloader,
    n_epoches=10,
    n_clusters=8,
    sample_ratio=0.5,
    epsilon=0.3,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.Adam,
    optimizer_params={},
    fgsm_parameters={},
    kmeans_parameters={},
    pgd_parameters={},
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

    # Iterate over e times of epoches
    for e in range(n_epoches):
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
            # Calculate perturbations
            fgsm_adver_images = fgsm(
                model,
                criterion,
                images,
                labels,
                epsilon,
                device=device,
                **fgsm_parameters,
            )
            # Cluster adversarial images
            np_adver_images = fgsm_adver_images.cpu().detach().numpy()
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans_dist = kmeans.fit_transform(np_adver_images, **kmeans_parameters)
            kmeans_inverse_dist = 1 / (kmeans_dist + 1e-3)
            kmeans_prob = np.exp(kmeans_inverse_dist) / sum(np.exp(kmeans_inverse_dist))
            # Downsample the k-meaned images (according to ratio)
            idx = np.random.choice(
                np.arange(len(trainloader)),
                size=int(len(trainloader) * sample_ratio),
                p=kmeans_prob,
            )
            images_subset, labels_subset = images[idx], labels[idx]
            pgd_adver_images_subset = pgd(
                model, criterion, images_subset, labels_subset, **pgd_parameters
            )
            optimizer.zero_grad()

            output = model(pgd_adver_images_subset)
            loss = criterion(output, labels_subset)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            if log is not None:
                log.info(f"\tTraining loss: {running_loss/len(trainloader)}")
    if log is not None:
        log.info(f"Training ended: {get_time()}")
    return model
