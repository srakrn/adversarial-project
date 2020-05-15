import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from kmeans_pytorch import kmeans
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from clustre.attacking import fgsm, pgd
from clustre.helpers import get_time


def cluster_training(
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
    kmeans_parameters={"distance": "euclidian"},
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
            if log is not None and i % 10 == 9:
                log.info(f"\t\t{get_time()}: Minibatch {i+1}")
            # Move tensors to device if desired
            if device is not None:
                images = images.to(device)
                labels = labels.to(device)
            cluster_input = images.reshape(len(images), -1)
            cluster_idxs, cluster_centres, = kmeans(
                X=cluster_input, num_clusters=n_clusters, device=device,
            )
            cluster_centres = cluster_centres.to(device)
            kmeans_dist = torch.max(
                torch.norm(
                    cluster_centres[:, None, :].repeat(1, len(images), 1)
                    - cluster_input[None, :],
                    dim=2,
                ),
                dim=0,
            ).values
            kmeans_inverse_dist = 1 / (kmeans_dist + 1e-4)
            kmeans_prob = (
                (torch.exp(kmeans_inverse_dist) / sum(torch.exp(kmeans_inverse_dist)))
                .cpu()
                .detach()
                .numpy()
            )
            # Downsample the k-meaned images (according to ratio)
            idx = np.random.choice(
                np.arange(len(kmeans_prob)),
                size=int(len(kmeans_prob) * sample_ratio),
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
