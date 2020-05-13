# %%
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from clustre.attacking.on_single_point import attack
from clustre.helpers.helpers import get_time

log = logging.getLogger(__name__)


def fgsm_reinforce(
    model,
    trainloader,
    n_epoches=10,
    epsilon=0.3,
    criterion=nn.CrossEntropyLoss,
    optimizer=optim.Adam,
    cuda=False,
):
    if cuda:
        model.to("cuda")
    else:
        model.to("cpu")
    log.info(f"Training started: {get_time()}")
    criterion = criterion()
    optimizer = optimizer(model.parameters())
    for e in range(n_epoches):
        running_loss = 0
        for i, (images, labels) in enumerate(trainloader):
            print(f"Epoch {e+1} Minibatch {i+1}")
            if cuda:
                images = images.to("cuda")
                labels = labels.to("cuda")

            perturbs = attack.fgsm_array(model, criterion, images, labels, cuda=cuda)
            if cuda and not perturbs.is_cuda:
                perturbs = perturbs.to("cuda")
            adver_images = images + epsilon * perturbs
            X = torch.cat([images, adver_images], 0)
            y = torch.cat([labels, labels], 0)
            if cuda:
                X = X.to("cuda")
                y = y.to("cuda")

            optimizer.zero_grad()

            output = F.log_softmax(model(X), dim=1)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            print(f"\tTraining loss: {running_loss/len(trainloader)}")
    log.info(f"Training ended: {get_time()}")
    return model
