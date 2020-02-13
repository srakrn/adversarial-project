import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch import nn, optim
from torch.utils.data import DataLoader


def calculate_k_perturbs(
    model, perturbs, training_set, k, n_epoches=20, verbose=False, log=False
):
    loader = DataLoader(training_set, batch_size=len(training_set), shuffle=False)
    X, y = next(iter(loader))
    km = KMeans(n_clusters=k)
    km_clusters = km.fit_predict(X.reshape(len(X), -1))
    print(f"Training {k} perturbs")

    k_points = []
    k_perturbs = []
    losses = []

    for i in set(km_clusters):
        if log:
            log_f = open(log, "a")
        idx = np.where(km_clusters == i)[0]
        data = [training_set[j] for j in idx]
        trainloader = DataLoader(data, batch_size=len(data), shuffle=False)

        perturb = torch.zeros([1, 28 * 28], requires_grad=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD([perturb], lr=0.03)

        if verbose:
            print(f"\tTraining # {i+1} perturb")
            print(f"\tThis set of perturbation will attack {len(data)} data points.")

        for e in range(n_epoches):
            running_loss = 0
            for images, labels in trainloader:
                images = images.reshape(-1, 28 * 28)
                optimizer.zero_grad()
                output = model(images + perturb)
                loss = -1 * criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                perturb.data.clamp(-1, 1)
        if verbose:
            print(f"\tTraining loss: {-1 * running_loss/len(trainloader)}")
        losses.append(-1 * running_loss / len(trainloader))
        k_points.append(idx)
        k_perturbs.append(perturb.detach())
    if log:
        t = time.strftime("%H:%M:%S", time.localtime())
        log_f.write(f"{t},{k},")
        log_f.write(",".join([f"{i:.5f}" for i in losses]))
        log_f.write("\n")
    return [k_points, k_perturbs, km]
