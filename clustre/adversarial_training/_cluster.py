import math
from datetime import datetime

import numpy as np
import numpy.linalg as la
from dateutil.relativedelta import relativedelta
from sklearn.cluster import KMeans

import torch
from clustre.attacking import fgsm, fgsm_perturbs, pgd, pgd_perturbs
from clustre.helpers import delta_time_string, delta_tostr, get_time
from libKMCUDA import kmeans_cuda
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


# %%
class KMeansWrapper:
    def __init__(self, X, n_clusters, n_init=3, method="kmcuda"):
        if method == "kmcuda":
            self.inertia = np.inf
            for _ in range(n_init):
                centers, y_pred = kmeans_cuda(X.astype(np.float32), n_clusters)
                full_idx = np.arange(len(X))
                centroids_idxs = []
                inertia = 0
                for i in range(n_clusters):
                    idx = full_idx[y_pred == i]
                    if len(idx) != 0:
                        X_sub = X[idx]
                        norm = la.norm(X_sub - centers[i], axis=1)
                        min_idx = norm.argmin()
                        centroids_idxs.append(idx[min_idx])
                        inertia += np.sum(norm)
                    else:
                        centroids_idxs.append(0)
                centroids_idxs = np.array(centroids_idxs)

                if inertia < self.inertia:
                    self.centers = centers
                    self.y_pred = y_pred
                    self.centroids_idxs = centroids_idxs
        elif method == "sklearn":
            km = KMeans(n_clusters, n_init=n_init)
            self.y_pred = km.fit_predict(X)
            self.centers = km.cluster_centers_
            self.centroids_idxs = km.transform(X).argmin(axis=0)
        else:
            raise NotImplementedError


def count_unique(keys):
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    return uniq_keys, np.bincount(bins)


class AdversarialDataset(Dataset):
    """
    Adversarial dataset to be feeded to the model
    """

    def __init__(
        self,
        model,
        dataset,
        criterion=nn.CrossEntropyLoss(),
        n_clusters=100,
        method="kmcuda",
        cluster_with="original_data",
        epsilon=0.3,
        n_init=3,
        transform=None,
        device="cuda",
    ):
        # Initialise things
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.criterion = criterion
        self.transform = transform

        # Create a k-Means instance and fit
        if cluster_with == "fgsm_perturb":
            dl = DataLoader(dataset, batch_size=64, shuffle=False)
            d = []
            for images, labels in iter(dl):
                y = (
                    fgsm_perturbs(
                        model,
                        criterion,
                        images,
                        labels,
                        epsilon=epsilon,
                        device=device,
                    )
                    .detach()
                    .cpu()
                    .reshape(len(images), -1)
                    .numpy()
                )
                d.append(y)
            d = np.concatenate(d)
        elif cluster_with == "fgsm_input":
            dl = DataLoader(dataset, batch_size=64, shuffle=False)
            d = []
            for images, labels in iter(dl):
                y = (
                    fgsm(model, criterion, images, labels, device=device)
                    .detach()
                    .cpu()
                    .reshape(len(images), -1)
                    .numpy()
                )
                d.append(y)
            d = np.concatenate(d)
        elif cluster_with == "pgd_perturb":
            dl = DataLoader(dataset, batch_size=64, shuffle=False)
            d = []
            for images, labels in iter(dl):
                y = (
                    pgd_perturbs(
                        model,
                        criterion,
                        images,
                        labels,
                        epsilon=epsilon,
                        device=device,
                    )
                    .detach()
                    .cpu()
                    .reshape(len(images), -1)
                    .numpy()
                )
                d.append(y)
            d = np.concatenate(d)
        elif cluster_with == "pgd_input":
            dl = DataLoader(dataset, batch_size=64, shuffle=False)
            d = []
            for images, labels in iter(dl):
                y = (
                    pgd(model, criterion, images, labels, device=device)
                    .detach()
                    .cpu()
                    .reshape(len(images), -1)
                    .numpy()
                )
                d.append(y)
            d = np.concatenate(d)
        elif cluster_with == "original_data":
            d = self.dataset.data.reshape(len(dataset), -1)
            if type(d) is not np.ndarray:
                d = d.detach().cpu().numpy()
        else:
            raise NotImplementedError
        self.km = KMeansWrapper(d, n_clusters, n_init, method)
        # Obtain targets and ids of each cluster centres
        self.cluster_ids = self.km.y_pred.astype(int)
        self.cluster_centers_idx = self.km.centroids_idxs.astype(int)

        # Extract only interested ones
        X = []
        y = []
        for i in self.cluster_centers_idx:
            x, u = self.dataset[i]
            X.append(x)
            y.append(u)

        # To be used in PGD
        self.centroids_X = torch.stack(X)
        self.centroids_y = torch.Tensor(
            y, device=self.centroids_X.device
        ).long()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        cluster_id = self.cluster_ids[idx]
        return image, target, cluster_id


def cluster_training(
    model,
    trainloader,
    n_epoches=10,
    n_clusters=100,
    method="kmcuda",
    cluster_with="fgsm",
    n_init=3,
    epsilon=0.3,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.Adam,
    optimizer_params={},
    pgd_parameters={"n_epoches": 7},
    device=None,
    log=None,
):
    if log is not None:
        log.info(f"k-Means started at: {get_time()}")
        kmeans_start = datetime.now()
    adversarial_dataset = AdversarialDataset(
        model,
        trainloader.dataset,
        criterion=criterion,
        n_clusters=n_clusters,
        method=method,
        cluster_with=cluster_with,
        epsilon=epsilon,
        n_init=n_init,
        transform=trainloader.dataset.transform,
        device=device,
    )
    adversarialloader = DataLoader(adversarial_dataset, batch_size=128)
    if log is not None:
        kmeans_end = datetime.now()
        kmeans_time = delta_time_string(kmeans_end, kmeans_start)

    # Move to device if desired
    if device is not None:
        model.to(device)
    # Log starting time if desired
    if log is not None:
        log.info(f"Training started: {get_time()}")

    # Create an optimiser instance
    optimizer = optimizer(model.parameters(), **optimizer_params)

    if device is not None:
        centroids_X = adversarial_dataset.centroids_X.to(device)
        centroids_y = adversarial_dataset.centroids_y.to(device)

    # Iterate over e times of epoches
    for e in range(n_epoches):
        # Log epoches
        if log is not None:
            pgd_start = datetime.now()
        # Generate PGD examples
        cluster_perturbs = pgd_perturbs(
            model,
            criterion,
            centroids_X,
            centroids_y,
            epsilon=epsilon,
            **pgd_parameters,
        )
        if log is not None:
            pgd_end = datetime.now()
            pgd_time = delta_time_string(pgd_end, pgd_start)
        # Running loss, for reference
        running_loss = 0

        # Store time to be calculated
        tensor_move_time = relativedelta()
        calc_input_time = relativedelta()
        input_time = relativedelta()
        backprop_time = relativedelta()
        # Iterate over minibatches of trainloader
        for i, (images, labels, cluster_idx) in enumerate(adversarialloader):
            tensor_move_timestamp = datetime.now()
            # Move tensors to device if desired
            if device is not None:
                images = images.to(device)
                labels = labels.to(device)
                cluster_perturbs = cluster_perturbs.to(device)
            optimizer.zero_grad()

            calc_input_timestamp = datetime.now()
            X_input = images + cluster_perturbs[cluster_idx.numpy()].reshape(
                images.shape
            )
            input_timestamp = datetime.now()
            output = model(X_input)
            backprop_timestamp = datetime.now()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            end_timestamp = datetime.now()

            running_loss += loss.item()

            tensor_move_time += relativedelta(
                calc_input_timestamp, tensor_move_timestamp
            )
            calc_input_time += relativedelta(
                input_timestamp, calc_input_timestamp
            )
            input_time += relativedelta(backprop_timestamp, input_timestamp)
            backprop_time += relativedelta(end_timestamp, backprop_timestamp)

        else:
            if log is not None:
                log.info(f"{delta_tostr(kmeans_time/n_epoches)}")
                log.info(
                    f"\t\tTensor move time: {delta_tostr(tensor_move_time)}"
                )
                log.info(
                    f"\t\tPerturb matching time: {delta_tostr(calc_input_time)}"
                )
                log.info(f"\t\tInput time: {delta_tostr(input_time)}")
                log.info(f"\t\tBackprop time: {delta_tostr(backprop_time)}")
                log.info(f"\t\tTraining loss: {running_loss/len(trainloader)}")
    if log is not None:
        log.info(f"Training ended: {get_time()}")
    return model
