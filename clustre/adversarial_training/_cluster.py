from datetime import datetime

import torch
from sklearn.cluster import KMeans
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from clustre.attacking import pgd_perturbs
from clustre.helpers import delta_time_string, format_time, get_time


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
        random_sample=False,
        kmeans_parameters={},
        transform=None,
    ):
        # Initialise things
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.criterion = criterion
        self.transform = transform

        if random_sample:
            kmeans_parameters["max_iter"] = 1
        # Create a k-Means instance and fit
        d = self.dataset.data.reshape(len(dataset), -1)
        self.km = KMeans(n_clusters=n_clusters, **kmeans_parameters)
        self.km.fit(d)
        # Obtain targets and ids of each cluster centres
        self.cluster_ids = self.km.predict(d)
        self.cluster_centers_idx = self.km.transform(d).argmin(axis=0)

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
    random_sample=False,
    epsilon=0.3,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optim.Adam,
    optimizer_params={},
    kmeans_parameters={"n_init": 3},
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
        random_sample=random_sample,
        kmeans_parameters=kmeans_parameters,
        transform=trainloader.dataset.transform,
    )
    adversarialloader = DataLoader(adversarial_dataset, batch_size=128)
    if log is not None:
        kmeans_end = datetime.now()
        log.info(f"k-Means ended: {get_time()}")
        log.info(
            f"k-Means time used: {delta_time_string(kmeans_end, kmeans_start)}"
        )

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
            log.info(f"\tEpoch {e+1}")
        # Log epoches
        if log is not None:
            log.info(f"\t\tThis epoch starts at {get_time()}")
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
            log.info(
                f"\t\tTime used for PGD generation: {delta_time_string(pgd_end, pgd_start)}"
            )
            back_start = datetime.now()
        # Running loss, for reference
        running_loss = 0
        # Iterate over minibatches of trainloader
        for i, (images, labels, cluster_idx) in enumerate(adversarialloader):
            # Move tensors to device if desired
            if device is not None:
                images = images.to(device)
                labels = labels.to(device)
                cluster_perturbs = cluster_perturbs.to(device)
            optimizer.zero_grad()

            output = model(
                images
                + cluster_perturbs[cluster_idx.numpy()].reshape(images.shape)
            )
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            if log is not None:
                back_end = datetime.now()
                log.info(
                    f"\t\tTime used for backprop: {delta_time_string(back_end, back_start)}"
                )
                log.info(f"\t\tThis epoch terminates at {get_time()}")
                log.info(f"\tTraining loss: {running_loss/len(trainloader)}")
    if log is not None:
        log.info(f"Training ended: {get_time()}")
    return model
