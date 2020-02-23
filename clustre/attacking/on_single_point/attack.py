import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def pgd(
    model,
    criterion,
    loader,
    epsilon=0.3,
    lr=0.01,
    n_epoches=40,
    verbose=False,
    cuda=False,
):
    """Generate perturbations on the dataset when given a model and a criterion
    using a maximised loss method

    Parameters
    ----------
    model: nn.module
        A model to attack
    criterion: function
        A criterion function
    loader: DataLoader
        A DataLoader instance with batch_size=1
    epsilon: float
        Maximum value to clamp for the perturbation
    lr: float
        Learning rate for the perturbation optimizer
    n_epohes: int
        Epoches to maximise the loss
    verbose: bool
        Verbosity setting

    Returns
    -------
    torch.tensor
        A tensor containing perturbations with the same length of the
        received dataset.
    """
    perturbs = []

    for i, (image, label) in enumerate(loader):
        if verbose:
            print("Image:", i + 1)

        perturb = pgd_single_point(
            model,
            criterion,
            image,
            label,
            epsilon=epsilon,
            lr=lr,
            n_epoches=n_epoches,
            verbose=verbose,
            cuda=cuda,
        )
        perturbs.append(perturb)

    perturbs = torch.stack(perturbs)
    return perturbs.cpu()


def pgd_array(
    model,
    criterion,
    images,
    labels,
    epsilon=0.3,
    lr=0.01,
    n_epoches=40,
    verbose=False,
    cuda=False,
):
    """Generate perturbations on the dataset when given a model and a criterion
    using a maximised loss method

    Parameters
    ----------
    model: nn.module
        A model to attack
    criterion: function
        A criterion function
    images: torch.Tensor
        Images to attack
    labels: torch.Tensor
        Labels for the images
    epsilon: float
        Maximum value to clamp for the perturbation
    lr: float
        Learning rate for the perturbation optimizer
    n_epohes: int
        Epoches to maximise the loss
    verbose: bool
        Verbosity setting

    Returns
    -------
    torch.tensor
        A tensor containing perturbations with the same length of the
        received dataset.
    """
    perturbs = []

    for i, (image, label) in zip(images, labels):
        if verbose:
            print("Image:", i + 1)

        image.unsqeeze_(0)
        label = torch.tensor([label])

        perturb = pgd_single_point(
            model,
            criterion,
            image,
            label,
            epsilon=epsilon,
            lr=lr,
            n_epoches=n_epoches,
            verbose=verbose,
            cuda=cuda,
        )
        perturbs.append(perturb)

    perturbs = torch.stack(perturbs)
    return perturbs.cpu()


def pgd_single_point(
    model,
    criterion,
    images,
    labels,
    epsilon=0.3,
    lr=0.01,
    n_epoches=40,
    verbose=False,
    cuda=False,
):
    """Generate a perturbation attacking a group of images and labels

    Parameters
    ----------
    model: nn.module
        A model to attack
    criterion: function
        A criterion function
    images: torch.Tensor
        Images to attack
    labels: torch.Tensor
        Labels for the images
    epsilon: float
        Maximum value to clamp for the perturbation
    lr: float
        Learning rate for the perturbation optimizer
    n_epohes: int
        Epoches to maximise the loss
    verbose: bool
        Verbosity setting

    Returns
    -------
    torch.tensor
        A tensor containing perturbations, with the shape of `images.shape[1:]`
    """
    model.eval()

    if cuda:
        model.to("cuda")
        images = images.to("cuda")
        labels = labels.to("cuda")
    else:
        model.to("cpu")
        images = images.to("cpu")
        labels = labels.to("cpu")

    #  Create a random array of perturbation
    if cuda:
        perturb = torch.zeros(images.shape[1:], device="cuda", requires_grad=True)
    else:
        perturb = torch.zeros(images.shape[1:], requires_grad=True)

    images.requires_grad = True

    for e in range(n_epoches):
        output = model(torch.clamp(images + perturb, -1, 1))
        loss = criterion(output, labels)
        loss.backward()

        perturb = epsilon * images.grad.mean(dim=0).sign()
        perturb.data.clamp_(-epsilon, epsilon)
    return perturb.cpu() / epsilon


def fgsm(model, criterion, loader, verbose=False, cuda=False):
    """Generate perturbations on the dataset when given a model and a criterion

    Parameters
    ----------
    model: nn.module
        A model to attack
    criterion: function
        A criterion function
    loader: DataLoader
        A DataLoader instance with batch_size=1
    verbose: bool
        Verbosity setting

    Returns
    -------
    torch.tensor
        A tensor containing perturbations with the same length of the
        received dataset.
    """
    perturbs = []
    model.eval()

    if cuda:
        model.to("cuda")
    else:
        model.to("cpu")

    for i, (image, label) in enumerate(loader):
        if verbose:
            print(f"Image {i+1}")

        perturb = fgsm_single_point(model, criterion, image, label, cuda=cuda,)
        perturbs.append(perturb)

    perturbs = torch.stack(perturbs)
    return perturbs.cpu()


def fgsm_array(model, criterion, images, labels, verbose=False, cuda=False):
    """Generate perturbations on the dataset when given a model and a criterion

    Parameters
    ----------
    model: nn.module
        A model to attack
    criterion: function
        A criterion function
    images: torch.Tensor
        Images to attack
    labels: torch.Tensor
        Labels for the images
    verbose: bool
        Verbosity setting

    Returns
    -------
    torch.tensor
        A tensor containing perturbations with the same length of the
        received dataset.
    """
    perturbs = []

    for i, (image, label) in enumerate(zip(images, labels)):
        if verbose:
            print(f"Image {i+1}")

        image.unsqeeze_(0)
        label = torch.tensor([label])

        perturb = fgsm_single_point(model, criterion, image, label, cuda=cuda,)
        perturbs.append(perturb)

    perturbs = torch.stack(perturbs)
    return perturbs.cpu()


def fgsm_single_point(model, criterion, images, labels, cuda=False):
    """Generate a perturbation attacking a group of images and labels

    Parameters
    ----------
    model: nn.module
        A model to attack
    criterion: function
        A criterion function
    images: torch.Tensor
        Images to attack
    labels: torch.Tensor
        Labels for the images
    verbose: bool
        Verbosity setting

    Returns
    -------
    torch.tensor
        A tensor containing perturbations, with the shape of `images.shape[1:]`
    """
    model.eval()

    if cuda:
        model.to("cuda")
        images = images.to("cuda")
        labels = labels.to("cuda")

    images.requires_grad = True

    output = model(images)
    loss = criterion(output, labels)
    loss.backward()

    perturb = images.grad.data.mean(dim=0).sign()
    return perturb.cpu()
