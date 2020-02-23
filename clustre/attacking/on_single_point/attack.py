import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def pgd(
    model, criterion, loader, epsilon=1, lr=0.1, n_epoches=10, verbose=False, cuda=False
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
    model.eval()

    if cuda:
        model = model.to("cuda")
    else:
        model = model.to("cpu")

    for i, (image, label) in enumerate(loader):
        if verbose:
            print("Image:", i + 1)

        if cuda:
            image = image.to("cuda")
            label = label.to("cuda")
        else:
            image = image.to("cpu")
            label = label.to("cpu")

        #  Create a random array of perturbation
        if cuda:
            perturb = torch.zeros(image.shape, device="cuda", requires_grad=True)
        else:
            perturb = torch.zeros(image.shape, requires_grad=True)

        optimizer = optim.Adam([perturb], lr=lr)

        for e in range(n_epoches):
            running_loss = 0
            optimizer.zero_grad()

            output = model(image + perturb)
            loss = -criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            perturb.data.clamp_(-epsilon, epsilon)
        if verbose:
            print("\tNoise loss:", -1 * loss.item())
        perturbs.append(perturb)

    if cuda:
        model.to("cpu")
    perturbs = torch.cat(perturbs)
    return perturbs.cpu()


def pgd_array(
    model,
    criterion,
    images,
    labels,
    epsilon=1,
    lr=0.1,
    n_epoches=10,
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
    model.eval()

    if cuda:
        model = model.to("cuda")
    else:
        model = model.to("cpu")

    for i, (image, label) in zip(images, labels):
        if verbose:
            print("Image:", i + 1)

        image.unsqeeze_(0)
        label = torch.tensor([label])
        
        if cuda:
            image = image.to("cuda")
            label = label.to("cuda")
        else:
            image = image.to("cpu")
            label = label.to("cpu")

        #  Create a random array of perturbation
        if cuda:
            perturb = torch.zeros(image.shape, device="cuda", requires_grad=True)
        else:
            perturb = torch.zeros(image.shape, requires_grad=True)

        optimizer = optim.Adam([perturb], lr=lr)

        for e in range(n_epoches):
            running_loss = 0
            optimizer.zero_grad()

            output = model(image + perturb)
            loss = -criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            perturb.data.clamp_(-epsilon, epsilon)
        if verbose:
            print("\tNoise loss:", -1 * loss.item())
        perturbs.append(perturb)

    if cuda:
        model.to("cpu")
    perturbs = torch.cat(perturbs)
    return perturbs.cpu()


def pgd_single_point(
    model,
    criterion,
    images,
    labels,
    epsilon=1,
    lr=0.1,
    n_epoches=10,
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

    # Using Adam optimiser
    optimizer = optim.Adam([perturb], lr=lr)

    for e in range(n_epoches):
        running_loss = 0
        optimizer.zero_grad()

        output = model(images + perturb)
        loss = -criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        perturb.data.clamp_(-epsilon, epsilon)
        if verbose:
            print("\tNoise loss:", -1 * running_loss)
    return perturb.cpu()


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
        if cuda:
            image = image.to("cuda")
            label = label.to("cuda")
        else:
            image = image.to("cpu")
            label = label.to("cpu")
        if verbose:
            print("Image:", i + 1)

        #  Epsilon defines the maximum density (-e, e). It should be
        #  in the range of the training set's scaled value.
        epsilon = 1

        image.requires_grad = True

        output = model(image)
        loss = criterion(output, label)
        loss.backward()

        perturb = image.grad.data.sign()
        perturbs.append(perturb)

    perturbs = torch.cat(perturbs)

    if cuda:
        model.to("cpu")
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
    model.eval()

    if cuda:
        model.to("cuda")
    else:
        model.to("cpu")

    for i, (image, label) in enumerate(zip(images, labels)):
        if cuda:
            image = image.to("cuda")
            label = label.to("cuda")
        else:
            image = image.to("cpu")
            label = label.to("cpu")

        image.unsqueeze_(0)
        label.unsqueeze_(0)

        if verbose:
            print("Image:", i + 1)

        image.requires_grad = True

        output = model(image)
        loss = criterion(output, label)
        loss.backward()

        perturb = image.grad.data.sign()
        perturbs.append(perturb)
    return torch.cat(perturbs).cpu()


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
