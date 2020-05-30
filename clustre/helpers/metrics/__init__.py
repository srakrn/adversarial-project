import torch
from sklearn.metrics import classification_report as cf
from torch import nn

from clustre.attacking import fgsm, pgd


def classification_report(model, testloader, device=None):
    if device is not None:
        model.to(device)
    y_true = []
    y_pred = []
    for (images, labels) in testloader:
        if device is not None:
            images = images.to(device)
            labels = labels.to(device)
        y_true.append(labels)
        y_pred.append(model(images))
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    y_pred = y_pred.argmax(dim=1)

    return cf(y_true.cpu().numpy(), y_pred.cpu().numpy())


def classification_report_fgsm(model, testloader, device=None, fgsm_params={}):
    if device is not None:
        model.to(device)
    y_true = []
    y_pred = []
    for (images, labels) in testloader:
        if device is not None:
            images = images.to(device)
            labels = labels.to(device)
        attacked_images = fgsm(
            model,
            nn.CrossEntropyLoss(),
            images,
            labels,
            device=device,
            **fgsm_params
        )
        y_true.append(labels)
        y_pred.append(model(attacked_images))
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    y_pred = y_pred.argmax(dim=1)

    return cf(y_true.cpu().numpy(), y_pred.cpu().numpy())


def classification_report_pgd(model, testloader, device=None, pgd_params={}):
    if device is not None:
        model.to(device)
    y_true = []
    y_pred = []
    for (images, labels) in testloader:
        if device is not None:
            images = images.to(device)
            labels = labels.to(device)
        attacked_images = pgd(
            model,
            nn.CrossEntropyLoss(),
            images,
            labels,
            device=device,
            **pgd_params
        )
        y_true.append(labels)
        y_pred.append(model(attacked_images))
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    y_pred = y_pred.argmax(dim=1)

    return cf(y_true.cpu().numpy(), y_pred.cpu().numpy())
