# %%
import numpy as np
import numpy.linalg as la
import seaborn as sns

import torch
from clustre.attacking import fgsm_perturbs, maxloss_perturbs, pgd_perturbs
from clustre.helpers.datasets import mnist_trainloader
from clustre.models import mnist_cnn, mnist_resnet18
from clustre.models.state_dicts import mnist_cnn_state, mnist_resnet18_state
from torch import nn

sns.set()


# %%
mnist_cnn.load_state_dict(mnist_cnn_state)
mnist_resnet18.load_state_dict(mnist_resnet18_state)

mnist_cnn.to("cuda")
mnist_resnet18.to("cuda")

# %%
criterion = nn.CrossEntropyLoss()

# %%
cnn_fgsm = []
for X, y in iter(mnist_trainloader):
    X = X.to("cuda")
    y = y.to("cuda")
    p = fgsm_perturbs(mnist_cnn, criterion, X, y)
    cnn_fgsm.append(p)

# %%
cnn_pgd = []
for X, y in iter(mnist_trainloader):
    X = X.to("cuda")
    y = y.to("cuda")
    p = pgd_perturbs(mnist_cnn, criterion, X, y)
    cnn_pgd.append(p)

# %%
cnn_maxloss = []
for X, y in iter(mnist_trainloader):
    X = X.to("cuda")
    y = y.to("cuda")
    p = maxloss_perturbs(mnist_cnn, criterion, X, y)
    cnn_maxloss.append(p)

# %%
resnet18_fgsm = []
for X, y in iter(mnist_trainloader):
    X = X.to("cuda")
    y = y.to("cuda")
    p = fgsm_perturbs(mnist_resnet18, criterion, X, y)
    resnet18_fgsm.append(p)

# %%
resnet18_pgd = []
for X, y in iter(mnist_trainloader):
    X = X.to("cuda")
    y = y.to("cuda")
    p = pgd_perturbs(mnist_resnet18, criterion, X, y)
    resnet18_pgd.append(p)

# %%
resnet18_maxloss = []
for X, y in iter(mnist_trainloader):
    X = X.to("cuda")
    y = y.to("cuda")
    p = maxloss_perturbs(mnist_resnet18, criterion, X, y)
    resnet18_maxloss.append(p)

# %%
cnn_fgsm = torch.cat(cnn_fgsm)
cnn_maxloss = torch.cat(cnn_maxloss)
cnn_pgd = torch.cat(cnn_pgd)

# %%
resnet18_fgsm = torch.cat(resnet18_fgsm)
resnet18_maxloss = torch.cat(resnet18_maxloss)
resnet18_pgd = torch.cat(resnet18_pgd)

# %%
cnn_fgsm = cnn_fgsm.cpu().detach().numpy()
cnn_maxloss = cnn_maxloss.cpu().detach().numpy()
cnn_pgd = cnn_pgd.cpu().detach().numpy()

# %%
resnet18_fgsm = resnet18_fgsm.cpu().detach().numpy()
resnet18_maxloss = resnet18_maxloss.cpu().detach().numpy()
resnet18_pgd = resnet18_pgd.cpu().detach().numpy()

# %%
perturbs = [
    cnn_fgsm,
    cnn_pgd,
    cnn_maxloss,
    resnet18_fgsm,
    resnet18_pgd,
    resnet18_maxloss,
]

label = [
    "cnn_fgsm",
    "cnn_pgd",
    "cnn_maxloss",
    "resnet18_fgsm",
    "resnet18_pgd",
    "resnet18_maxloss",
]

# %%
dists = []
for i in perturbs:
    s = []
    for j in perturbs:
        norm = 0
        for im1, im2 in zip(i, j):
            norm += la.norm(im1.ravel() - im2.ravel(), ord=1)
        s.append(norm)
    dists.append(s)

# %%
dists = np.array(dists)

# %%
sns.heatmap(dists, annot=True, xticklabels=label, yticklabels=label)

# %%
