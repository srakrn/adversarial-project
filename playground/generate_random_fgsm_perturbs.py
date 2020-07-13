# %%
import numpy as np
import numpy.linalg as la
import seaborn as sns
import torch
from torch import nn

from clustre.attacking import fgsm_perturbs
from clustre.helpers.datasets import mnist_trainloader
from clustre.models import mnist_cnn, mnist_resnet18
from clustre.models.state_dicts import mnist_cnn_state, mnist_resnet18_state

sns.set()


# %%
mnist_cnn.load_state_dict(mnist_cnn_state)
mnist_resnet18.load_state_dict(mnist_resnet18_state)

mnist_cnn.to("cuda")
mnist_resnet18.to("cuda")

# %%
criterion = nn.CrossEntropyLoss()

# %%
for i in range(100):
    print(i)
    torch.manual_seed(i)
    ps = []
    for X, y in iter(mnist_trainloader):
        X = X.to("cuda")
        y = y.to("cuda")
        p = fgsm_perturbs(mnist_cnn, criterion, X, y, random=True)
        ps.append(p)
    ps = torch.cat(ps).cpu().detach().numpy()
    np.save(
        f"playground/random_fgsm_perturbs/random_fgsm_mnist_cnn_{i}.npy", ps
    )

# %%
for i in range(100):
    print(i)
    torch.manual_seed(i)
    ps = []
    for X, y in iter(mnist_trainloader):
        X = X.to("cuda")
        y = y.to("cuda")
        p = fgsm_perturbs(mnist_resnet18, criterion, X, y, random=True)
        ps.append(p)
    ps = torch.cat(ps).cpu().detach().numpy()
    np.save(
        f"playground/random_fgsm_perturbs/random_fgsm_mnist_resnet18_{i}.npy",
        ps,
    )
