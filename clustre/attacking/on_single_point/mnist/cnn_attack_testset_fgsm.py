# %%
import logging
import os
import sys

import torch
from torch import nn, optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attack import fgsm  # isort:skip
from mnist_helpers import mnist_cnn_model, mnist_testset  # isort:skip

logging.basicConfig(
    filename=f"logs/{os.path.basename(__file__)}.log",
    filemode="a",
    level="INFO",
    format="%(process)d-%(levelname)s-%(asctime)s-%(message)s",
)
# %%
OUTPUT_PATH = "perturbs/on_single_point/mnist/cnn_fgsm_perturbs_testset.pt"

# %%
criterion = nn.CrossEntropyLoss()
logging.info("Started running")
perturbs = fgsm(mnist_cnn_model, criterion, mnist_testset, verbose=True)
logging.info("Ended running")
#  %%
torch.save(perturbs, OUTPUT_PATH)
