import torch
from torch import nn


def init_params(m, method="random"):
    if type(m) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]:
        if method == "random":
            m.weight.data = torch.randn(m.weight.size()) * 0.01
        elif method == "zeros":
            m.wright.data = torch.zeros(m.weight.size())
        else:
            raise NotImplementedError("Method not recognised.")
        m.bias.data = torch.zeros(m.bias.size())
