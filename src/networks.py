from typing import List

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        dims: List[int],
        nonlinearity: nn.Module = nn.ReLU(),
        activation: nn.Module = nn.Identity(),
        bias: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.device = device
        self.net = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1], bias=bias).to(self.device))

            if i == len(dims) - 2:
                self.net.append(activation)
            else:
                self.net.append(nonlinearity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.net:
            x = layer(x)
        return x
