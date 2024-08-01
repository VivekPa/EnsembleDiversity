from typing import List, Union

import torch
from torch import nn

from .networks import MLP


def _convert_to_size(shape: Union[int, torch.Size]):
    if isinstance(shape, int):
        return torch.Size((shape,))
    return shape


class Regressor(nn.Module):
    def __init__(
        self,
        input_shape: Union[int, torch.Size],
        output_dim: int,
        py_x: nn.Module,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.input_shape = _convert_to_size(input_shape)
        self.output_dim = output_dim
        self.py_x = py_x
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.py_x(x)

    def input_grad(self, x, create_graph=False):
        x.requires_grad_(True)
        y_dist = self.py_x(x)
        y_mean = y_dist.mean()  # Extract the mean from the Gaussian distribution
        grad, = torch.autograd.grad(y_mean, x, grad_outputs=torch.ones_like(y_mean), create_graph=create_graph)
        return grad


class MLPRegressor(Regressor):
    def __init__(
        self,
        input_shape: Union[int, torch.Size],
        output_dim: int,
        hidden_dims: List[int],
        nonlinearity: nn.Module = nn.ReLU(),
        bias: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        input_shape = _convert_to_size(input_shape)
        assert len(input_shape) == 1

        py_x = MLP(
            [input_shape[0]] + hidden_dims + [output_dim],
            nonlinearity=nonlinearity,
            bias=bias,
            device=device,
        )

        super().__init__(input_shape, output_dim, py_x, device)


class LinearModel(Regressor):
    def __init__(
        self,
        input_shape: Union[int, torch.Size],
        output_dim: int,
        device: torch.device = torch.device("cpu"),
    ):
        input_shape = _convert_to_size(input_shape)
        assert len(input_shape) == 1

        py_x = nn.Linear(input_shape[0], output_dim * 2, bias=True)

        super().__init__(input_shape, output_dim, py_x, device)

    def forward(self, x: torch.Tensor) -> torch.distributions.normal.Normal:
        mean, log_std = torch.chunk(self.py_x(x), 2, dim=-1)
        std = torch.exp(log_std)
        dist = torch.distributions.normal.Normal(mean, std)
        return dist

