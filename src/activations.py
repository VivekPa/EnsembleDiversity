import torch
from torch import nn


class NormalActivation(nn.Module):
    def forward(self, x: torch.Tensor):
        assert x.shape[-1] % 2 == 0, "x must have a multiple of 2 dimensions."

        loc, logvar = torch.split(x, x.shape[-1] // 2, dim=-1)
        logvar = logvar - 4.0
        return torch.distributions.Normal(loc, logvar.exp().pow(0.5)+1e-6*torch.eye(1))


class CategoricalActivation(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.distributions.Categorical(logits=x)


class BernoulliActivation(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.distributions.Bernoulli(logits=x)

class BernoulliActivation_c(nn.Module):
    def __init__(self, num_channels=1):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, x):
        return torch.distributions.Independent(
            torch.distributions.Bernoulli(logits=x), reinterpreted_batch_ndims=self.num_channels
        )

