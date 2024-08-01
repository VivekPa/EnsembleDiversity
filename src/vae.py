from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18

from .activations import BernoulliActivation, NormalActivation, BernoulliActivation_c
from .networks import MLP


def _convert_to_size(shape: Union[int, torch.Size]):
    if isinstance(shape, int):
        return torch.Size((shape,))
    return shape


class VAE(nn.Module):
    def __init__(
        self,
        x_shape: Union[int, torch.Size],
        z_shape: Union[int, torch.Size],
        qz_x: nn.Module,
        px_z: nn.Module,
        c_dim=None,  # New argument for the dimension of c
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.x_shape = _convert_to_size(x_shape)
        self.z_shape = _convert_to_size(z_shape)
        self.c_dim = _convert_to_size(c_dim) if c_dim else None  # Store c_dim

        assert len(self.z_shape) == 1, "Assumes flattened latent space."

        self.qz_x = qz_x.to(device)
        self.px_z = px_z.to(device)
        self.device = device

        # Define prior.
        self.pz = torch.distributions.Normal(
            torch.zeros(self.z_shape[0], device=device),
            torch.ones(self.z_shape[0], device=device),
        )

    def encode(self, x: torch.Tensor, c: torch.Tensor=None) -> torch.Tensor:
        """Returns the mean of q(z | x, c) if c is provided, else q(z | x)."""
        if self.c_dim is not None and c is not None:
            inputs = torch.cat([x, c], dim=1)
        else:
            inputs = x
        return self.qz_x(inputs).mean  # Note that qz_x should also take c into account

    def decode(self, z: torch.Tensor, c: torch.Tensor=None) -> torch.Tensor:
        """Returns the mean of p(x | z, c) if c is provided, else p(x | z)."""
        if self.c_dim is not None and c is not None:
            inputs = torch.cat([z, c], dim=1)
        else:
            inputs = z
        return self.px_z(inputs).mean  # Note that px_z should also take c into account

    def reconstruct_x(
        self, x: torch.Tensor, c: Optional[torch.Tensor] = None, deterministic: bool = False, num_samples: int = 1
    ) -> torch.Tensor:
        assert x.shape[1:] == self.x_shape

        if deterministic:
            return self.decode(self.encode(x, c), c)

        x = x.repeat(num_samples, *[1 for _ in range(len(self.x_shape))])
        z = self.qz_x(x, c).sample()
        x_recon = self.px_z(z, c).sample()

        if num_samples > 1:
            x_recon = x_recon.reshape(num_samples, -1, *x_recon.shape[1:])

        return x_recon

    def forward(
        self, x: torch.Tensor, c: Optional[torch.Tensor] = None, deterministic: bool = False, num_samples: int = 1
    ) -> torch.Tensor:
        return self.reconstruct_x(x, c, deterministic, num_samples)

    def elbo(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None, 
        num_samples: int = 1,
        n: Optional[int] = None,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x.shape[1:] == self.x_shape

        x = x.repeat(num_samples, *[1 for _ in range(len(self.x_shape))])
        qz_x = self.qz_x(x, c)
        z = qz_x.rsample()
        px_z = self.px_z(z, c)

        # Average across sample dimension, sum across others.
        exp_ll = px_z.log_prob(x).reshape(num_samples, -1, *self.x_shape)
        exp_ll = exp_ll.mean(0).sum()
        kl = torch.distributions.kl_divergence(qz_x, self.pz).reshape(
            num_samples, -1, *self.z_shape
        )
        kl = kl.mean(0).sum()

        if n is not None:
            # Correct for batch size.
            exp_ll = exp_ll * (n / len(x))

        elbo = exp_ll - beta * kl  # Multiply the KL-divergence term by beta
        return elbo, exp_ll, kl


    def manifold_jac(self, x, c: Optional[torch.Tensor] = None, create_graph=True):
        x.requires_grad_(True)
        y_dist = self.qz_x(x, c)
        vae_enc = lambda v: self.qz_x(v, c).mean
        grad, = torch.autograd.functional.jacobian(vae_enc, x, create_graph=create_graph)
        return grad

# class VAE(nn.Module):
#     def __init__(
#         self,
#         x_shape: Union[int, torch.Size],
#         z_shape: Union[int, torch.Size],
#         qz_x: nn.Module,
#         px_z: nn.Module,
#         device: torch.device = torch.device("cpu"),
#     ):
#         super().__init__()

#         self.x_shape = _convert_to_size(x_shape)
#         self.z_shape = _convert_to_size(z_shape)

#         assert len(self.z_shape) == 1, "Assumes flattened latent space."

#         self.qz_x = qz_x.to(device)
#         self.px_z = px_z.to(device)
#         self.device = device

#         # Define prior.
#         self.pz = torch.distributions.Normal(
#             torch.zeros(self.z_shape[0], device=device),
#             torch.ones(self.z_shape[0], device=device),
#         )

#     def encode(self, x: torch.Tensor) -> torch.Tensor:
#         """Returns the mean of q(z | x)."""
#         return self.qz_x(x).mean

#     def decode(self, z: torch.Tensor) -> torch.Tensor:
#         """Returns the mean of p(x | z)."""
#         return self.px_z(z).mean

#     def reconstruct_x(
#         self, x: torch.Tensor, deterministic: bool = False, num_samples: int = 1
#     ) -> torch.Tensor:
#         assert x.shape[1:] == self.x_shape

#         if deterministic:
#             return self.decode(self.encode(x))

#         x = x.repeat(num_samples, *[1 for _ in range(len(self.x_shape))])
#         z = self.qz_x(x).sample()
#         x_recon = self.px_z(z).sample()

#         if num_samples > 1:
#             x_recon = x_recon.reshape(num_samples, -1, *x_recon.shape[1:])

#         return x_recon

#     def forward(
#         self, x: torch.Tensor, deterministic: bool = False, num_samples: int = 1
#     ) -> torch.Tensor:
#         return self.reconstruct_x(x, deterministic, num_samples)

#     # def elbo(
#     #     self,
#     #     x: torch.Tensor,
#     #     num_samples: int = 1,
#     #     n: Optional[int] = None,
#     # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     #     assert x.shape[1:] == self.x_shape

#     #     x = x.repeat(num_samples, *[1 for _ in range(len(self.x_shape))])
#     #     qz_x = self.qz_x(x)
#     #     z = qz_x.rsample()
#     #     px_z = self.px_z(z)

#     #     # Average across sample dimension, sum across others.
#     #     exp_ll = px_z.log_prob(x).reshape(num_samples, -1, *self.x_shape)
#     #     exp_ll = exp_ll.mean(0).sum()
#     #     kl = torch.distributions.kl_divergence(qz_x, self.pz).reshape(
#     #         num_samples, -1, *self.z_shape
#     #     )
#     #     kl = kl.mean(0).sum()

#     #     if n is not None:
#     #         # Correct for batch size.
#     #         exp_ll = exp_ll * (n / len(x))

#     #     elbo = exp_ll - kl
#     #     return elbo, exp_ll, kl

#     def elbo(
#         self,
#         x: torch.Tensor,
#         num_samples: int = 1,
#         n: Optional[int] = None,
#         beta: float = 1.0,  # Add the beta parameter (default value is 1, which is the same as the standard VAE)
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         assert x.shape[1:] == self.x_shape

#         x = x.repeat(num_samples, *[1 for _ in range(len(self.x_shape))])
#         qz_x = self.qz_x(x)
#         z = qz_x.rsample()
#         px_z = self.px_z(z)

#         # Average across sample dimension, sum across others.
#         exp_ll = px_z.log_prob(x).reshape(num_samples, -1, *self.x_shape)
#         exp_ll = exp_ll.mean(0).sum()
#         kl = torch.distributions.kl_divergence(qz_x, self.pz).reshape(
#             num_samples, -1, *self.z_shape
#         )
#         kl = kl.mean(0).sum()

#         if n is not None:
#             # Correct for batch size.
#             exp_ll = exp_ll * (n / len(x))

#         elbo = exp_ll - beta * kl  # Multiply the KL-divergence term by beta
#         return elbo, exp_ll, kl


#     def manifold_jac(self, x, create_graph=True):
#         x.requires_grad_(True)
#         y_dist = self.qz_x(x)
#         vae_enc = lambda v: self.qz_x(v).mean
#         # y_mean = y_dist.mean  # Extract the mean from the Gaussian distribution
#         grad, = torch.autograd.functional.jacobian(vae_enc, x, create_graph=create_graph)
#         return grad


class MLPVAE(VAE):
    def __init__(
        self,
        x_shape: Union[int, torch.Size],
        z_shape: Union[int, torch.Size],
        encoder_dims: List[int],
        decoder_dims: List[int],
        bias: bool = True,
        decoder_activation: Union[
            NormalActivation, BernoulliActivation
        ] = NormalActivation(),
        nonlinearity: nn.Module = nn.ReLU(),
        device: torch.device = torch.device("cpu"),
    ):
        x_shape = _convert_to_size(x_shape)
        z_shape = _convert_to_size(z_shape)
        assert len(x_shape) == 1, "Assumes flattened output shape."
        assert len(z_shape) == 1, "Assumes flattened latent shape."

        x_dim = x_shape[0]
        z_dim = z_shape[0]

        qz_x = MLP(
            [_convert_to_size(x_shape)[0]]
            + encoder_dims
            + [2 * _convert_to_size(z_shape)[0]],
            nonlinearity=nonlinearity,
            activation=NormalActivation(),
            bias=bias,
            device=device,
        )

        if isinstance(decoder_activation, NormalActivation):
            output_dim = 2 * x_dim
        else:
            output_dim = x_dim

        px_z = MLP(
            [z_dim] + decoder_dims + [output_dim],
            nonlinearity=nonlinearity,
            activation=decoder_activation,
            bias=bias,
            device=device,
        )

        super().__init__(x_shape, z_shape, qz_x, px_z, device)


class MNISTConvVAE(VAE):
    def __init__(
        self,
        z_shape: Union[int, torch.Size],
        device: torch.device = torch.device("cpu"),
    ):
        x_shape = torch.Size((1, 28, 28))
        z_shape = _convert_to_size(z_shape)
        assert len(z_shape) == 1, "Assumes flattened latent space."

        qz_x = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * z_shape[0]),
            NormalActivation(),
        )

        px_z = nn.Sequential(
            nn.Linear(z_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(32, 3, 3)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
            BernoulliActivation(),
        )

        super().__init__(x_shape, z_shape, qz_x, px_z, device)


class CELEBAConvVAE(VAE):
    def __init__(
        self,
        z_shape: Union[int, torch.Size],
        device: torch.device = torch.device("cpu"),
    ):
        x_shape = torch.Size((3, 128, 128))
        z_shape = _convert_to_size(z_shape)
        assert len(z_shape) == 1, "Assumes flattened latent space."

        qz_x = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),  # (8, 65, 65)
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),  # (16, 33, 33)
            nn.BatchNorm2d(16),
            # nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # (32, 17, 17)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # (128, 3, 3)
            nn.Flatten(),
            nn.Linear(4 * 4 * 128, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * z_shape[0]),
            NormalActivation(),
        )

        px_z = nn.Sequential(
            nn.Linear(z_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 4 * 4 * 128),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(128, 4, 4)),
            nn.ConvTranspose2d(
                128, 64, 3, stride=2, padding=1, output_padding=1
            ),  # (64, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, 3, stride=2, padding=1, output_padding=1
            ),  # (32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 16, 3, stride=2, padding=1, output_padding=1
            ),  # (16, 32, 32)
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(
                16, 8, 3, stride=2, padding=1, output_padding=1
            ),  # (8, 64, 64)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(
                8, 3, 3, stride=2, padding=1, output_padding=1
            ),  # (3, 128, 128)
            BernoulliActivation(),
        )

        super().__init__(x_shape, z_shape, qz_x, px_z, device)


class CELEBAConvVAE1(VAE):
    def __init__(
        self,
        z_shape: Union[int, torch.Size],
        device: torch.device = torch.device("cpu"),
    ):
        x_shape = torch.Size((3, 128, 128))
        z_shape = _convert_to_size(z_shape)
        assert len(z_shape) == 1, "Assumes flattened latent space."

        qz_x = nn.Sequential(
            nn.Conv2d(3, 8, 6, stride=2, padding=2),  # (8, 64, 64)
            nn.ReLU(),
            nn.Conv2d(8, 16, 6, stride=2, padding=2),  # (16, 32, 32)
            nn.BatchNorm2d(16),
            # nn.ReLU(),
            nn.Conv2d(16, 16, 6, stride=2, padding=2),  # (16, 16, 16)
            nn.ReLU(),
            nn.Conv2d(16, 32, 6, stride=2, padding=2),  # (32, 8, 8)
            nn.ReLU(),
            # nn.Conv2d(32, 128, 3, stride=2, padding=1), # (128, 3, 3)
            nn.Flatten(),
            nn.Linear(8 * 8 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * z_shape[0]),
            NormalActivation(),
        )

        px_z = nn.Sequential(
            nn.Linear(z_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 4 * 4 * 128),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(128, 4, 4)),
            nn.ConvTranspose2d(
                128, 64, 3, stride=2, padding=1, output_padding=1
            ),  # (64, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, 3, stride=2, padding=1, output_padding=1
            ),  # (32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 16, 3, stride=2, padding=1, output_padding=1
            ),  # (16, 32, 32)
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(
                16, 8, 3, stride=2, padding=1, output_padding=1
            ),  # (8, 64, 64)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(
                8, 3, 3, stride=2, padding=1, output_padding=1
            ),  # (3, 128, 128)
            BernoulliActivation(),
        )

        super().__init__(x_shape, z_shape, qz_x, px_z, device)

class CIFARResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class CIFARResConvVAE(VAE):
    def __init__(
        self,
        z_shape: Union[int, torch.Size],
        device: torch.device = torch.device("cpu"),
    ):
        x_shape = torch.Size((3, 32, 32))
        z_shape = _convert_to_size(z_shape)
        assert len(z_shape) == 1, "Assumes flattened latent space."

        qz_x = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            CIFARResidualBlock(64, 128, 2),
            CIFARResidualBlock(128, 256, 2),
            CIFARResidualBlock(256, 512, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 2 * z_shape[0]),
            NormalActivation(),
        )

        px_z = nn.Sequential(
            nn.Linear(z_shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, 4 * 4 * 512),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(512, 4, 4)),
            CIFARResidualBlock(512, 256, 1),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            BernoulliActivation(),
        )

        super().__init__(x_shape, z_shape, qz_x, px_z, device)

# class CIFARResNetConvVAE(VAE):
#     def __init__(
#         self,
#         z_shape: Union[int, torch.Size],
#         device: torch.device = torch.device("cpu"),
#     ):
#         x_shape = torch.Size((3, 32, 32))
#         z_shape = _convert_to_size(z_shape)
#         assert len(z_shape) == 1, "Assumes flattened latent space."

#         resnet = resnet18(pretrained=False)

#         # Modify the first layer
#         resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         resnet.bn1 = nn.BatchNorm2d(64)
#         resnet.relu = nn.ReLU(inplace=True)

#         # Remove the max pooling layer
#         resnet.maxpool = nn.Identity()

#         qz_x = nn.Sequential(
#             resnet.conv1,
#             resnet.bn1,
#             resnet.relu,
#             resnet.maxpool,
#             resnet.layer1,
#             resnet.layer2,
#             resnet.layer3,
#             resnet.layer4,
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(512, 2 * z_shape[0]),
#             NormalActivation(),
#         )

#         px_z = nn.Sequential(
#             nn.Linear(z_shape[0], 512),
#             nn.ReLU(),
#             nn.Linear(512, 2 * 2 * 512), # Changing here
#             nn.ReLU(),
#             nn.Unflatten(dim=1, unflattened_size=(512, 2, 2)), # And here
#             nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # Comment this and the next 3 lines to get 16x16 output
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
#             BernoulliActivation(),
#         )


#         super().__init__(x_shape, z_shape, qz_x, px_z, device)

class CIFARResNetConvQz_x(nn.Module):
    def __init__(self, z_shape):
        super().__init__()
        resnet = resnet18(pretrained=False)

        # Modify the first layer
        resnet.conv1 = nn.Conv2d(13, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.bn1 = nn.BatchNorm2d(64)
        resnet.relu = nn.ReLU(inplace=True)

        # Remove the max pooling layer
        resnet.maxpool = nn.Identity()

        self.resnet_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 2 * z_shape[0]),
            NormalActivation(),
        )

    def forward(self, x, c):
        c = c.unsqueeze(-1).unsqueeze(-1)  # Adds two dimensions to c making it (batch_size, 10, 1, 1)
        c = c.repeat(1, 1, 32, 32)  # Repeats c along the spatial dimensions to match the size of the input image
        x = torch.cat([x, c], dim=1)  # Concatenate along the channel dimension
        return self.resnet_layers(x)


class CIFARResNetConvPx_z(nn.Module):
    def __init__(self, z_shape):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(z_shape[0] + 10, 512), # Adding condition vector size to input here
            nn.ReLU(),
            nn.Linear(512, 2 * 2 * 512),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(512, 2, 2)),
        )
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # Comment this and the next 3 lines to get 16x16 output
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            BernoulliActivation(),
        )

    def forward(self, z, c):
        z = torch.cat([z, c], dim=1)  # Concatenate along the latent dimension
        z = self.linear_layers(z)
        return self.conv_layers(z)

class CIFARResNetConvCVAE(VAE):
    def __init__(
        self,
        z_shape: Union[int, torch.Size],
        c_dim: int,
        device: torch.device = torch.device("cpu"),
    ):
        x_shape = torch.Size((3, 32, 32))
        z_shape = _convert_to_size(z_shape)
        assert len(z_shape) == 1, "Assumes flattened latent space."

        qz_x = CIFARResNetConvQz_x(z_shape)
        px_z = CIFARResNetConvPx_z(z_shape)

        super().__init__(x_shape, z_shape, qz_x, px_z, c_dim, device)





