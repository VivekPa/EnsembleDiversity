from typing import List, Union

import torch
from torch import nn
from torch.nn import functional as F

from .activations import CategoricalActivation
from .networks import MLP


def _convert_to_size(shape: Union[int, torch.Size]):
    if isinstance(shape, int):
        return torch.Size((shape,))
    return shape


class Classifier(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.distributions.Categorical:
        return self.py_x(x)

    def binary_logits(self, x):
        if self.output_dim != 2:
            pass
        else:
            return self.py_x(x).logits

    def input_grad(self, x, create_graph=False):
        x.requires_grad_(True)
        y_dist = self.py_x(x)
        logits = y_dist.logits  # Extract the logits from the Categorical distribution
        grad, = torch.autograd.grad(logits, x, grad_outputs=torch.ones_like(logits), create_graph=create_graph)
        return grad


class LinearClassifier(Classifier):
    def __init__(
        self,
        input_shape: Union[int, torch.Size],
        output_dim: int,
        device: torch.device = torch.device("cpu"),
    ):
        input_shape = _convert_to_size(input_shape)
        assert len(input_shape) == 1

        py_x = nn.Sequential(
            nn.Linear(input_shape[0], output_dim),
            CategoricalActivation(),
        )

        super().__init__(input_shape, output_dim, py_x, device)


class MLPClassifier(Classifier):
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
            activation=CategoricalActivation(),
            bias=bias,
            device=device,
        )

        super().__init__(input_shape, output_dim, py_x, device)


class MNISTConvClassifier(Classifier):
    def __init__(self, device: torch.device = torch.device("cpu")):
        input_shape = torch.Size((1, 28, 28))
        output_dim = 10

        py_x = nn.Sequential(
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
            nn.Linear(128, output_dim),
            CategoricalActivation(),
        )
        super().__init__(input_shape, output_dim, py_x, device)

class MNISTSimpleConvClassifier(Classifier):
    def __init__(self, device: torch.device = torch.device("cpu")):
        input_shape = torch.Size((1, 28, 28))
        output_dim = 10

        py_x = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=2, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(8, 16, 3, stride=2, padding=0),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(2 * 2 * 16, 64),
            nn.Sigmoid(),
            nn.Linear(64, output_dim),
            CategoricalActivation(),
        )
        super().__init__(input_shape, output_dim, py_x, device)


class CELEBAConvClassifier(Classifier):
    def __init__(self, device: torch.device = torch.device("cpu")):
        input_shape = torch.Size((3, 128, 128))
        output_dim = 10

        py_x = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=4, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            CategoricalActivation(),
        )
        super().__init__(input_shape, output_dim, py_x, device)

class CIFAR10ConvClassifier(Classifier):
    def __init__(self, device: torch.device = torch.device("cpu")):
        input_shape = torch.Size((3, 32, 32))
        output_dim = 10

        py_x = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(15 * 15 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            CategoricalActivation(),
        )
        super().__init__(input_shape, output_dim, py_x, device)

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

class CIFAR10ResNetClassifier(Classifier):
    def __init__(self, device: torch.device = torch.device("cpu")):
        input_shape = torch.Size((3, 32, 32))
        output_dim = 10  # CIFAR-10 has 10 classes

        py_x = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            CIFARResidualBlock(64, 128, stride=2),
            CIFARResidualBlock(128, 256, stride=2),
            CIFARResidualBlock(256, 512, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(512, output_dim),
            CategoricalActivation(),
        )

        super().__init__(input_shape, output_dim, py_x, device)

class ImageNetConvClassifier(Classifier):
    def __init__(self, device: torch.device = torch.device("cpu")):
        input_shape = torch.Size((3, 128, 128))
        output_dim = 1000

        py_x = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=4, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            CategoricalActivation(),
        )
        super().__init__(input_shape, output_dim, py_x, device)

class ImageNetLargeClassifier(Classifier):
    def __init__(self, device: torch.device = torch.device("cpu")):
        input_shape = torch.Size((3, 128, 128))
        output_dim = 1000

        py_x = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, output_dim),
            CategoricalActivation(),
        )
        super().__init__(input_shape, output_dim, py_x, device)

class AlexNetClassifier(Classifier):
    def __init__(self, device: torch.device = torch.device("cpu")):
        input_shape = torch.Size((3, 128, 128))
        output_dim = 1000

        py_x = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_dim),
            CategoricalActivation(),
        )
        super().__init__(input_shape, output_dim, py_x, device)

class ImageNetMLPClassifier(Classifier):
    def __init__(self, device: torch.device = torch.device("cpu")):
        input_shape = torch.Size((3, 128, 128))
        output_dim = 1000

        py_x = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 128 * 128, 128 * 128),
            nn.ReLU(),
            # nn.Linear(128 * 128, 10 * 128),
            # nn.ReLU(),
            nn.Linear(128 * 128, output_dim),
            CategoricalActivation(),
        )
        super().__init__(input_shape, output_dim, py_x, device)

class VGG16Classifier(Classifier):
    def __init__(self, device: torch.device = torch.device("cpu")):
        input_shape = torch.Size((3, 128, 128))
        output_dim = 1000

        def make_layers(cfg, batch_norm=False):
            layers = []
            in_channels = 3
            for v in cfg:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = v
            return nn.Sequential(*layers)

        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        features = make_layers(cfg, batch_norm=False)

        py_x = nn.Sequential(
            features,
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, output_dim),
            CategoricalActivation(),
        )
        super().__init__(input_shape, output_dim, py_x, device)

