# %%
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

from typing import Union, Optional, Tuple, List, Dict
from jaxtyping import Float, Int

# %%
class ReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(x, torch.tensor(0.0))

class Linear(nn.Module):

    def __init__(
        self,
        in_feat: torch.Tensor,
        out_feat: torch.Tensor,
        use_bias: bool = True,
    ):
        super().__init__()

        # Kaiming initialization
        kaiming_scaling = 1 / np.sqrt(in_feat)
        self.weights = nn.Parameter(
            (torch.rand(in_feat, out_feat) * kaiming_scaling) - kaiming_scaling
        )
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(
                (torch.rand(out_feat) * kaiming_scaling) - kaiming_scaling
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_x = einops.einsum(
            x, 
            self.weights,
            "batch in, in out -> batch out"
        )

        if self.use_bias:
            new_x += self.bias

        return new_x

class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_dims = list(x.shape)
        self.end_dim = self.end_dim if self.end_dim > 0 else self.end_dim + len(x.shape)
        return x.reshape(
            new_dims[: self.start_dim] + [-1] + new_dims[self.end_dim + 1 :]
        )

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.linear1 = Linear(28 * 28, 100)
        self.relu = ReLU()
        self.linear2 = Linear(100, 10)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MLP of form input -> flatten -> linear -> relu -> linear -> out
        # Shapes (28x28) -> (28^2 x 100) -> (100 x 10) -> (10)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

class Conv2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()

        # Xavier initialization
        sf = 1 / np.sqrt(in_channels * kernel_size * kernel_size)
        weight = sf * (2 * torch.rand(out_channels, in_channels, kernel_size, kernel_size) - 1)
        self.weights = nn.Parameter(weight)
        self.stride = stride
        self.padding = padding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weights, stride=self.stride, padding=self.padding)

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

class BatchNorm2d(nn.Module):
    running_mean: Float[Tensor, "num_features"]
    running_var: Float[Tensor, "num_features"]
    num_batches_tracked: Int[Tensor, ""] # This is how we denote a scalar tensor

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculating mean and var over all dims except for the channel dim
        if self.training:
            # Take mean over all dimensions except the feature dimension
            # Using keepdim=True so we don't have to worry about broadasting them with x at the end
            mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
            var = torch.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True)
            # Updating running mean and variance, in line with PyTorch documentation
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
            self.num_batches_tracked += 1
        else:
            mean = einops.rearrange(self.running_mean, "channels -> 1 channels 1 1")
            var = einops.rearrange(self.running_var, "channels -> 1 channels 1 1")

        # Rearranging these so they can be broadcasted (although there are other ways you could do this)
        weight = einops.rearrange(self.weight, "channels -> 1 channels 1 1")
        bias = einops.rearrange(self.bias, "channels -> 1 channels 1 1")

        return ((x - mean) / torch.sqrt(var + self.eps)) * weight + bias

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["num_features", "eps", "momentum"]])

class AveragePool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3))

class Sequential(nn.Module):
    _modules: Dict[str, nn.Module]

    def __init__(self, *modules: nn.Module):
        super().__init__()
        self._modules = {str(i): module for i, module in enumerate(modules)}

    def __get_item__(self, key: str) -> nn.Module:
        return self._modules[key]
    
    def __set_item__(self, key: str, module: nn.Module) -> None:
        self._modules[key] = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self._modules.values():
            x = module(x)
        return x

class ResidualBlock(nn.Module):

    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.first_stride = first_stride

        self.left_branch = Sequential(
            Conv2d(
                in_feats,
                out_feats,
                kernel_size=3,
                stride=first_stride,
                padding=1
            ),
            BatchNorm2d(
                out_feats
            ),
            ReLU(),
            Conv2d(
                out_feats,
                out_feats,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            BatchNorm2d(
                out_feats
            )
        )

        if first_stride > 1:
            self.right_branch = Sequential(
                Conv2d(
                    in_feats,
                    out_feats,
                    kernel_size=1,
                    stride=first_stride, 
                ),
                BatchNorm2d(
                    out_feats
                )
            )
        else:
            self.right_branch = nn.Identity()

        self.relu = ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_left = self.left_branch(x)
        x_right = self.right_branch(x)
        
        return self.relu(x_left + x_right)

class BlockGroup(nn.Module):

    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        super().__init__()
        blocks_list = []
        blocks_list.append(
            ResidualBlock(
                in_feats,
                out_feats,
                first_stride=first_stride
            )
        )

        for i in range(n_blocks - 1):
            blocks_list.append(
                ResidualBlock(
                    out_feats,
                    out_feats
                )
            )
        
        self.blocks = Sequential(
            *blocks_list
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
# %%
