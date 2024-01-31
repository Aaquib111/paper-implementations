# %%
import torch
import torch.nn as nn
import numpy as np
import einops

from typing import Union, Optional, Tuple, List, Dict
from jaxtyping import Float, Int

# %%
class ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        pass

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


# %%
