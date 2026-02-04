import math
import torch
from torch import Tensor
from torch.nn.parameter import Parameter

class Linear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features:int, out_features:int, device=None, dtype=None):
        """Construct a linear transformation module"""
        # Initialize torch.nn.Module(father class)
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_features = in_features # final dimension of the input
        self.out_features = out_features # final dimension of the output
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        a = -3 * std
        b = 3 * std
        torch.nn.init.trunc_normal_(self.weight, 0.0, std, a, b)
        

    def forward(self, x: Tensor) -> Tensor:
        """Apply the linear transformation to the input."""
        return torch.matmul(x, self.weight.t())