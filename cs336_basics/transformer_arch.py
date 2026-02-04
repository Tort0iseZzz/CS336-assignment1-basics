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
        super().__init__()

        self.in_features = in_features # final dimension of the input
        self.out_features = out_features # final dimension of the output

        factory_kwargs = {"device": device, "dtype": dtype}
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
    


class Embedding(torch.nn.Module):
    __constants__ = [
        "num_embeddings",
        "embedding_dim"
    ]
    num_embeddings: int
    embedding_dim: int
    weight: Tensor
    
    def __init__(self, num_embeddings:int, embedding_dim:int, device=None, dtype=None):
        """Construct an embedding module."""
        super().__init__()

        self.num_embeddings = num_embeddings # Size of the vocabulary
        self.embedding_dim = embedding_dim # d_model, dimension of the embedding vectors
        
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = Parameter(
            torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.trunc_normal_(self.weight, 0.0, 1.0, -3.0, 3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Lookup the embedding vectors for the given token IDs."""
        return self.weight[token_ids]
    

class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization"""
    __constants__ = [
        "d_model",
        "eps"
    ]
    d_model: int
    eps: float
    gain: Tensor

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """Construct the RMSNorm module."""
        super().__init__()

        self.d_model = d_model # Hidden dimension of the model
        self.eps = eps # Epsilon value for numerical stability

        factory_kwargs = {"device": device, "dtype": dtype}
        self.gain = Parameter(
            torch.ones((d_model), **factory_kwargs)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        input:  (batch_size, sequence_length, d_model)
        return: (batch_size, sequence_length, d_model)
        """
        # upcast input to torch.float32 to prevent overflow
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # (sum{xi^2}) / d_model -> (..., 1)
        # "/d_model" is already in mean process
        ms = torch.mean(x**2, dim=-1, keepdim=True)
        # xi / sqrt(ms + eps): (..., d_model) * (..., 1) = (..., d_model)
        normed_x = x * torch.rsqrt(ms + self.eps)
        # normed_x * gaini: (..., d_model) * (d_model) = (..., d_model)
        result = normed_x * self.gain
        # Return the result in the original dtype
        return result.to(in_dtype)