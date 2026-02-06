import math
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import einops

class Linear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features:int, out_features:int, device=None, dtype=None):
        """
        Construct a linear transformation module
        -> (out_features, in_features)
        """
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
    

class SwiGLU_FeedForward(torch.nn.Module):
    __constants__ = [
        "d_model",
        "d_ff"
    ]
    d_model: int
    d_ff: int

    def __init__(self, d_model: int, d_ff=None, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        if d_ff:
            self.d_ff = d_ff
        else:
            d_ff = int(8/3 * d_model)
            self.d_ff = 64 * ((d_ff + 64 - 1) // 64) # 64x to make good use of memory

        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)

    #def forward_SiLU(self, x: Tensor) -> Tensor:
        #"""
        #SiLU(x) = x * sigmoid(x)
        #in SwiGLU, x here is w1(x)
        #"""
        #return x * torch.sigmoid(x)

    def forward_SwiGLU(self, x: Tensor) -> Tensor:
        """
        input: (d_model)
        output: (d_model)
        SiLU + Gate: w2(SiLU(w1(x)) * w3(x))
        """
        silu = self.w1(x) * torch.sigmoid(self.w1(x))
        glu = silu * self.w3(x)
        return self.w2(glu)
    

class RotaryPositionalEmbedding(torch.nn.Module):
    cos: Tensor
    sin: Tensor
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers.
            theta # Θ value for the RoPE
            d_k # dimension of query and key vectors
            max_seq_len # Maximum sequence length that will be inputted
            device # Device to store the buffer on
        """
        super().__init__()
        
        # 1 / theta^((2k-2)/d), for k = 1,...,d/2
        # i.e. 1 / theta^((0,2,...,d-2)/d)
        angles = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        # i = 0,1,...,max_len prepared for all positions
        position_is = torch.arange(max_seq_len, device=device, dtype=angles.dtype)

        # i / theta^((0,2,...,d-2)/d)
        angle_is = torch.einsum("i, j -> ij", position_is, angles)
        
        # compute sin(angle) cos(angle) and buffer them
        # persistent=False 表示不把这个表存在模型权重文件里，每次初始化重算即可
        self.register_buffer("cos", angle_is.cos(), persistent=False)
        self.register_buffer("sin", angle_is.sin(), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len) 
        # note that tokens are not necessarily start from 0
        return: (..., seq_len, d_k)
        """
        # (..., seq_len, d/2)
        cos_selected = self.cos[token_positions]
        sin_selected = self.sin[token_positions]

        # for every two neighbor input elements (x1, x2)
        # after rotate: (x1cos - x2sin, x1sin + x2cos)
        # -> we don't need to actually build the huge rotate matrix R
        # instead, (x1, x2) * cos + (-x2, x1) * sin!
        # therefore, we need tensor (cos, cos), (sin, sin) (doubled)
        # and tensor (x1, x2), (-x2, x1)

        # 1. double: (..., seq_len, d_k)
        cos_selected = torch.repeat_interleave(cos_selected, 2, dim=-1)
        sin_selected = torch.repeat_interleave(sin_selected, 2, dim=-1)

        # 2. (x1, x2) -> (-x2, x1)
        x_interleaved = einops.rearrange(x, "... (n c) -> ... n c", c=2)
        x_1, x_2 = x_interleaved[..., 0], x_interleaved[..., 1]
        x_interleaved = einops.rearrange(torch.stack([-x_2, x_1], dim=-1), "... n c -> ... (n c)")

        # 3. (x1, x2) * cos + (-x2, x1) * sin
        return x * cos_selected + x_interleaved * sin_selected
    

def softmax(in_features: Tensor, dim: int) -> Tensor:
    # softmax(xi) = exp(xi - max) / sum_j(exp(xj - max))
    # substract max from all xi is to avoid inf

    # 1. find the max_x
    max_x = torch.max(in_features, dim=dim, keepdim=True)[0]
    
    # 2. e^(x - max)
    exp_x = torch.exp(in_features - max_x)
    
    # 3. sum(exp(x))
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    
    return exp_x / sum_exp