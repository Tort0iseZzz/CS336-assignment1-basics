import math
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from jaxtyping import Bool, Float, Int
import einops

def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class (token) for the ith example (text).
        targets (Int[Tensor, "batch_size"]): the index of the correct class (token).
            Each value must be between 0 and `num_classes - 1`.
    li = -log(softmax(inputi) * [targeti]) (note that targeti is the index)
       = -log (exp(input[i][targeti]) / sum_j{exp(input[i][j])})
       = log(sum_j{exp(input[i][j])}) - input[i][targeti]
    return the average of loss
    - Subtract the largest element for numerical stability to avoid explosion.
    """
    # 1. find the max_o
    max_o = torch.max(inputs, dim=-1, keepdim=True)[0]

    # 2. log(sum(exp(input - max_o))) + max_o
    log_sum_exp = torch.log(torch.sum(torch.exp(inputs - max_o), dim=-1, keepdim=True)) + max_o

    # 3. gather input[i][targeti]
    # mark: target should expend to batch_size vocab_size to feed the dim
    target_logits = torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1))

    # 4. log_sum_exp - target_logits
    loss = log_sum_exp - target_logits
    return loss.mean()

