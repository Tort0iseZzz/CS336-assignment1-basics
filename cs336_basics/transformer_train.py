import math
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from collections.abc import Callable, Iterable
from typing import Optional
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
    log_sum_exp = (
        torch.log(torch.sum(torch.exp(inputs - max_o), dim=-1, keepdim=True)) + max_o
    )

    # 3. gather input[i][targeti]
    # mark: target should expend to batch_size vocab_size to feed the dim
    target_logits = torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1))

    # 4. log_sum_exp - target_logits
    loss = log_sum_exp - target_logits
    return loss.mean()


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3):
        """
        lr: learning rate (alpha)
        beta = (beta1, beta2):  control the updates to the moment estimates
        eps: a small value used to improve numerical stability in case we get extremely small values in v
        wdr: weight decay rate (lambda)
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay
                    }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            # get the parameters
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wdr = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Get state associated with p.
                state = self.state[p] 
                # Get or initialize values from the state.
                t = state.get("t", 0) + 1 # Increment iteration number.
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                
                # 1: Get the gradient of the loss
                grad = p.grad.data 

                # 2: Update the first moment estimate
                # m_t = β1 * m_{t-1} + (1 - β1) * g_t
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 3: Update the second raw moment estimate
                # v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 4: Compute adjusted alpha for iteration t
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                
                # 5: Apply weight decay
                # θ_t = θ_{t-1} - γ * λ * θ_t
                if wdr != 0:
                    p.data.mul_(1 - lr * wdr)
                
                # 6: Update the parameters
                # θ_t = θ_t - lr_t * m_t / (sqrt(v_t) + ε)
                p.data.addcdiv_(m, v.sqrt()+eps, value=-lr_t)

                # 7: write back
                state["m"] = m
                state["v"] = v
                state["t"] = t 

        return loss