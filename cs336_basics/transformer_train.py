import math
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from collections.abc import Callable, Iterable
from typing import Optional
from jaxtyping import Bool, Float, Int


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
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3
    ):
        """
        lr: learning rate (alpha)
        beta = (beta1, beta2):  control the updates to the moment estimates
        eps: a small value used to improve numerical stability in case we get extremely small values in v
        wdr: weight decay rate (lambda)
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
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
                t = state.get("t", 0) + 1  # Increment iteration number.
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
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # 5: Apply weight decay
                # θ_t = θ_{t-1} - γ * λ * θ_t
                if wdr != 0:
                    p.data.mul_(1 - lr * wdr)

                # 6: Update the parameters
                # θ_t = θ_t - lr_t * m_t / (sqrt(v_t) + ε)
                p.data.addcdiv_(m, v.sqrt() + eps, value=-lr_t)

                # 7: write back
                state["m"] = m
                state["v"] = v
                state["t"] = t

        return loss


def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    # warm up
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate

    # cosine annealing
    if it <= cosine_cycle_iters:
        in_cos = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi
        return (
            min_learning_rate
            + (1 + math.cos(in_cos)) * (max_learning_rate - min_learning_rate) / 2
        )

    # post annealing
    return min_learning_rate


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float
) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    # 1: get the gradients for all parameters
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return

    # 2: compute l2 norm
    total_norm_sq = torch.tensor(0.0)  # compute total norm square first
    for g in grads:
        total_norm_sq += torch.sum(g**2)
    total_norm = torch.sqrt(total_norm_sq)

    # 3. update
    eps = 1e-6
    if total_norm > max_l2_norm:
        # scale down: M / (||g|| + eps)
        clip = max_l2_norm / (total_norm + eps)

        # in-place update grad
        for g in grads:
            g.mul_(clip)
