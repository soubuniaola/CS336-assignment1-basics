from __future__ import annotations

import math
import torch
from typing import Optional
from collections.abc import Callable


def cos_lr(
        it: int,
        min_lr: float,
        max_lr: float,
        warmup_iters: int,
        cosine_annealing_iters: int,
):
    if it < warmup_iters:
        cosine_lr = max_lr * it / warmup_iters
    elif it > cosine_annealing_iters:
        cosine_lr = min_lr
    else:
        decay_ratio = (it - warmup_iters) / (cosine_annealing_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coef = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        cosine_lr = min_lr + coef * (max_lr - min_lr)

    return cosine_lr

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss= None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t",0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t+1) * grad
                state["t"] = t + 1
                
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), weight_decay=1e-1, eps=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "weight_decay": weight_decay,
            "eps": eps
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, beta1, beta2, decay_rate, eps = group["lr"], group["beta1"], group["beta2"], group["weight_decay"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t",1)
                grad = p.grad.data
                m = beta1 * state.get("m",torch.zeros_like(grad)) + (1 - beta1) * grad
                v = beta2 * state.get("v",torch.zeros_like(grad)) + (1 - beta2) * torch.square(grad)

                lr_t = lr * math.sqrt(1 - math.pow(beta2, t)) / (1 - math.pow(beta1, t))
                p.data -= lr_t * m / torch.sqrt(v + eps)
                p.data -= lr * decay_rate * p.data

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss



if __name__ == '__main__':
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = AdamW([weights])
    for t in range(100):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights ** 2).mean()  # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.