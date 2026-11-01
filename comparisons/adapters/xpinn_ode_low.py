from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

LOCAL_WIDTHS = {2: 70, 3: 57}


class AuthorActivationMLP(nn.Module):
    """XPINN author-style activation: tanh/sin/cos with trainable scales."""

    def __init__(self, width: int, activation: str, depth: int = 6):
        super().__init__()
        if activation not in {"tanh", "sin", "cos"}:
            raise ValueError(activation)
        sizes = [1] + [width] * depth + [1]
        self.hidden = nn.ModuleList()
        self.scales = nn.ParameterList()
        self.activation = activation
        for in_features, out_features in zip(sizes[:-2], sizes[1:-1]):
            linear = nn.Linear(in_features, out_features)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.hidden.append(linear)
            self.scales.append(nn.Parameter(torch.tensor(0.05)))
        self.final = nn.Linear(sizes[-2], sizes[-1])
        nn.init.xavier_uniform_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x):
        activation = getattr(torch, self.activation)
        for linear, scale in zip(self.hidden, self.scales):
            x = activation(20.0 * scale * linear(x))
        return self.final(x)


def forcing(
    x: torch.Tensor, alpha: float = 2.0, k: float = 10.0 * np.pi
) -> torch.Tensor:
    exp_ax = torch.exp(alpha * x)
    g1 = 1.0 - x - x**2 + alpha * (x - x**2)
    g2 = -2.0 + 2.0 * alpha * (1.0 - 2.0 * x) + (alpha**2 - k**2) * (x - x**2)
    return exp_ax * (g2 * torch.sin(k * x) + 2.0 * k * g1 * torch.cos(k * x))


def residual(
    model: nn.Module,
    x: torch.Tensor,
    alpha: float,
    k: float,
    create_graph: bool = True,
) -> torch.Tensor:
    if not x.requires_grad:
        x = x.detach().clone().requires_grad_(True)
    u = model(x)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=create_graph)[
        0
    ]
    return u_xx - forcing(x, alpha=alpha, k=k)


def exact_solution(x: np.ndarray, alpha: float, k: float) -> np.ndarray:
    # Reference v2; forcing() and all training losses remain unchanged.
    x = np.asarray(x, dtype=np.float64)
    z = complex(alpha, k)

    def H(t):
        return (
            2.0
            * k
            * np.real(
                np.exp(z * t)
                * (
                    t * (1.0 - t) / z**2
                    - 2.0 * (1.0 - 2.0 * t) / z**3
                    - 6.0 / z**4
                )
            )
        )

    return (
        x * (1.0 - x) * np.exp(alpha * x) * np.sin(k * x)
        + H(x)
        - (1.0 - x) * H(0.0)
        - x * H(1.0)
    )


def allocate_points(total: int, count: int) -> list[int]:
    base, remainder = divmod(total, count)
    return [base + (1 if i < remainder else 0) for i in range(count)]
