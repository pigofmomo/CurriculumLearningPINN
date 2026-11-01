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


def poisson_exact(x: torch.Tensor) -> torch.Tensor:
    xx, yy = x[:, 0:1], x[:, 1:2]
    kpi = 14.0 * np.pi
    value = torch.sin(np.pi * xx) * torch.sin(np.pi * yy)
    centers = ((0.30, 0.30), (0.70, 0.35), (0.55, 0.75), (0.80, 0.80))
    oscillation = torch.sin(kpi * xx) * torch.sin(kpi * yy)
    for cx, cy in centers:
        g = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (0.1**2))
        value = value + 5.0 * g * oscillation
    return value


def poisson_forcing(x: torch.Tensor) -> torch.Tensor:
    xx, yy = x[:, 0:1], x[:, 1:2]
    kpi = 14.0 * np.pi
    sinx, cosx = torch.sin(kpi * xx), torch.cos(kpi * xx)
    siny, cosy = torch.sin(kpi * yy), torch.cos(kpi * yy)
    s = sinx * siny
    sx, sy = kpi * cosx * siny, kpi * sinx * cosy
    sxx, syy = -(kpi**2) * s, -(kpi**2) * s
    lap_packet = torch.zeros_like(xx)
    invs2 = 1.0 / (0.1**2)
    for cx, cy in ((0.30, 0.30), (0.70, 0.35), (0.55, 0.75), (0.80, 0.80)):
        dx, dy = xx - cx, yy - cy
        g = torch.exp(-(dx * dx + dy * dy) * invs2)
        gx, gy = g * (-2.0 * dx * invs2), g * (-2.0 * dy * invs2)
        gxx = g * (4.0 * dx * dx * (invs2**2) - 2.0 * invs2)
        gyy = g * (4.0 * dy * dy * (invs2**2) - 2.0 * invs2)
        lap_packet = lap_packet + 5.0 * (
            gxx * s + 2.0 * gx * sx + g * sxx + gyy * s + 2.0 * gy * sy + g * syy
        )
    u0 = torch.sin(np.pi * xx) * torch.sin(np.pi * yy)
    return 2.0 * (np.pi**2) * u0 - lap_packet


def adr_coefficients(x: torch.Tensor):
    xx, yy = x[:, 0:1], x[:, 1:2]
    bx, by = 100.0 * (-(yy - 0.5)), 100.0 * (xx - 0.5)
    reaction = 1.0 + 1e4 * torch.exp(
        -((xx - 0.65) ** 2 + (yy - 0.35) ** 2) / (0.10**2)
    )
    return bx, by, reaction


def adr_exact(x: torch.Tensor) -> torch.Tensor:
    return adr_solution_terms(x)[0]


def adr_forcing(x: torch.Tensor) -> torch.Tensor:
    u, ux, uy, lap = adr_solution_terms(x)
    bx, by, reaction = adr_coefficients(x)
    return -1e-4 * lap + bx * ux + by * uy + reaction * u


def adr_solution_terms(x: torch.Tensor):
    xx, yy = x[:, 0:1], x[:, 1:2]
    sx, cx = torch.sin(np.pi * xx), torch.cos(np.pi * xx)
    sy, cy = torch.sin(np.pi * yy), torch.cos(np.pi * yy)
    u_low = sx * sy
    ux_low, uy_low = np.pi * cx * sy, np.pi * sx * cy
    lap_low = -2.0 * (np.pi**2) * u_low
    invs2 = 1.0 / (0.12**2)
    dx, dy = xx - 0.60, yy - 0.40
    g = torch.exp(-(dx * dx + dy * dy) * invs2)
    frequency = 8.0 * np.pi
    c, s = torch.cos(frequency * xx), torch.sin(frequency * xx)
    gx, gy = g * (-2.0 * dx * invs2), g * (-2.0 * dy * invs2)
    gxx = g * (4.0 * dx * dx * (invs2**2) - 2.0 * invs2)
    gyy = g * (4.0 * dy * dy * (invs2**2) - 2.0 * invs2)
    cx_packet, cxx_packet = -frequency * s, -(frequency**2) * c
    packet = g * c
    ux_packet, uy_packet = gx * c + g * cx_packet, gy * c
    lap_packet = gxx * c + 2.0 * gx * cx_packet + g * cxx_packet + gyy * c
    return (
        u_low + 0.35 * packet,
        ux_low + 0.35 * ux_packet,
        uy_low + 0.35 * uy_packet,
        lap_low + 0.35 * lap_packet,
    )
