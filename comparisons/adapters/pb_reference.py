"""Finite-difference reference for the fixed Poisson-Boltzmann example."""
import numpy as np
from scipy.fft import dstn, idstn


def solve(n):
    h = 2 / n
    q = np.linspace(-1, 1, n + 1)[1:-1]
    x, y = np.meshgrid(q, q)
    f = (
        5 * (20 + x * x + y * y) * np.sin(2 * np.pi * x) * np.sin(4 * np.pi * y)
        - 25 * 0.2
    )
    lam = 4 / h**2 * np.sin(np.pi * np.arange(1, n) / (2 * n)) ** 2
    v = idstn(
        dstn(f, type=1, norm="ortho") / (lam[:, None] + lam[None, :] + 25),
        type=1,
        norm="ortho",
    )
    u = np.full((n + 1, n + 1), 0.2)
    u[1:-1, 1:-1] += v
    return u


def generate(path):
    coarse = solve(1024)[::4, ::4]
    fine = solve(2048)[::8, ::8]
    assert np.linalg.norm(fine - coarse) / np.linalg.norm(fine) < 3e-4
    q = np.linspace(-1, 1, 257)
    x, y = np.meshgrid(q, q)
    np.savez_compressed(
        path,
        xy=np.c_[x.ravel(), y.ravel()],
        truth=fine.ravel(),
        coarse_truth=coarse.ravel(),
    )
