"""Reusable PINN runners and utilities. / 可复用的PINN训练器和工具。"""

from .pinn import PINN, FunctionInterpolator, PINNWeightedSamples
from .model import ModelWeightedSamples
from .data import PDEWeightedSamples
from .domain_decomp import Pointset1D, Pointset2D