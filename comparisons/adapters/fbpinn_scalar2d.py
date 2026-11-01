from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
from fbpinns.analysis import FBPINN_solution, load_model
from fbpinns.constants import Constants, get_subdomain_ws
from fbpinns.decompositions import RectangularDecompositionND
from fbpinns.domains import RectangularDomainND
from fbpinns.networks import FCN
from fbpinns.trainers import FBPINNTrainer, FBPINN_model, get_inputs
from fbpinn_scalar2d_problem import ADRFBPINNProblem, PoissonHFFBPINNProblem

CASE_SPECS = {
    "poisson": {
        "problem": PoissonHFFBPINNProblem,
        "depth": 6,
        "widths": {2: 35, 3: 28},
        "learning_rate": 1e-3,
        "default_steps": 20000,
    },
    "adr": {
        "problem": ADRFBPINNProblem,
        "depth": 5,
        "widths": {2: 35, 3: 28},
        "learning_rate": 5e-4,
        "default_steps": 10000,
    },
}


class StepDecay:
    def __init__(
        self, initial: float, transition_steps: int = 1000, decay: float = 0.9
    ):
        self.initial = initial
        self.transition_steps = transition_steps
        self.decay = decay

    def __call__(self, count):
        return self.initial * (self.decay ** (count // self.transition_steps))

    def __repr__(self):
        return (
            f"StepDecay(initial={self.initial}, transition_steps={self.transition_steps}, "
            f"decay={self.decay})"
        )


def count_trainable(all_params) -> int:
    return int(
        sum(
            np.asarray(leaf).size
            for leaf in jax.tree_util.tree_leaves(all_params["trainable"])
        )
    )


def make_constants(args: argparse.Namespace, seed: int, run_name: str) -> Constants:
    spec = CASE_SPECS[args.case]
    centers = [np.linspace(0.0, 1.0, args.subdomains), np.array([0.5])]
    widths = [get_subdomain_ws([centers[0]], args.overlap_factor)[0], np.array([1.9])]
    width = spec["widths"][args.subdomains]
    depth = spec["depth"]
    steps = args.steps if args.steps is not None else spec["default_steps"]
    side = int(round(args.num_domain**0.5))
    return Constants(
        run=run_name,
        domain=RectangularDomainND,
        domain_init_kwargs={"xmin": np.array([0.0, 0.0]), "xmax": np.array([1.0, 1.0])},
        problem=spec["problem"],
        problem_init_kwargs={"points_per_side": args.num_boundary // 4},
        decomposition=RectangularDecompositionND,
        decomposition_init_kwargs={
            "subdomain_xs": centers,
            "subdomain_ws": widths,
            "unnorm": (0.0, 1.0),
        },
        network=FCN,
        network_init_kwargs={"layer_sizes": [2] + [width] * depth + [1]},
        ns=((side, side),),
        n_test=(args.num_test_side, args.num_test_side),
        n_steps=steps,
        sampler="grid",
        optimiser=optax.adam,
        optimiser_kwargs={"learning_rate": StepDecay(spec["learning_rate"])},
        seed=seed,
        summary_freq=max(1, min(1000, steps)),
        test_freq=steps,
        model_save_freq=steps,
        show_figures=False,
        save_figures=False,
        clear_output=False,
    )
