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
from fbpinns.analysis import FBPINN_solution
from fbpinns.constants import Constants, get_subdomain_ws
from fbpinns.decompositions import RectangularDecompositionND
from fbpinns.domains import RectangularDomainND
from fbpinns.networks import FCN
from fbpinns.trainers import FBPINNTrainer, FBPINN_model, get_inputs
from fbpinns import plot_trainer
from fbpinn_ns_problem import NavierStokesFBPINNProblem

DATA_FILE = (
    Path(__file__).resolve().parents[2] / "experiments/ns2d/data/lid_driven_a8.dat"
)
LOCAL_WIDTHS = {2: 42, 3: 34}


class StepDecay:
    def __init__(self, initial=1e-3, transition_steps=1000, decay=0.9):
        self.initial = initial
        self.transition_steps = transition_steps
        self.decay = decay

    def __call__(self, count):
        return self.initial * self.decay ** (count // self.transition_steps)

    def __repr__(self):
        return f"StepDecay({self.initial}, {self.transition_steps}, {self.decay})"


def count_trainable(all_params):
    return int(
        sum(
            np.asarray(leaf).size
            for leaf in jax.tree_util.tree_leaves(all_params["trainable"])
        )
    )


def make_constants(args, seed, run_name):
    centers = [np.linspace(0.0, 1.0, args.subdomains), np.array([0.5])]
    widths = [get_subdomain_ws([centers[0]], args.overlap_factor)[0], np.array([1.9])]
    width = LOCAL_WIDTHS[args.subdomains]
    side = int(round(args.num_domain**0.5))
    return Constants(
        run=run_name,
        domain=RectangularDomainND,
        domain_init_kwargs={"xmin": np.array([0.0, 0.0]), "xmax": np.array([1.0, 1.0])},
        problem=NavierStokesFBPINNProblem,
        problem_init_kwargs={
            "data_file": str(DATA_FILE),
            "points_per_side": args.num_boundary // 4,
            "test_side": args.test_side,
        },
        decomposition=RectangularDecompositionND,
        decomposition_init_kwargs={
            "subdomain_xs": centers,
            "subdomain_ws": widths,
            "unnorm": (0.0, 1.0),
        },
        network=FCN,
        network_init_kwargs={"layer_sizes": [2] + [width] * 6 + [3]},
        ns=((side, side),),
        n_test=(args.test_side, args.test_side),
        n_steps=args.steps,
        sampler="grid",
        optimiser=optax.adam,
        optimiser_kwargs={"learning_rate": StepDecay()},
        seed=seed,
        summary_freq=max(1, min(500, args.steps)),
        test_freq=args.steps,
        model_save_freq=args.steps,
        show_figures=False,
        save_figures=False,
        clear_output=False,
    )
