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
from fbpinns.trainers import FBPINNTrainer
from fbpinn_ode_low_problem import ODELowFBPINNProblem, StepDecay

LOCAL_WIDTHS = {5: 44}


def count_trainable(all_params) -> int:
    return int(
        sum(
            np.asarray(leaf).size
            for leaf in jax.tree_util.tree_leaves(all_params["trainable"])
        )
    )


def make_constants(args: argparse.Namespace, seed: int, run_name: str) -> Constants:
    centers = [np.linspace(0.1, 0.9, 5)]
    widths = [np.full(5, 0.2 * args.overlap_factor)]
    local_width = LOCAL_WIDTHS[args.subdomains]
    if args.case == "ode_low":
        alpha, k, boundary_weight = 2.0, 10.0 * np.pi, 100.0
    else:
        alpha, k, boundary_weight = 3.0, 20.0 * np.pi, 1000.0
    return Constants(
        run=run_name,
        domain=RectangularDomainND,
        domain_init_kwargs={"xmin": np.array([0.0]), "xmax": np.array([1.0])},
        problem=ODELowFBPINNProblem,
        problem_init_kwargs={
            "alpha": alpha,
            "k": k,
            "boundary_weight": boundary_weight,
        },
        decomposition=RectangularDecompositionND,
        decomposition_init_kwargs={
            "subdomain_xs": centers,
            "subdomain_ws": widths,
            "unnorm": (0.0, 1.0),
        },
        network=FCN,
        network_init_kwargs={"layer_sizes": [1] + [local_width] * 6 + [1]},
        ns=((args.num_domain,),),
        n_test=(args.num_test,),
        n_steps=args.steps,
        sampler="grid",
        optimiser=optax.adam,
        optimiser_kwargs={"learning_rate": StepDecay()},
        seed=seed,
        summary_freq=max(1, min(1000, args.steps)),
        test_freq=max(1, min(1000, args.steps)),
        model_save_freq=args.steps,
        show_figures=False,
        save_figures=False,
        clear_output=False,
    )
