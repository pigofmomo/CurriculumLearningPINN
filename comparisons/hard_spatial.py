"""Cumulative spatial mask with fixed Adam-stage transitions.

Uniform-window inspiration: Penwarden et al., JCP 493 (2023) 112464.
This is a spatial, fixed-budget adaptation, not a reproduction of time marching.
"""
import json
from pathlib import Path

import numpy as np


def stage_mask(distances, step, adam_iterations):
    distances = np.asarray(distances, dtype=int)
    ranks = np.unique(distances)
    if len(ranks) != 3 or adam_iterations < 3:
        raise ValueError(
            "This registered protocol requires exactly three spatial rings."
        )
    switches = (adam_iterations // 3, (2 * adam_iterations) // 3)
    stage = 1 + sum(step >= switch for switch in switches)
    return (distances <= ranks[stage - 1]).astype(float).tolist(), stage


def install(pinn_pro, cfg, output):
    """Install process-local hooks before dataset/network construction."""
    rc = cfg.reweight_config
    assert rc["low_fre_data_weight"] == 0 and rc.get("frame_data_weight", 0) == 0
    assert rc["reweight_causal_begin"] > cfg.adam_iterations + cfg.lbfgs_iterations
    assert rc["reweight_adaptive_begin"] > cfg.adam_iterations + cfg.lbfgs_iterations
    original_init = pinn_pro.PDEWeightedSamples.init_data
    original_step = pinn_pro.ModelWeightedSamples._train_step
    events = []
    output = Path(output)

    def update(data, step):
        weights, stage = stage_mask(
            data.geometry_partitions.distance_to_boundary, step, cfg.adam_iterations
        )
        data.pde_loss_weights = weights
        if not events or stage != events[-1]["stage"]:
            event = {
                "optimizer_step_before_update": int(step),
                "stage": stage,
                "active_regions": int(sum(weights)),
                "total_regions": len(weights),
                "region_mask": weights,
                "boundary_conditions": "all enforced throughout",
                "previous_rings": "remain active",
            }
            events.append(event)
            p = output / "hard_stage_events.json"
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(events, indent=2))
            tmp.replace(p)
            print("HARD_SPATIAL_STAGE", json.dumps(event), flush=True)

    def init_data(data, savepath=None):
        original_init(data, savepath=savepath)
        update(data, 0)

    def train_step(model, *args, **kwargs):
        update(model.data, model.train_state.step)
        return original_step(model, *args, **kwargs)

    pinn_pro.PDEWeightedSamples.init_data = init_data
    pinn_pro.ModelWeightedSamples._train_step = train_step
    return {
        "type": "fixed_budget_cumulative_spatial_hard_mask",
        "adam_stage_switches": [cfg.adam_iterations // 3, 2 * cfg.adam_iterations // 3],
        "lbfgs": "all regions active",
        "bridge": False,
        "source": "https://arxiv.org/html/2302.14227v2",
        "adaptation": "uniform moving-window mask; temporal order replaced by spatial rings; fixed iteration budget replaces adaptive loss tolerance",
        "normalization": "same global residual MSE as baseline, inactive regional residuals multiplied by zero",
        "initialization": "outer ring only from optimizer update zero",
    }
