from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from scipy.interpolate import griddata

from fbpinns.problems import Problem


class NavierStokesFBPINNProblem(Problem):
    """Steady lid-driven-cavity NS problem matching the manuscript configuration."""

    @staticmethod
    def init_params(
        data_file,
        reynolds=100.0,
        lid_coefficient=8.0,
        boundary_weight=100.0,
        points_per_side=100,
        test_side=120,
    ):
        data = np.loadtxt(data_file, comments="%")
        coords, values = data[:, :2], data[:, 2:5]
        axis = np.linspace(0.0, 1.0, test_side)
        xx, yy = np.meshgrid(axis, axis, indexing="ij")
        test_points = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
        test_values = griddata(coords, values, test_points, method="linear")
        missing = np.isnan(test_values).any(axis=1)
        if missing.any():
            test_values[missing] = griddata(
                coords, values, test_points[missing], method="nearest"
            )
        return {
            "dims": (3, 2),
            "reynolds": float(reynolds),
            "lid_coefficient": float(lid_coefficient),
            "boundary_weight": float(boundary_weight),
            "points_per_side": int(points_per_side),
            "test_values": jnp.asarray(test_values),
        }, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        x_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_phys = (
            (0, ()),
            (1, ()),
            (0, (0,)),
            (0, (1,)),
            (0, (0, 0)),
            (0, (1, 1)),
            (1, (0,)),
            (1, (1,)),
            (1, (0, 0)),
            (1, (1, 1)),
            (2, (0,)),
            (2, (1,)),
        )
        p = all_params["static"]["problem"]
        sides = domain.sample_boundaries(
            all_params, key, sampler, ((p["points_per_side"],),) * 4
        )
        x_boundary = jnp.concatenate(sides, axis=0)
        top = jnp.isclose(x_boundary[:, 1:2], 1.0)
        u_target = jnp.where(
            top,
            p["lid_coefficient"] * x_boundary[:, 0:1] * (1.0 - x_boundary[:, 0:1]),
            0.0,
        )
        velocity_target = jnp.concatenate((u_target, jnp.zeros_like(u_target)), axis=1)
        required_boundary = ((0, ()), (1, ()))
        x_pressure = jnp.array([[0.0, 0.0]])
        pressure_target = jnp.array([[0.0]])
        required_pressure = ((2, ()),)
        return [
            [x_phys, required_phys],
            [x_boundary, velocity_target, required_boundary],
            [x_pressure, pressure_target, required_pressure],
        ]

    @staticmethod
    def loss_fn(all_params, constraints):
        (_, u, v, ux, uy, uxx, uyy, vx, vy, vxx, vyy, px, py) = constraints[0]
        inv_re = 1.0 / all_params["static"]["problem"]["reynolds"]
        momentum_x = u * ux + v * uy + px - inv_re * (uxx + uyy)
        momentum_y = u * vx + v * vy + py - inv_re * (vxx + vyy)
        continuity = ux + vy
        _, velocity_target, ub, vb = constraints[1]
        _, pressure_target, pressure = constraints[2]
        weight = all_params["static"]["problem"]["boundary_weight"]
        return (
            jnp.mean(momentum_x**2)
            + jnp.mean(momentum_y**2)
            + jnp.mean(continuity**2)
            + weight * jnp.mean((ub - velocity_target[:, 0:1]) ** 2)
            + weight * jnp.mean((vb - velocity_target[:, 1:2]) ** 2)
            + weight * jnp.mean((pressure - pressure_target) ** 2)
        )

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        values = all_params["static"]["problem"]["test_values"]
        if values.shape[0] != x_batch.shape[0]:
            return jnp.zeros((x_batch.shape[0], 3))
        return values
