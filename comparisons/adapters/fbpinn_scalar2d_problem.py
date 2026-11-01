from __future__ import annotations

import jax.numpy as jnp

from fbpinns.problems import Problem


def _boundary_points(all_params, domain, key, sampler, points_per_side: int):
    sides = domain.sample_boundaries(
        all_params,
        key,
        sampler,
        ((points_per_side,),) * 4,
    )
    return jnp.concatenate(sides, axis=0)


class PoissonHFFBPINNProblem(Problem):
    """High-frequency manufactured Poisson problem used in the manuscript."""

    @staticmethod
    def init_params(
        k=14,
        sigma=0.1,
        centers=((0.30, 0.30), (0.70, 0.35), (0.55, 0.75), (0.80, 0.80)),
        amplitudes=(5.0, 5.0, 5.0, 5.0),
        boundary_weight=10000.0,
        points_per_side=100,
    ):
        return {
            "dims": (1, 2),
            "k": int(k),
            "sigma": float(sigma),
            "centers": jnp.asarray(centers),
            "amplitudes": jnp.asarray(amplitudes),
            "boundary_weight": float(boundary_weight),
            "points_per_side": int(points_per_side),
        }, {}

    @staticmethod
    def _packet_terms(all_params, x_batch):
        p = all_params["static"]["problem"]
        x, y = x_batch[:, 0:1], x_batch[:, 1:2]
        kpi = p["k"] * jnp.pi
        sinx, cosx = jnp.sin(kpi * x), jnp.cos(kpi * x)
        siny, cosy = jnp.sin(kpi * y), jnp.cos(kpi * y)
        s = sinx * siny
        sx, sy = kpi * cosx * siny, kpi * sinx * cosy
        sxx = -(kpi**2) * s
        syy = -(kpi**2) * s
        invs2 = 1.0 / (p["sigma"] ** 2)
        up = jnp.zeros_like(x)
        lap_up = jnp.zeros_like(x)
        for center, amplitude in zip(p["centers"], p["amplitudes"]):
            dx, dy = x - center[0], y - center[1]
            g = jnp.exp(-(dx * dx + dy * dy) * invs2)
            gx, gy = g * (-2.0 * dx * invs2), g * (-2.0 * dy * invs2)
            gxx = g * (4.0 * dx * dx * (invs2**2) - 2.0 * invs2)
            gyy = g * (4.0 * dy * dy * (invs2**2) - 2.0 * invs2)
            up = up + amplitude * g * s
            lap_up = lap_up + amplitude * (
                gxx * s + 2.0 * gx * sx + g * sxx + gyy * s + 2.0 * gy * sy + g * syy
            )
        return up, lap_up

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        x, y = x_batch[:, 0:1], x_batch[:, 1:2]
        up, _ = PoissonHFFBPINNProblem._packet_terms(all_params, x_batch)
        return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) + up

    @staticmethod
    def forcing(all_params, x_batch):
        x, y = x_batch[:, 0:1], x_batch[:, 1:2]
        u0 = jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
        _, lap_up = PoissonHFFBPINNProblem._packet_terms(all_params, x_batch)
        return 2.0 * (jnp.pi**2) * u0 - lap_up

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        x_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_phys = ((0, (0, 0)), (0, (1, 1)))
        points_per_side = all_params["static"]["problem"]["points_per_side"]
        x_boundary = _boundary_points(all_params, domain, key, sampler, points_per_side)
        u_boundary = PoissonHFFBPINNProblem.exact_solution(all_params, x_boundary)
        required_boundary = ((0, ()),)
        return [
            [x_phys, required_phys],
            [x_boundary, u_boundary, required_boundary],
        ]

    @staticmethod
    def loss_fn(all_params, constraints):
        x_phys, u_xx, u_yy = constraints[0]
        residual = -(u_xx + u_yy) - PoissonHFFBPINNProblem.forcing(all_params, x_phys)
        _, target, u_boundary = constraints[1]
        weight = all_params["static"]["problem"]["boundary_weight"]
        return jnp.mean(residual**2) + weight * jnp.mean((u_boundary - target) ** 2)


class ADRFBPINNProblem(Problem):
    """Manufactured 2D advection-diffusion-reaction manuscript problem."""

    @staticmethod
    def init_params(
        nu=1e-4,
        beta=100.0,
        cmin=1.0,
        cmax=1e4,
        xc=0.65,
        yc=0.35,
        sig_c=0.10,
        delta=0.35,
        x0=0.60,
        y0=0.40,
        sig_u=0.12,
        k=8,
        boundary_weight=10000.0,
        points_per_side=100,
    ):
        return {
            "dims": (1, 2),
            "nu": float(nu),
            "beta": float(beta),
            "cmin": float(cmin),
            "cmax": float(cmax),
            "xc": float(xc),
            "yc": float(yc),
            "sig_c": float(sig_c),
            "delta": float(delta),
            "x0": float(x0),
            "y0": float(y0),
            "sig_u": float(sig_u),
            "k": int(k),
            "boundary_weight": float(boundary_weight),
            "points_per_side": int(points_per_side),
        }, {}

    @staticmethod
    def _solution_terms(all_params, x_batch):
        p = all_params["static"]["problem"]
        x, y = x_batch[:, 0:1], x_batch[:, 1:2]
        sx, cx = jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x)
        sy, cy = jnp.sin(jnp.pi * y), jnp.cos(jnp.pi * y)
        u_low = sx * sy
        ux_low, uy_low = jnp.pi * cx * sy, jnp.pi * sx * cy
        lap_low = -2.0 * (jnp.pi**2) * u_low
        invs2 = 1.0 / (p["sig_u"] ** 2)
        dx, dy = x - p["x0"], y - p["y0"]
        g = jnp.exp(-(dx * dx + dy * dy) * invs2)
        frequency = p["k"] * jnp.pi
        c, s = jnp.cos(frequency * x), jnp.sin(frequency * x)
        gx, gy = g * (-2.0 * dx * invs2), g * (-2.0 * dy * invs2)
        gxx = g * (4.0 * dx * dx * (invs2**2) - 2.0 * invs2)
        gyy = g * (4.0 * dy * dy * (invs2**2) - 2.0 * invs2)
        cx_packet, cxx_packet = -frequency * s, -(frequency**2) * c
        u_packet = g * c
        ux_packet = gx * c + g * cx_packet
        uy_packet = gy * c
        lap_packet = gxx * c + 2.0 * gx * cx_packet + g * cxx_packet + gyy * c
        delta = p["delta"]
        return (
            u_low + delta * u_packet,
            ux_low + delta * ux_packet,
            uy_low + delta * uy_packet,
            lap_low + delta * lap_packet,
        )

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        return ADRFBPINNProblem._solution_terms(all_params, x_batch)[0]

    @staticmethod
    def coefficients(all_params, x_batch):
        p = all_params["static"]["problem"]
        x, y = x_batch[:, 0:1], x_batch[:, 1:2]
        bx = p["beta"] * (-(y - 0.5))
        by = p["beta"] * (x - 0.5)
        r2 = (x - p["xc"]) ** 2 + (y - p["yc"]) ** 2
        reaction = p["cmin"] + p["cmax"] * jnp.exp(-r2 / (p["sig_c"] ** 2))
        return bx, by, reaction

    @staticmethod
    def forcing(all_params, x_batch):
        p = all_params["static"]["problem"]
        u, ux, uy, lap = ADRFBPINNProblem._solution_terms(all_params, x_batch)
        bx, by, reaction = ADRFBPINNProblem.coefficients(all_params, x_batch)
        return -p["nu"] * lap + bx * ux + by * uy + reaction * u

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        x_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_phys = ((0, ()), (0, (0,)), (0, (1,)), (0, (0, 0)), (0, (1, 1)))
        points_per_side = all_params["static"]["problem"]["points_per_side"]
        x_boundary = _boundary_points(all_params, domain, key, sampler, points_per_side)
        u_boundary = ADRFBPINNProblem.exact_solution(all_params, x_boundary)
        required_boundary = ((0, ()),)
        return [
            [x_phys, required_phys],
            [x_boundary, u_boundary, required_boundary],
        ]

    @staticmethod
    def loss_fn(all_params, constraints):
        x_phys, u, ux, uy, uxx, uyy = constraints[0]
        p = all_params["static"]["problem"]
        bx, by, reaction = ADRFBPINNProblem.coefficients(all_params, x_phys)
        residual = (
            -p["nu"] * (uxx + uyy)
            + bx * ux
            + by * uy
            + reaction * u
            - ADRFBPINNProblem.forcing(all_params, x_phys)
        )
        _, target, u_boundary = constraints[1]
        weight = p["boundary_weight"]
        return jnp.mean(residual**2) + weight * jnp.mean((u_boundary - target) ** 2)
