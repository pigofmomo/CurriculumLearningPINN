from __future__ import annotations

import jax.numpy as jnp

from fbpinns.problems import Problem


class StepDecay:
    """Pickle-friendly staircase decay matching the manuscript ODE schedule."""

    def __init__(
        self, initial: float = 1e-3, transition_steps: int = 2000, decay: float = 0.9
    ):
        self.initial = initial
        self.transition_steps = transition_steps
        self.decay = decay

    def __call__(self, count):
        exponent = count // self.transition_steps
        return self.initial * (self.decay**exponent)

    def __repr__(self) -> str:
        return (
            f"StepDecay(initial={self.initial}, transition_steps={self.transition_steps}, "
            f"decay={self.decay})"
        )


class ODELowFBPINNProblem(Problem):
    """Manufactured ODE-low problem used by the manuscript experiments."""

    @staticmethod
    def init_params(alpha=2.0, k=10.0 * jnp.pi, boundary_weight=100.0):
        return {
            "dims": (1, 1),
            "alpha": float(alpha),
            "k": float(k),
            "boundary_weight": float(boundary_weight),
        }, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        x_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_phys = ((0, ()), (0, (0, 0)))
        x_boundary = jnp.array([[0.0], [1.0]])
        u_boundary = jnp.zeros((2, 1))
        required_boundary = ((0, ()),)
        return [
            [x_phys, required_phys],
            [x_boundary, u_boundary, required_boundary],
        ]

    @staticmethod
    def forcing(all_params, x):
        p = all_params["static"]["problem"]
        alpha, k = p["alpha"], p["k"]
        exp_ax = jnp.exp(alpha * x)
        g1 = 1.0 - x - x**2 + alpha * (x - x**2)
        g2 = -2.0 + 2.0 * alpha * (1.0 - 2.0 * x) + (alpha**2 - k**2) * (x - x**2)
        return exp_ax * (g2 * jnp.sin(k * x) + 2.0 * k * g1 * jnp.cos(k * x))

    @staticmethod
    def loss_fn(all_params, constraints):
        x_phys, u_phys, u_xx = constraints[0]
        residual = u_xx - ODELowFBPINNProblem.forcing(all_params, x_phys)
        _, target, u_boundary = constraints[1]
        boundary_weight = all_params["static"]["problem"]["boundary_weight"]
        # 两个边界各乘 100，与原 DeepXDE loss_weights=(1,100,100) 对齐。
        return jnp.mean(residual**2) + boundary_weight * jnp.sum(
            (u_boundary - target) ** 2
        )

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        p = all_params["static"]["problem"]
        x = x_batch[:, 0:1]
        # Reference v2; the source and boundary terms in loss_fn are unchanged.
        alpha, k = p["alpha"], p["k"]
        z = alpha + 1j * k

        def H(t):
            return (
                2.0
                * k
                * jnp.real(
                    jnp.exp(z * t)
                    * (
                        t * (1.0 - t) / z**2
                        - 2.0 * (1.0 - 2.0 * t) / z**3
                        - 6.0 / z**4
                    )
                )
            )

        return (
            x * (1.0 - x) * jnp.exp(alpha * x) * jnp.sin(k * x)
            + H(x)
            - (1.0 - x) * H(0.0)
            - x * H(1.0)
        )
