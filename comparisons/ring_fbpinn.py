"""C2-smooth square-ring partition of unity for the frozen official FBPINN trainer."""
import jax
import jax.numpy as jnp
import numpy as np
from fbpinns.decompositions import Decomposition
from fbpinns.decompositions_base import inside_models_batch, inside_points_batch


def smooth(t):
    t = jnp.clip(t, 0.0, 1.0)
    return jnp.clip(t * t * t * (10.0 + t * (-15.0 + 6.0 * t)), 0.0, 1.0)


def windows(x):
    def square(a):
        return jnp.prod(
            smooth((x - (a - 0.05)) / 0.1) * smooth(((1 - a + 0.05) - x) / 0.1), axis=-1
        )

    a, b = square(0.2), square(0.4)
    return jnp.stack([1 - a, a - b, b], axis=-1)


class RingDecomposition(Decomposition):
    @staticmethod
    def init_params(**kwargs):
        return (
            dict(
                m=3,
                xd=2,
                subdomain=dict(
                    index=jnp.arange(3),
                    halfwidth=jnp.array([0.5, 0.35, 0.15]),
                    pou=jnp.zeros((3, 1)),
                ),
                xmins0=np.array([[0, 0], [0.2, 0.2], [0.4, 0.4]]),
                xmaxs0=np.array([[1, 1], [0.8, 0.8], [0.6, 0.6]]),
            ),
            {},
        )

    @staticmethod
    def norm_fn(params, x):
        return (x - 0.5) / params["static"]["decomposition"]["subdomain"]["halfwidth"]

    @staticmethod
    def unnorm_fn(params, u):
        return u

    @staticmethod
    def window_fn(params, x):
        return windows(x)[params["static"]["decomposition"]["subdomain"]["index"]][None]

    @staticmethod
    def _inside(params, x, ims):
        return windows(x)[:, ims] > 0

    @staticmethod
    def inside_points(params, x):
        return inside_points_batch(
            {}, x, jnp.arange(3), min(100000, len(x)), RingDecomposition._inside
        )

    @staticmethod
    def inside_models(params, x, ims):
        return inside_models_batch(
            {}, x, ims, min(100000, len(x)), RingDecomposition._inside
        )

    @staticmethod
    def plot(*args, **kwargs):
        return None
