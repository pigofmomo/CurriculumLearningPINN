"""Apply SciPy L-BFGS-B to the exact official FBPINN training objective."""
from pathlib import Path
import json, time, os, pickle
import numpy as np
import jax
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize
from fbpinns.trainers import FBPINN_loss


def refine(
    active,
    fixed,
    static,
    takess,
    constraints,
    model_fns,
    jmapss,
    loss_fn,
    iterations,
    output,
):
    output = Path(output)
    flat, unravel = ravel_pytree(active)

    @jax.jit
    def value_grad(flat):
        def loss(z):
            return FBPINN_loss(
                unravel(z),
                fixed,
                static,
                takess,
                constraints,
                model_fns,
                jmapss,
                loss_fn,
            )

        return jax.value_and_grad(loss)(flat)

    calls = 0
    start = time.monotonic()
    history = (output / "lbfgs_history.jsonl").open("w")

    def fun(x):
        nonlocal calls
        val, grad = value_grad(jax.numpy.asarray(x, dtype=flat.dtype))
        calls += 1
        val = float(val)
        grad = np.asarray(grad, dtype=np.float64)
        if not np.isfinite(val) or not np.all(np.isfinite(grad)):
            raise FloatingPointError("Nonfinite FBPINN L-BFGS objective/gradient")
        if calls == 1 or calls % 100 == 0:
            row = {
                "evaluation": calls,
                "objective": val,
                "wall_seconds": time.monotonic() - start,
            }
            history.write(json.dumps(row) + "\n")
            history.flush()
            print("LBFGS", json.dumps(row), flush=True)
        return val, grad

    iterations_done = 0

    def keep(x):
        nonlocal iterations_done
        iterations_done += 1
        if iterations_done == 1 or iterations_done % 100 == 0:
            p = output / "lbfgs_active_checkpoint.pkl"
            tmp = p.with_suffix(".tmp")
            with tmp.open("wb") as f:
                pickle.dump(
                    dict(
                        iteration=iterations_done,
                        active=jax.tree_util.tree_map(
                            np.asarray, unravel(jax.numpy.asarray(x, dtype=flat.dtype))
                        ),
                    ),
                    f,
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, p)

    result = minimize(
        fun,
        np.asarray(flat, dtype=np.float64),
        jac=True,
        callback=keep,
        method="L-BFGS-B",
        options={
            "maxiter": iterations,
            "maxfun": int(iterations * 1.25),
            "maxcor": 100,
            "ftol": 0.0,
            "gtol": 1e-8,
            "maxls": 50,
        },
    )
    history.close()
    report = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "actual_iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "objective": float(result.fun),
        "wall_seconds": time.monotonic() - start,
        "dtype": str(flat.dtype),
        "optimizer": "SciPy L-BFGS-B, unbounded; exact original FBPINN objective; JAX gradients",
    }
    (output / "lbfgs_result.json").write_text(json.dumps(report, indent=2))
    print("LBFGS_RESULT", json.dumps(report), flush=True)
    return unravel(jax.numpy.asarray(result.x, dtype=flat.dtype))
