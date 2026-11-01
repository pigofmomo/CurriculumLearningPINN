"""Five overlapping ODE subdomains with the paper training budgets."""
from pathlib import Path
import os, sys, json, datetime, time, hashlib, traceback, pickle

O = Path(__file__).resolve().parent
R = O.parent
sys.path.insert(0, str(O / "adapters"))
import jax

# Equivalent compatibility alias removed from recent JAX versions.
if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree_util.tree_map
import jax.numpy as jnp
import numpy as np
from types import SimpleNamespace
from fbpinn_ode_low import make_constants, count_trainable
from fbpinns.trainers import FBPINNTrainer
from fbpinns.analysis import FBPINN_solution


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, v):
    t = p.with_suffix(".tmp")
    t.write_text(json.dumps(v, indent=2, ensure_ascii=False))
    t.replace(p)


def exact(x, alpha, k):
    x = np.asarray(x, dtype=np.float64)
    z = complex(alpha, k)

    def H(t):
        return (
            2
            * k
            * np.real(
                np.exp(z * t)
                * (t * (1 - t) / z**2 - 2 * (1 - 2 * t) / z**3 - 6 / z**4)
            )
        )

    return (
        x * (1 - x) * np.exp(alpha * x) * np.sin(k * x)
        + H(x)
        - (1 - x) * H(0)
        - x * H(1)
    )


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    source = R / job["source_config"]
    cfg = json.loads(source.read_text())
    assert job["seed"] == cfg["seed"]
    out = R / job["output"]
    out.mkdir(parents=True, exist_ok=True)
    smoke = job.get("validation_only", False)
    adam = 3 if smoke else cfg["adam_iterations"]
    lbfgs = 2 if smoke else cfg["lbfgs_iterations"]
    args = SimpleNamespace(
        case=job["case"],
        subdomains=job["subdomains"],
        num_domain=20 if smoke else cfg["num_domain"],
        num_test=50 if smoke else cfg["num_test"],
        steps=adam,
        overlap_factor=1.9,
    )
    assert job["case"] in ["ode_low", "ode_high"]
    m = {
        "job": job,
        "status": "running",
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_at": datetime.datetime.now().astimezone().isoformat(),
        "jax": jax.__version__,
        "gpu": str(jax.devices()[0]),
        "source_config_sha256": sha(source),
        "runner_sha256": sha(Path(__file__)),
        "source_hash_manifest_sha256": sha(R / "third_party/FBPINNs_LICENSE"),
        "refine_sha256": sha(O / "fbpinn_refine.py"),
        "reference_version": "ode_fixed_source_reference_v2_20260911",
        "protocol": {
            "adam_iterations": adam,
            "lbfgs_max_iterations": lbfgs,
            "overlap_factor": 1.9,
            "num_domain": args.num_domain,
            "adam_learning_rate": cfg["learning_rate"],
            "adam_decay": None,
            "subdomains": args.subdomains,
            "optimizer_comparison": "main iteration limits, official FBPINN architecture and main constant Adam learning rate; SciPy L-BFGS-B differs from PyTorch implementation",
            "legacy_runtime": "JAX 0.4.23 on V100; current runtime separately recorded",
        },
    }
    save(out / "run_manifest.json", m)
    start = time.monotonic()
    try:
        os.chdir(out)
        c = make_constants(args, job["seed"], job["id"])
        c.revision_lbfgs_iterations = lbfgs
        c.revision_output = str(out)
        c.optimiser_kwargs = {"learning_rate": cfg["learning_rate"]}
        params = FBPINNTrainer(c).train()
        x = np.linspace(0, 1, 1000, dtype=np.float32)[:, None]
        pred = np.asarray(
            FBPINN_solution(
                c, params, jnp.ones(args.subdomains, dtype=int), jnp.asarray(x)
            )
        )
        alpha, k = (2.0, 10 * np.pi) if job["case"] == "ode_low" else (3.0, 20 * np.pi)
        true = exact(x, alpha, k)
        err = pred - true
        metrics = dict(
            u_l2_relative=float(np.linalg.norm(err) / np.linalg.norm(true)),
            u_mse=float(np.mean(err**2)),
            u_max_abs=float(np.max(np.abs(err))),
        )
        np.savez_compressed(
            out / "predictions.npz", x=x, reference=true, prediction=pred
        )
        with (out / "final_parameters.pkl").open("wb") as f:
            pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)
        result = (
            json.loads((out / "lbfgs_result.json").read_text())
            if lbfgs
            else {"actual_iterations": 0}
        )
        m.update(
            status="complete"
            if all(np.isfinite(v) for v in metrics.values())
            else "nonfinite",
            metrics=metrics,
            actual_final_step=adam + result["actual_iterations"],
            lbfgs_result=result,
            parameter_count=count_trainable(params),
        )
    except FloatingPointError:
        m.update(status="nonfinite", traceback=traceback.format_exc())
        traceback.print_exc()
    except BaseException:
        m.update(status="failed", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        m.update(
            wall_seconds=time.monotonic() - start,
            finished_at=datetime.datetime.now().astimezone().isoformat(),
        )
        save(out / "run_manifest.json", m)
    print("RESULT", json.dumps(m), flush=True)
    return 0 if m["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
