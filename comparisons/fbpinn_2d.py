"""Three overlapping strip or square-ring FBPINN subdomains."""
from pathlib import Path
import os, sys, json, time, datetime, hashlib, pickle, traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "adapters"))
import jax

if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree_util.tree_map
assert not jax.config.x64_enabled
import jax.numpy as jnp
import numpy as np
import fbpinn_scalar2d as scalar
import fbpinn_ns as ns
from fbpinns.trainers import FBPINNTrainer
from fbpinns.analysis import FBPINN_solution
from fbpinns import plot_trainer
from ring_fbpinn import RingDecomposition
import ring_refine

DATA = ROOT / "experiments/ns2d/data/lid_driven_a8.dat"
ns.DATA_FILE = DATA


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, x):
    t = p.with_suffix(".tmp")
    t.write_text(json.dumps(x, indent=2, ensure_ascii=False))
    t.replace(p)


def evaluate_saved(args, c, params, out, smoke):
    if args.case == "ns":
        data = np.loadtxt(DATA, comments="%")
        data = data[np.linspace(0, len(data) - 1, 49, dtype=int)] if smoke else data
        x = data[:, :2].astype(np.float32)
        ref = data[:, 2:5].astype(np.float64)
    else:
        axis = np.asarray(jnp.linspace(0, 1, args.num_test_side))
        xx, yy = np.meshgrid(axis, axis, indexing="ij")
        x = np.column_stack((xx.ravel(), yy.ravel()))
        ref = np.asarray(
            c.problem.exact_solution(params, jnp.asarray(x)), dtype=np.float64
        )
    active = jnp.ones(3, dtype=int)
    pred = []
    for start in range(0, len(x), 2500):
        pred.append(
            np.asarray(
                FBPINN_solution(c, params, active, jnp.asarray(x[start : start + 2500]))
            )
        )
    pred = np.concatenate(pred)
    metrics = {}
    for i, key in enumerate(["u", "v", "p"] if args.case == "ns" else ["u"]):
        error = pred[:, i].astype(np.float64) - ref[:, i]
        metrics.update(
            {
                f"{key}_l2_relative": float(
                    np.linalg.norm(error) / np.linalg.norm(ref[:, i])
                ),
                f"{key}_mse": float(np.mean(error**2)),
                f"{key}_max_abs": float(np.max(np.abs(error))),
            }
        )
    np.savez_compressed(out / "predictions.npz", x=x, reference=ref, prediction=pred)
    return metrics, len(x)


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    case = job["case"]
    assert (
        case in ["poisson", "adr", "ns"]
        and job["seed"] in [0, 1, 2]
        and job["subdomains"] == 3
    )
    source = ROOT / job["source_config"]
    cfg = json.loads(source.read_text())
    assert cfg["seed"] == job["seed"]
    out = ROOT / job["output"]
    out.mkdir(parents=True, exist_ok=True)
    smoke = bool(job.get("validation_only", False))
    module = ns if case == "ns" else scalar
    adam = 2 if smoke else cfg["adam_iterations"]
    lbfgs = 2 if smoke else cfg["lbfgs_iterations"]
    args = SimpleNamespace(
        case=case,
        subdomains=3,
        num_domain=16 if smoke else cfg["num_domain"],
        num_boundary=16 if smoke else cfg["num_boundary"],
        num_test_side=7 if smoke else 250,
        test_side=7 if smoke else 120,
        steps=adam,
        overlap_factor=1.9,
        eval_chunk_size=2500,
    )
    depth = 6 if case != "adr" else 5
    width = 34 if case == "ns" else 28
    outputs = 3 if case == "ns" else 1
    expected = 3 * (
        (2 + 1) * width + (depth - 1) * (width + 1) * width + (width + 1) * outputs
    )
    m = {
        "job": job,
        "status": "running",
        "pid": os.getpid(),
        "started_at": datetime.datetime.now().astimezone().isoformat(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jax": jax.__version__,
        "gpu": str(jax.devices()[0]),
        "source_config_sha256": sha(source),
        "runner_sha256": sha(Path(__file__)),
        "ring_module_sha256": sha(HERE / "ring_fbpinn.py"),
        "refine_sha256": sha(HERE / "ring_refine.py"),
        "source_snapshot_sha256": {
            p.name: sha(p) for p in (HERE / "adapters").glob("*.py")
        },
        "protocol": {
            "training_dtype": "float32",
            "subdomains": 3,
            "partition": job.get("partition"),
            "overlap_factor": 1.9,
            "width": width,
            "depth": depth,
            "expected_parameter_count": expected,
            "adam_iterations": adam,
            "lbfgs_max_iterations": lbfgs,
            "adam_learning_rate_initial": cfg["learning_rate"],
            "adam_learning_rate_decay": "legacy StepDecay, factor0.9 every1000 updates",
            "num_domain": args.num_domain,
            "num_boundary": args.num_boundary,
            "sampler": "legacy grid",
            "change": "replace three vertical strips by three concentric square rings; preserve frozen FBPINN objective and optimizer budget",
            "evaluation": "legacy 250x250 grid scalar / supplied NS data coordinates",
            "comparison_limits": [
                "JAX/SciPy vs main PyTorch optimizers; same iteration caps do not equal FLOPs",
                "legacy grid differs from DeepXDE coordinate conventions",
                "three concentric square-ring support geometry; not rectangular strip decomposition",
                "legacy NS uses full-boundary component means; main method separates lid and other walls",
            ],
        },
    }
    if case == "ns":
        m["reference_data_sha256"] = sha(DATA)
    save(out / "run_manifest.json", m)
    start = time.monotonic()
    try:
        os.chdir(out)
        c = module.make_constants(args, job["seed"], job["id"])
        c.revision_lbfgs_iterations = lbfgs
        c.revision_output = str(out)
        if job.get("partition") == "rings":
            c.decomposition = RingDecomposition
            c.decomposition_init_kwargs = {}
            sys.modules["fbpinn_refine"] = ring_refine
        else:
            import fbpinn_refine as strip_refine

            sys.modules["fbpinn_refine"] = strip_refine
        c.model_save_freq = 500 if not smoke else 2
        plot_trainer.plot = lambda *args, **kwargs: None
        params = FBPINNTrainer(c).train()
        metrics, n = evaluate_saved(args, c, params, out, smoke)
        parameter_count = module.count_trainable(params)
        assert parameter_count == expected
        with (out / "final_parameters.pkl").open("wb") as f:
            pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)
        result = json.loads((out / "lbfgs_result.json").read_text())
        m.update(
            status="complete"
            if all(np.isfinite(v) for v in metrics.values())
            else "nonfinite",
            metrics=metrics,
            num_test=n,
            parameter_count=parameter_count,
            actual_final_step=adam + result["actual_iterations"],
            lbfgs_result=result,
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
