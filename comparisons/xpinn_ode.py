"""Five-subdomain XPINN with explicit optional normal-derivative penalty."""
from pathlib import Path
import sys, os, json, time, datetime, hashlib, traceback, random

R = Path(__file__).resolve().parents[1]
O = R / "comparisons"
sys.path.insert(0, str(O / "adapters"))
import numpy as np
import torch
from xpinn_ode_low import (
    AuthorActivationMLP,
    forcing,
    residual,
    exact_solution,
    allocate_points,
    LOCAL_WIDTHS,
)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, v):
    t = p.with_suffix(".tmp")
    t.write_text(json.dumps(v, indent=2, ensure_ascii=False))
    t.replace(p)


class LocalCoordinates(torch.nn.Module):
    def __init__(self, network, left, right):
        super().__init__()
        self.network = network
        self.left = left
        self.right = right

    def forward(self, x):
        return self.network(2 * (x - self.left) / (self.right - self.left) - 1)


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    cfgpath = R / job["source_config"]
    cfg = json.loads(cfgpath.read_text())
    assert cfg["seed"] == job["seed"] and job["case"] in ("ode_low", "ode_high")
    out = R / job["output"]
    out.mkdir(parents=True, exist_ok=True)
    smoke = job.get("validation_only", False)
    device = torch.device(job.get("device", "cpu"))
    torch.set_num_threads(2)
    seed = job["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    M = job["subdomains"]
    width = 44
    assert M == 5
    alpha, k = (2.0, 10 * np.pi) if job["case"] == "ode_low" else (3.0, 20 * np.pi)
    adam = 5 if smoke else cfg["adam_iterations"]
    lbfgs = 3 if smoke else cfg["lbfgs_iterations"]
    N = 20 if smoke else cfg["num_domain"]
    flux_weight = job["flux_weight"]
    assert flux_weight in (0.0, 20.0)
    m = {
        "job": job,
        "status": "running",
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_at": datetime.datetime.now().astimezone().isoformat(),
        "source_config_sha256": sha(cfgpath),
        "runner_sha256": sha(Path(__file__)),
        "adapter_sha256": sha(O / "adapters/xpinn_ode_low.py"),
        "official_source_sha256": sha(R / "third_party/XPINNs_LICENSE"),
        "torch": torch.__version__,
        "gpu": str(device),
        "reference_version": "ode_fixed_source_reference_v2_20260911",
        "protocol": {
            "subdomains": M,
            "width": width,
            "depth": 6,
            "activation": (["tanh", "sin", "cos"] * ((M + 2) // 3))[:M],
            "adam_iterations": adam,
            "lbfgs_max_iterations": lbfgs,
            "learning_rate": 8e-4,
            "adam_decay": None,
            "boundary_weights": cfg["loss_weights"][1:],
            "interface_u_weight": 20.0,
            "interface_residual_weight": 2.0,
            "interface_flux_weight": flux_weight,
            "loss_normalization": "point-count-weighted global PDE mean; interface terms retain their declared coefficients",
            "optimizer_comparison": "same Adam/L-BFGS iteration limits; method-specific author Adam learning rate retained",
            "scope": "PyTorch adaptation; flux augmentation explicitly separate from author interface formulation",
        },
    }
    m["geometry_alignment"] = {
        "base_edges": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "overlap": False,
        "matches": "main ODE five base intervals",
    }
    m["precision_protocol"] = {
        "training": "float32",
        "initialization": "float32 initialization and training",
        "evaluation_grid": "same float32 coordinate values as prior cohort",
        "paired_original": None,
        "decomposition_protocol": "5 non-overlapping subdomains, width44 depth6; each input normalized to [-1,1]; two interface variants; PDE global-mean normalization",
    }
    save(out / "run_manifest.json", m)
    start = time.monotonic()
    step = 0
    evals = 0
    try:
        edges = np.linspace(0, 1, M + 1)
        activations = (["tanh", "sin", "cos"] * ((M + 2) // 3))[:M]
        models = [
            LocalCoordinates(
                AuthorActivationMLP(width, act), edges[i], edges[i + 1]
            ).to(device)
            for i, act in enumerate(activations)
        ]
        models = [net.to(dtype=torch.float32) for net in models]
        torch.set_default_dtype(torch.float32)
        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        params = [p for model in models for p in model.parameters()]
        edges = np.linspace(0, 1, M + 1)
        interior = [
            torch.linspace(edges[i], edges[i + 1], count + 2, device=device)[
                1:-1, None
            ].requires_grad_()
            for i, count in enumerate(allocate_points(N, M))
        ]
        interfaces = [
            torch.tensor([[v]], device=device, dtype=torch.float32, requires_grad=True)
            for v in edges[1:-1]
        ]
        x0 = torch.zeros((1, 1), device=device)
        x1 = torch.ones((1, 1), device=device)

        def loss_terms():
            pde = sum(
                (len(x) / N) * residual(net, x, alpha, k).square().mean()
                for net, x in zip(models, interior)
            )
            bc = (
                cfg["loss_weights"][1] * models[0](x0).square().mean()
                + cfg["loss_weights"][2] * models[-1](x1).square().mean()
            )
            ui = torch.zeros((), device=device)
            ri = ui.clone()
            fi = ui.clone()
            for i, x in enumerate(interfaces):
                left = models[i](x)
                right = models[i + 1](x)
                avg = (left + right) / 2
                ui = ui + 20 * (
                    (left - avg).square().mean() + (right - avg).square().mean()
                )
                ri = (
                    ri
                    + 2
                    * (
                        residual(models[i], x, alpha, k)
                        - residual(models[i + 1], x, alpha, k)
                    )
                    .square()
                    .mean()
                )
                if flux_weight:
                    dl = torch.autograd.grad(
                        left, x, torch.ones_like(left), create_graph=True
                    )[0]
                    dr = torch.autograd.grad(
                        right, x, torch.ones_like(right), create_graph=True
                    )[0]
                    fi = fi + flux_weight * (dl - dr).square().mean()
            return dict(
                pde=pde,
                boundary=bc,
                interface_value=ui,
                interface_residual=ri,
                interface_flux=fi,
            )

        history = (out / "training_history.jsonl").open("w")

        def record(terms, phase):
            r = {
                "step": step,
                "phase": phase,
                "closure_evaluations": evals,
                "wall_seconds": time.monotonic() - start,
                **{k: float(v.detach().cpu()) for k, v in terms.items()},
            }
            history.write(json.dumps(r) + "\n")
            history.flush()
            print("PROGRESS", json.dumps(r), flush=True)

        torch.cuda.synchronize() if device.type == "cuda" else None
        training_start = time.monotonic()
        opt = torch.optim.Adam(params, lr=8e-4)
        for step in range(1, adam + 1):
            opt.zero_grad(set_to_none=True)
            for x in interior + interfaces:
                x.grad = None
            terms = loss_terms()
            loss = sum(terms.values())
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training objective")
            loss.backward()
            opt.step()
            evals += 1
            if step == 1 or step % 500 == 0 or step == adam:
                record(terms, "Adam")
        if lbfgs:
            opt = torch.optim.LBFGS(
                params,
                lr=1,
                max_iter=min(1000, lbfgs),
                max_eval=1250,
                history_size=100,
                tolerance_grad=1e-8,
                tolerance_change=0,
                line_search_fn="strong_wolfe",
            )
            actual = 0

            def closure():
                nonlocal evals
                opt.zero_grad(set_to_none=True)
                for x in interior + interfaces:
                    x.grad = None
                loss = sum(loss_terms().values())
                evals += 1
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite L-BFGS objective")
                loss.backward()
                return loss

            while actual < lbfgs:
                opt.param_groups[0]["max_iter"] = min(1000, lbfgs - actual)
                opt.step(closure)
                count = opt.state[params[0]]["n_iter"]
                step = adam + count
                record(loss_terms(), "L-BFGS")
                if count - actual <= 1:
                    break
                actual = count
        torch.cuda.synchronize() if device.type == "cuda" else None
        m["training_seconds_including_periodic_loss_logging"] = (
            time.monotonic() - training_start
        )
        m["peak_torch_allocated_bytes"] = (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        )
        history.close()
        x = np.linspace(0, 1, 1000, dtype=np.float32)[:, None]
        pred = np.zeros(x.shape, dtype=np.float32)
        rr = np.zeros(x.shape, dtype=np.float32)
        for i, net in enumerate(models):
            idx = (x[:, 0] >= edges[i]) & (
                (x[:, 0] <= edges[i + 1]) if i == M - 1 else (x[:, 0] < edges[i + 1])
            )
            xx = torch.tensor(
                x[idx], device=device, dtype=torch.float32, requires_grad=True
            )
            pred[idx] = net(xx).detach().cpu().numpy()
            rr[idx] = (
                residual(net, xx, alpha, k, create_graph=False).detach().cpu().numpy()
            )
        true = exact_solution(x, alpha, k)
        e = pred - true
        metrics = dict(
            u_l2_relative=float(np.linalg.norm(e) / np.linalg.norm(true)),
            u_mse=float(np.mean(e**2)),
            u_max_abs=float(np.max(np.abs(e))),
            pde_residual=float(np.mean(np.abs(rr))),
        )
        jumps = []
        for i, xif in enumerate(interfaces):
            a = models[i](xif)
            b = models[i + 1](xif)
            da = torch.autograd.grad(a, xif, torch.ones_like(a))[0]
            db = torch.autograd.grad(b, xif, torch.ones_like(b))[0]
            jumps.append(
                {
                    "u": float((a - b).abs().detach().cpu()),
                    "derivative": float((da - db).abs().detach().cpu()),
                }
            )
        torch.save(
            {"edges": edges.tolist(), "models": [net.state_dict() for net in models]},
            out / "model.pt",
        )
        np.savez_compressed(
            out / "predictions.npz", x=x, reference=true, prediction=pred
        )
        m.update(
            status="complete"
            if all(np.isfinite(v) for v in metrics.values())
            else "nonfinite",
            metrics=metrics,
            interface_jumps=jumps,
            parameter_count=sum(p.numel() for p in params),
        )
    except FloatingPointError:
        m.update(status="nonfinite", traceback=traceback.format_exc())
        traceback.print_exc()
    except BaseException:
        m.update(status="failed", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        m.update(
            actual_final_step=step,
            closure_evaluations=evals,
            wall_seconds=time.monotonic() - start,
            finished_at=datetime.datetime.now().astimezone().isoformat(),
        )
        save(out / "run_manifest.json", m)
    print("RESULT", json.dumps(m), flush=True)
    return 0 if m["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
