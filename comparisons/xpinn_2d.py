"""Three-ring XPINN; L-BFGS function-evaluation limits are tracked separately."""
from pathlib import Path
import os, sys, json, time, math, random, signal, hashlib, datetime, traceback
import numpy as np
import torch
from torch import nn

W = Path(__file__).resolve().parent
R = W.parent
sys.path.insert(0, str(W / "adapters"))
import xpinn_scalar2d as equations

STOP = False


def stop(signum, frame):
    global STOP
    STOP = True


def now():
    return datetime.datetime.now().astimezone().isoformat()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def atomic_json(p, data):
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def atomic_torch(p, data):
    tmp = p.with_suffix(".tmp")
    with tmp.open("wb") as f:
        torch.save(data, f)
        f.flush()
        os.fsync(f.fileno())
    if p.exists():
        os.replace(p, p.with_name("checkpoint_previous.pt"))
    os.replace(tmp, p)


def grad(v, x):
    return torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]


def field_residual(net, x, ids, case, forcing=None):
    y = net(x, ids)
    if case != "ns":
        g = grad(y, x)
        lap = grad(g[..., 0:1], x)[..., 0:1] + grad(g[..., 1:2], x)[..., 1:2]
        if forcing is None:
            fn = (
                equations.poisson_forcing
                if case == "poisson"
                else equations.adr_forcing
            )
            forcing = fn(x.reshape(-1, 2)).reshape_as(y).detach()
        if case == "poisson":
            res = -lap - forcing
        else:
            bx, by, c = equations.adr_coefficients(x.reshape(-1, 2))
            res = (
                -1e-4 * lap
                + bx.reshape_as(y) * g[..., 0:1]
                + by.reshape_as(y) * g[..., 1:2]
                + c.reshape_as(y) * y
                - forcing
            )
        return y, res
    u, v = y[..., 0:1], y[..., 1:2]
    gu, gv, gp = grad(u, x), grad(v, x), grad(y[..., 2:3], x)
    lu = grad(gu[..., 0:1], x)[..., 0:1] + grad(gu[..., 1:2], x)[..., 1:2]
    lv = grad(gv[..., 0:1], x)[..., 0:1] + grad(gv[..., 1:2], x)[..., 1:2]
    return y, torch.cat(
        (
            u * gu[..., 0:1] + v * gu[..., 1:2] + gp[..., 0:1] - 0.01 * lu,
            u * gv[..., 0:1] + v * gv[..., 1:2] + gp[..., 1:2] - 0.01 * lv,
            gu[..., 0:1] + gv[..., 1:2],
        ),
        dim=-1,
    )


def evaluate(net, case, out, device):
    if case == "ns":
        ref = np.loadtxt(R / "experiments/ns2d/data/lid_driven_a8.dat", comments="%")
        coords = ref[:, :2].astype(np.float32)
        truth = ref[:, 2:5]
    else:
        ax = np.linspace(0, 1, 250, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax, indexing="ij")
        coords = np.stack((xx.ravel(), yy.ravel()), -1)
        fn = equations.poisson_exact if case == "poisson" else equations.adr_exact
        truth = fn(torch.from_numpy(coords).double()).numpy()
    with torch.no_grad():
        pred = np.concatenate(
            [
                pointwise_prediction(
                    net, torch.from_numpy(coords[k : k + 512]).to(device)
                )
                .cpu()
                .numpy()
                for k in range(0, len(coords), 512)
            ]
        )
    metrics = {}
    for i, name in enumerate(["u", "v", "p"][: pred.shape[-1]]):
        err = pred[:, i].astype(float) - truth[:, i]
        metrics.update(
            {
                name
                + "_l2_relative": float(
                    np.linalg.norm(err) / np.linalg.norm(truth[:, i])
                ),
                name + "_mse": float(np.mean(err**2)),
                name + "_max_abs": float(np.max(np.abs(err))),
            }
        )
    tmp = out / "predictions.tmp.npz"
    np.savez_compressed(tmp, x=coords, reference=truth, prediction=pred)
    os.replace(tmp, out / "predictions.npz")
    return metrics


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    out = R / job["output"]
    out.mkdir(parents=True, exist_ok=True)
    old = (
        json.loads((out / "run_manifest.json").read_text())
        if (out / "run_manifest.json").exists()
        else {}
    )
    if old.get("status") == "complete":
        return
    device = torch.device(job.get("device", "cpu"))
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed = job["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = json.loads((R / job["source_config"]).read_text())
    assert (
        cfg["seed"] == seed
        and sha(R / job["source_config"]) == job["source_config_sha256"]
    )
    net = LocalEnsemble(
        job["width"], cfg["net_depth"], 3 if job["case"] == "ns" else 1
    ).to(device)
    data = prepare_data(cfg, job["case"], device, job.get("validation_only", False))
    adam_steps = job.get("adam_override", cfg["adam_iterations"])
    lbfgs_steps = job.get("lbfgs_override", cfg["lbfgs_iterations"])
    manifest = dict(
        job=job,
        status="running",
        pid=os.getpid(),
        started_at=old.get("started_at", now()),
        resumed_at=now() if old else None,
        runner_sha256=sha(__file__),
        ring_module_sha256=sha(W / "ring_xpinn.py"),
        torch=torch.__version__,
        cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
        gpu=str(device) if device.type == "cpu" else torch.cuda.get_device_name(0),
        parameter_count=sum(p.numel() for p in net.parameters()),
        sampling=dict(
            domain_points=sum(z.numel() // 2 for z in data["inside"]),
            boundary_points=data["boundary"].numel() // 2,
            interface_points=sum(z.numel() // 2 for z in data["interface"]),
        ),
        protocol=dict(
            rings=[0.2, 0.4],
            networks=3,
            overlap=False,
            activation="tanh/sin/cos cyclic per subdomain; trainable per-layer scales",
            coordinate_normalization="ring bounding box mapped to [-1,1]^2",
            adam=adam_steps,
            lbfgs_max=lbfgs_steps,
            adam_lr=8e-4,
            interface_value=20,
            interface_residual=2,
            interface_normal_derivative=job.get("flux_weight", 0.0),
            pde_reduction="point-weighted global mean, sum equations",
            boundary_reduction="main scalar loss or NS top/other component means and one pressure reference",
            interface_reduction="sum per-interface losses, original XPINN two-sided mean-solution formula",
            precision="float32",
            checkpoint="optimizer and RNG saved; stable-step atomic checkpoint every 250 Adam / 100 LBFGS iterations",
            scope="three square rings; 16/8/1 base cells merged, one network per ring; near matched total parameters; no normal-derivative penalty",
        ),
    )
    phase = "adam"
    astep = 0
    lstep = 0
    evals = 0
    prior_seconds = 0.0
    checkpoint = out / "checkpoint.pt"
    saved = None
    source_checkpoint = R / job["resume_from"] if job.get("resume_from") else None
    if checkpoint.exists() or source_checkpoint:
        load_path = checkpoint if checkpoint.exists() else source_checkpoint
        saved = torch.load(load_path, map_location=device, weights_only=False)
        if load_path == checkpoint:
            assert saved["job"] == job and saved["runner_sha256"] == sha(__file__)
        else:
            assert sha(load_path) == job["resume_checkpoint_sha256"]
            assert saved["runner_sha256"] == job["original_runner_sha256"]
            for key in [
                "case",
                "seed",
                "source_config",
                "source_config_sha256",
                "width",
                "flux_weight",
            ]:
                assert saved["job"][key] == job[key], key
            assert saved["phase"] == "done" and saved["lbfgs_step"] < lbfgs_steps
            saved["phase"] = "lbfgs"
            manifest["budget_fix_resume"] = dict(
                source=str(load_path),
                sha256=sha(load_path),
                original_step=saved["adam_step"] + saved["lbfgs_step"],
                original_evaluations=saved["evaluations"],
                original_wall_seconds=saved["wall_seconds"],
                reason="v1 per-block function-evaluation cap was mistaken for convergence",
            )
        net.load_state_dict(saved["model"])
        phase = saved["phase"]
        astep = saved["adam_step"]
        lstep = saved["lbfgs_step"]
        evals = saved["evaluations"]
        prior_seconds = saved["wall_seconds"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
    atomic_json(out / "run_manifest.json", manifest)
    started = time.monotonic()
    history = (out / "training_history.jsonl").open("a", buffering=1)
    last_terms = {}

    def ckpt(opt):
        atomic_torch(
            checkpoint,
            dict(
                job=job,
                runner_sha256=sha(__file__),
                model=net.state_dict(),
                optimizer=opt.state_dict(),
                phase=phase,
                adam_step=astep,
                lbfgs_step=lstep,
                evaluations=evals,
                wall_seconds=prior_seconds + time.monotonic() - started,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else [],
                numpy_rng=np.random.get_state(),
                python_rng=random.getstate(),
            ),
        )
        manifest.update(
            last_checkpoint_step=astep + lstep,
            updated_at=now(),
            wall_seconds=prior_seconds + time.monotonic() - started,
        )
        atomic_json(out / "run_manifest.json", manifest)

    def record(opt):
        row = dict(
            step=astep + lstep,
            phase=phase,
            evaluations=evals,
            wall_seconds=prior_seconds + time.monotonic() - started,
            losses={k: float(v.detach()) for k, v in last_terms.items()},
        )
        history.write(json.dumps(row) + "\n")
        history.flush()
        os.fsync(history.fileno())
        print("PROGRESS", json.dumps(row), flush=True)
        ckpt(opt)

    def objective():
        nonlocal evals, last_terms
        for name in ["inside", "interface"]:
            for points in data[name]:
                points.grad = None
        last_terms = loss_terms(
            net, data, cfg, job["case"], job.get("flux_weight", 0.0)
        )
        loss = sum(last_terms.values())
        evals += 1
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "nonfinite objective; retained without seed replacement"
            )
        return loss

    try:
        if phase == "adam":
            opt = torch.optim.Adam(net.parameters(), lr=8e-4)
            if saved:
                opt.load_state_dict(saved["optimizer"])
            if not checkpoint.exists():
                ckpt(opt)
            while astep < adam_steps and not STOP:
                opt.zero_grad(set_to_none=True)
                objective().backward()
                opt.step()
                astep += 1
                if astep == 1 or astep % 250 == 0 or astep == adam_steps:
                    record(opt)
            ckpt(opt)
        if not STOP and phase != "done":
            phase = "lbfgs"
            opt = torch.optim.LBFGS(
                net.parameters(),
                lr=1.0,
                max_iter=min(100, lbfgs_steps or 1),
                max_eval=125,
                tolerance_grad=1e-9,
                tolerance_change=1e-12,
                history_size=50,
                line_search_fn="strong_wolfe",
            )
            if saved and saved["phase"] == "lbfgs":
                opt.load_state_dict(saved["optimizer"])

            def closure():
                opt.zero_grad(set_to_none=True)
                loss = objective()
                loss.backward()
                return loss

            def terminal_tolerance_check():
                # Reevaluate at accepted parameters: .grad can belong to a line-search trial.
                value = float(closure().detach())
                state = opt.state[next(net.parameters())]
                group = opt.param_groups[0]
                flat = torch.cat(
                    [
                        p.grad.reshape(-1)
                        if p.grad is not None
                        else torch.zeros_like(p).reshape(-1)
                        for p in net.parameters()
                    ]
                )
                grad_inf = float(flat.abs().max())
                step_inf = (
                    float((state["d"] * state["t"]).abs().max())
                    if "d" in state
                    else None
                )
                change = (
                    abs(value - state["prev_loss"])
                    if state.get("prev_loss") is not None
                    else None
                )
                gtd = (
                    float(state["prev_flat_grad"].dot(state["d"]))
                    if state.get("prev_flat_grad") is not None and "d" in state
                    else None
                )
                flags = dict(
                    gradient=grad_inf <= group["tolerance_grad"],
                    step=step_inf is not None and step_inf <= group["tolerance_change"],
                    loss_change=change is not None
                    and change < group["tolerance_change"],
                    direction=gtd is not None and gtd > -group["tolerance_change"],
                )
                check = dict(
                    lbfgs_step=lstep,
                    objective=value,
                    gradient_inf=grad_inf,
                    step_inf=step_inf,
                    loss_change=change,
                    previous_directional_derivative=gtd,
                    flags=flags,
                    stop=any(flags.values()),
                )
                manifest.setdefault("terminal_tolerance_checks", []).append(check)
                manifest["diagnostic_evaluations"] = (
                    manifest.get("diagnostic_evaluations", 0) + 1
                )
                return check["stop"]

            source_converged = bool(
                source_checkpoint
                and not checkpoint.exists()
                and terminal_tolerance_check()
            )
            if source_converged:
                manifest[
                    "lbfgs_stop_reason"
                ] = "accepted source parameters satisfy fixed tolerance; retained without further updates"
            ckpt(opt)
            while lstep < lbfgs_steps and not STOP and not source_converged:
                before = opt.state[next(net.parameters())].get("n_iter", 0)
                eval_before = opt.state[next(net.parameters())].get("func_evals", 0)
                opt.param_groups[0]["max_iter"] = min(100, lbfgs_steps - lstep)
                opt.step(closure)
                after = opt.state[next(net.parameters())].get("n_iter", 0)
                delta = after - before
                lstep += delta
                record(opt)
                eval_delta = (
                    opt.state[next(net.parameters())].get("func_evals", 0) - eval_before
                )
                manifest.setdefault("lbfgs_blocks", []).append(
                    dict(
                        iterations=delta,
                        function_evaluations=eval_delta,
                        max_iter=opt.param_groups[0]["max_iter"],
                        max_eval=opt.param_groups[0]["max_eval"],
                    )
                )
                if delta <= 0:
                    manifest[
                        "lbfgs_stop_reason"
                    ] = "no additional LBFGS iterations under fixed tolerance"
                    break
                if (
                    delta < opt.param_groups[0]["max_iter"]
                    and eval_delta < opt.param_groups[0]["max_eval"]
                ):
                    manifest[
                        "lbfgs_stop_reason"
                    ] = "optimizer stopped under fixed tolerances before either per-block budget"
                    break
                if delta < opt.param_groups[0]["max_iter"]:
                    if terminal_tolerance_check():
                        manifest[
                            "lbfgs_stop_reason"
                        ] = "per-call evaluation cap coincides with fixed tolerance at accepted parameters"
                        break
                    ckpt(opt)
                    print("CONTINUE_EVAL_CAP", lstep, delta, eval_delta, flush=True)
        if STOP:
            ckpt(opt)
            manifest.update(status="paused", reason="requested safe checkpoint")
            return
        metrics = evaluate(net, job["case"], out, device)
        with torch.enable_grad():
            jumps = ring_jumps(net, data, job["case"])
        manifest.update(
            status="complete"
            if all(math.isfinite(v) for v in metrics.values())
            else "nonfinite",
            metrics=metrics,
            interface_jumps=jumps,
            actual_final_step=astep + lstep,
            function_evaluations=evals,
            finished_at=now(),
        )
        phase = "done"
        ckpt(opt)
    except BaseException:
        manifest.update(
            status="failed", traceback=traceback.format_exc(), finished_at=now()
        )
        traceback.print_exc()
        raise
    finally:
        manifest["wall_seconds"] = prior_seconds + time.monotonic() - started
        atomic_json(out / "run_manifest.json", manifest)
        history.close()


import ring_xpinn

ring_xpinn.install(sys.modules[__name__])
if __name__ == "__main__":
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGUSR1, stop)
    main()
