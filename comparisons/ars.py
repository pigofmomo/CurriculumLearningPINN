"""Eq. (15) regional weighting after full C1; joint Adam then frozen-weight L-BFGS."""
from pathlib import Path
import argparse, copy, dataclasses, datetime, hashlib, json, os, sys, time, traceback

W = Path(__file__).resolve().parent
ROOT = W.parent
p = argparse.ArgumentParser()
p.add_argument("job")
p.add_argument("--smoke", action="store_true")
args = p.parse_args()
job = json.loads(Path(args.job).read_text())
code = ROOT
import numpy as np
import torch
import deepxde as dde
import pinn_pro
from experiments.utils import build_reference_function
from experiments.adr.adr import ADR, ADRConfig
from experiments.ns2d.ns2d import NS2D, NS2DConfig
from experiments.poisson_high_fre.poisson_high_fre import Poisson2dHF, Poisson2dHFConfig
import c1_checkpoint as ckpt


def now():
    return datetime.datetime.now().astimezone().isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def setup(cfg, cls, out):
    dde.config.set_random_seed(cfg.seed)
    pde = cls(cfg)
    pde.pde = pde.build_pde()
    if pde.reference_fn is None:
        pde.reference = pde.load_reference(cfg.data_file)
        pde.reference_fn = build_reference_function(*pde.reference)
    pde.geom = pde.make_domain(cfg)
    pde.data = pde.make_dataset_pinn_weighted_samples(pde.geom, pde.reference_fn, cfg)
    pde.net = pde.make_network(cfg)
    trainer = pinn_pro.PINNWeightedSamples(pde, out)
    trainer.run_dir.mkdir(parents=True, exist_ok=True)
    pde.model = pinn_pro.ModelWeightedSamples(pde.data, pde.net, trainer)
    return pde, trainer


class PrefixReached(Exception):
    pass


def run():
    torch.set_num_threads(2)
    cfgtype, cls = {
        "adr": (ADRConfig, ADR),
        "ns": (NS2DConfig, NS2D),
        "poisson": (Poisson2dHFConfig, Poisson2dHF),
    }[job["case"]]
    cfg = cfgtype()
    original = json.loads((ROOT / job["source_config"]).read_text())
    names = {f.name for f in dataclasses.fields(cfg)}
    for k, v in original.items():
        if k in names:
            setattr(cfg, k, copy.deepcopy(v))
        elif not k.startswith("cuda"):
            raise ValueError(k)
    cfg.reweight_config.update(job.get("reweight_overrides", {}))
    if cfg.data_file:
        cfg.data_file = str(code / "experiments/ns2d/data" / Path(cfg.data_file).name)
    assert cfg.seed == job["seed"]
    if args.smoke:
        cfg.adam_iterations = 2
        cfg.lbfgs_iterations = 4 if job["case"] == "adr" else 6
        cfg.num_domain = 100
        cfg.num_boundary = 40
        cfg.num_test = 100
        cfg.net_width = 12
        cfg.net_depth = 2
        cfg.reweight_config["record_every"] = 2
        smoke_begin = 2 if job["case"] == "adr" else 4
        cfg.reweight_config.update(
            reweight_every=1 if job["case"] == "adr" else 2,
            reweight_causal_begin=1,
            reweight_causal_end=smoke_begin,
            reweight_adaptive_begin=smoke_begin,
            reweight_adaptive_end=smoke_begin + 2,
        )
    begin = cfg.reweight_config["reweight_adaptive_begin"]
    end = cfg.reweight_config["reweight_adaptive_end"]
    total = cfg.adam_iterations + cfg.lbfgs_iterations
    nadam = (end - begin) // 2
    nfixed = end - begin - nadam
    ntail = total - end
    assert begin >= cfg.adam_iterations and min(nadam, nfixed) > 0 and ntail >= 0
    identity = {
        "case": job["case"],
        "seed": cfg.seed,
        "config": dataclasses.asdict(cfg),
        "source_sha256": hashlib.sha256(
            "".join(
                sha(p)
                for folder in [ROOT / "pinn_pro", ROOT / "experiments"]
                for p in sorted(folder.rglob("*.py"))
            ).encode()
        ).hexdigest(),
        "deepxde_version": dde.__version__,
        "torch_version": torch.__version__,
        "boundary_step": begin,
        "smoke": args.smoke,
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    cache = Path(job["output"]).parent / "c1_cache" / f'{job["case"]}_s{cfg.seed}_{key}'
    cache.mkdir(parents=True, exist_ok=True)
    checkpoint = cache / "full_c1_before_c2.pt"
    out = Path(job["output"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run_manifest.json").exists():
        raise FileExistsError(f"Preserve prior attempt: {out}")
    manifest = dict(
        job=job,
        status="running",
        stage="prefix",
        pid=os.getpid(),
        started_at=now(),
        cache=str(cache),
        identity=identity,
        source_config_sha256=sha(ROOT / job["source_config"]),
        runner_sha256=sha(__file__),
        checkpoint_module_sha256=sha(W / "c1_checkpoint.py"),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        protocol="Eq15-derived regional weighting, full C1 preserved, changed C2 optimizer allocation",
        schedule=dict(
            prefix_end=begin,
            joint_adam=nadam,
            frozen_lbfgs=nfixed,
            unweighted_tail=ntail,
        ),
        smoke=args.smoke,
    )
    start = time.monotonic()
    write(out / "run_manifest.json", manifest)
    try:
        dde.optimizers.LBFGS_options["iter_per_step"] = cfg.reweight_config.get(
            "reweight_every", 1000
        )
        pde, trainer = setup(cfg, cls, cache / "prefix_artifacts")
        model = pde.model

        def compile_saved():
            model.compile("adam", lr=cfg.learning_rate, loss_weights=cfg.loss_weights)
            if begin > cfg.adam_iterations:
                model.compile("L-BFGS-B", loss_weights=cfg.loss_weights)

        if checkpoint.exists():
            compile_saved()
            model.data.init_data(savepath=None)
            saved = ckpt.load(checkpoint, model, identity)
            manifest["prefix_reused"] = True
        else:
            old_record = model.record_and_plot

            def boundary_record():
                old_record()
                if model.train_state.step >= begin:
                    assert (
                        model.train_state.step == begin
                    ), "Original LBFGS overshot C2 boundary"
                    state = ckpt.rng_state()
                    x, y = pde.reference
                    prediction = model.predict(x)
                    metrics = pde.component_metrics(y, prediction)
                    np.savez_compressed(
                        cache / "prefix_predictions.npz",
                        x=x,
                        reference=y,
                        prediction=prediction,
                    )
                    ckpt.restore_rng(state)
                    # prediction has no numerical training effect; restore mode before saving
                    model.net.train(True)
                    ckpt.save(
                        checkpoint,
                        model,
                        identity,
                        {"prefix_metrics": metrics, "config": dataclasses.asdict(cfg)},
                    )
                    write(
                        cache / "prefix_summary.json",
                        dict(
                            step=begin,
                            metrics=metrics,
                            checkpoint_sha256=sha(checkpoint),
                            at=now(),
                        ),
                    )
                    raise PrefixReached()

            model.record_and_plot = boundary_record
            try:
                trainer.optimize()
            except PrefixReached:
                pass
            finally:
                model.record_and_plot = old_record
            if not checkpoint.exists():
                raise RuntimeError(
                    "Prefix stopped before C2 boundary; no branch launched"
                )
            saved = ckpt.load(checkpoint, model, identity)
            manifest["prefix_reused"] = False
        assert model.train_state.step == begin
        # Round trip verifies restoration after deliberate model corruption, without changing saved state.
        probe = pde.reference[0][:: max(1, len(pde.reference[0]) // 256)]
        expected = model.predict(probe).copy()
        with torch.no_grad():
            next(model.net.parameters()).add_(1.0)
        ckpt.load(checkpoint, model, identity)
        actual = model.predict(probe)
        assert np.array_equal(
            expected, actual
        ), "checkpoint prediction round trip mismatch"
        ckpt.load(checkpoint, model, identity)
        manifest.update(
            stage="ars_joint_adam",
            prefix_checkpoint_sha256=sha(checkpoint),
            roundtrip_prediction_max_abs=float(np.max(np.abs(expected - actual))),
        )
        write(out / "run_manifest.json", manifest)
        model.reset_reweight_and_anchors()
        model.data_loss_weight = 0.0
        q = torch.nn.Parameter(
            torch.zeros(
                len(model.data.train_x_group),
                device=next(model.net.parameters()).device,
            )
        )

        def objective(regularizer):
            s = model.train_state
            losses = model.outputs_losses_train(
                s.x_train_group, s.x_train_bc, None, None
            )
            parts = model.f_partition
            n = sum(len(x) for x in s.x_train_group) + sum(len(x) for x in s.x_train_bc)
            regional = torch.stack(
                [
                    sum(
                        (f.square().sum() / n) * cfg.loss_weights[k]
                        for k, f in enumerate(part)
                    )
                    for part in parts
                ]
            )
            objective = losses.sum() + torch.sum((torch.exp(-q) - 1) * regional)
            if regularizer:
                objective = objective + 0.5 * q.sum()
            return objective, regional

        check_loss, check_regional = objective(True)
        check_gradient = torch.autograd.grad(check_loss, q)[0]
        assert torch.allclose(
            check_gradient, 0.5 - check_regional.detach(), rtol=1e-6, atol=1e-7
        ), "Eq15 log-weight gradient mismatch"
        manifest["eq15_initial_weight_gradient_checked"] = True
        opt = torch.optim.Adam(list(model.net.parameters()) + [q], lr=cfg.learning_rate)
        history = []

        def finite(t):
            if not torch.isfinite(t).all():
                raise FloatingPointError(
                    "non-finite objective/weight; retain this attempt"
                )

        def report(stage, objective_value):
            # Optimizer.step returns a pre-update value (LBFGS: start of the block).
            # Evaluate PDE derivatives at the accepted state, with gradients enabled.
            current, _ = objective(False)
            regularized = current + 0.5 * q.sum()
            weights = torch.exp(-q)
            finite(current)
            finite(regularized)
            finite(weights)
            if not torch.all(weights > 0):
                raise FloatingPointError(
                    "nonpositive weight after exponentiation; retain attempt"
                )
            network_value = float(current.detach())
            regularized_value = float(regularized.detach())
            del current, regularized
            x, y = pde.reference
            pred = model.predict(x)
            metrics = pde.component_metrics(y, pred)
            row = dict(
                step=model.train_state.step,
                stage=stage,
                objective=regularized_value
                if stage == "ars_joint_adam"
                else network_value,
                network_objective=network_value,
                regularized_objective=regularized_value,
                optimizer_pre_update_value=float(objective_value),
                objective_evaluation="accepted_state_same_as_metrics",
                weights=weights.detach().cpu().tolist(),
                metrics=metrics,
            )
            history.append(row)
            write(out / "history.json", history)
            manifest.update(
                stage=stage, last_step=model.train_state.step, latest_metrics=metrics
            )
            write(out / "run_manifest.json", manifest)
            print("PROGRESS", json.dumps(row), flush=True)

        def save_branch(stage, optimizer):
            path = out / "branch_checkpoint.pt"
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as f:
                torch.save(
                    dict(
                        model_state_dict=model.net.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        q=q.detach(),
                        step=model.train_state.step,
                        stage=stage,
                        rng=ckpt.rng_state(),
                        prefix_checkpoint=str(checkpoint),
                        prefix_sha256=sha(checkpoint),
                        identity=identity,
                    ),
                    f,
                )
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(path)

        for i in range(nadam):
            opt.zero_grad()
            loss, _ = objective(True)
            finite(loss)
            loss.backward()
            for par in list(model.net.parameters()) + [q]:
                if par.grad is not None:
                    finite(par.grad)
            opt.step()
            finite(q)
            finite(torch.exp(-q))
            if not torch.all(torch.exp(-q) > 0):
                raise FloatingPointError("nonpositive regional weight after update")
            model.train_state.step += 1
            if (i + 1) % 250 == 0 or i + 1 == nadam:
                report("ars_joint_adam", loss.detach())
                save_branch("ars_joint_adam", opt)
        q.requires_grad_(False)

        # The paper explicitly fixes weights during LBFGS. New optimizer starts at this transition.
        def lbfgs_stage(count, stage):
            if not count:
                return
            model.compile("L-BFGS-B", loss_weights=cfg.loss_weights)
            done = 0
            frozen_q = q.detach().clone()
            block = cfg.reweight_config.get("reweight_every", 500)
            while done < count:
                group = model.opt.param_groups[0]
                group["max_iter"] = min(block, count - done)
                group["max_eval"] = int(group["max_iter"] * 1.25)
                before_state = model.opt.state_dict().get("state", {}).get(0, {})
                before = before_state.get("n_iter", 0)
                eval_before = before_state.get("func_evals", 0)

                def closure():
                    model.opt.zero_grad()
                    loss, _ = objective(False)
                    finite(loss)
                    loss.backward()
                    for par in model.net.parameters():
                        if par.grad is not None:
                            finite(par.grad)
                    return loss

                loss = model.opt.step(closure)
                assert torch.equal(
                    q, frozen_q
                ), "Frozen regional weights changed during LBFGS"
                after = model.opt.state_dict()["state"][0]["n_iter"]
                delta = after - before
                if delta <= 0:
                    break
                done += delta
                model.train_state.step += delta
                report(stage, loss.detach())
                save_branch(stage, model.opt)
                eval_delta = (
                    model.opt.state_dict()["state"][0]["func_evals"] - eval_before
                )
                manifest.setdefault("lbfgs_blocks", []).append(
                    dict(
                        stage=stage,
                        iterations=delta,
                        function_evaluations=eval_delta,
                        max_iter=group["max_iter"],
                        max_eval=group["max_eval"],
                    )
                )
                if delta < group["max_iter"] and eval_delta < group["max_eval"]:
                    break
            return done

        fixed_actual = lbfgs_stage(nfixed, "ars_frozen_lbfgs")
        # Keep the original end-of-C2 unweighted tail, even when LBFGS converged early.
        q.zero_()
        tail_actual = lbfgs_stage(ntail, "unweighted_tail")
        x, y = pde.reference
        prediction = model.predict(x)
        metrics = pde.component_metrics(y, prediction)
        np.savez_compressed(
            out / "predictions.npz", x=x, reference=y, prediction=prediction
        )
        manifest.update(
            status="complete"
            if all(np.isfinite(v) for v in metrics.values())
            else "nonfinite",
            metrics=metrics,
            actual_final_step=model.train_state.step,
            actual_frozen_lbfgs=fixed_actual,
            actual_tail_lbfgs=tail_actual,
        )
    except BaseException:
        manifest.update(status="failed", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        manifest.update(finished_at=now(), wall_seconds=time.monotonic() - start)
        write(out / "run_manifest.json", manifest)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(run())
