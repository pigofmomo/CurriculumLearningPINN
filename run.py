"""Run one saved experiment configuration or a short installation check."""
from pathlib import Path
import argparse, copy, hashlib, json, os, subprocess, sys

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true", help="List available presets")
    p.add_argument("--preset", help="Preset identifier from --list")
    p.add_argument("--output", type=Path, help="New output directory")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--smoke", action="store_true", help="Short run; not a paper result")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layers", type=int, choices=[2, 3, 4, 5])
    p.add_argument("--epsilon", type=float)
    p.add_argument("--bridge-weight", type=float)
    p.add_argument("--degree", type=int, choices=[1, 2, 3, 4])
    p.add_argument("--stage", choices=["full", "c1", "plain"], default="full")
    args = p.parse_args()
    presets = json.loads((ROOT / "configs/presets.json").read_text())
    if args.list:
        print("\n".join(sorted(presets)))
        return
    if args.preset not in presets:
        p.error("Choose a preset shown by --list")
    spec = copy.deepcopy(presets[args.preset])
    out = (args.output or ROOT / "outputs" / args.preset).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = json.loads((ROOT / "configs/cases.json").read_text())
    config = copy.deepcopy(cases.get(spec["case"], {"seed": args.seed}))
    config.update(spec.pop("overrides", {}))
    config["seed"] = args.seed
    edits = (
        any(
            v is not None
            for v in [args.layers, args.epsilon, args.bridge_weight, args.degree]
        )
        or args.stage != "full"
    )
    if edits and (spec["runner"] != "internal" or spec.get("gate_schedule")):
        p.error("Sensitivity/ablation options apply to main/* presets")
    if spec["runner"] == "internal":
        rw = config["reweight_config"]
        if args.layers is not None:
            n = 2 * args.layers - 1
            rw["num_subdomains"] = n if spec["case"].startswith("ode") else [n, n]
        if args.epsilon is not None:
            rw["decay_epsi"] = args.epsilon
        if args.bridge_weight is not None:
            rw["low_fre_data_weight"] = args.bridge_weight
        if args.degree is not None:
            rw["low_fre_n"] = args.degree
        if args.stage in ["c1", "plain"]:
            rw.update(reweight_adaptive_begin=10**9, reweight_adaptive_end=10**9)
        if args.stage == "plain":
            rw.update(
                reweight_causal_begin=10**9,
                reweight_causal_end=10**9,
                low_fre_data_weight=0,
                frame_data_weight=0,
            )

    if config.get("data_file"):
        config["data_file"] = str(ROOT / config["data_file"])
    config_path = out / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    job = dict(
        spec,
        id=out.name,
        seed=config["seed"],
        output=str(out),
        source_config=str(config_path),
        source_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        validation_only=args.smoke,
        device=args.device,
        preset=args.preset,
    )
    job["subdomains"] = 5 if spec["case"].startswith("ode") else 3
    if args.smoke:
        job.update(adam_override=3, lbfgs_override=2)
    job_path = out / "job.json"
    job_path.write_text(json.dumps(job, indent=2))
    runner = spec["runner"]
    if runner in ["fbpinn", "xpinn"]:
        runner += "_ode" if spec["case"].startswith("ode") else "_2d"
    env = os.environ.copy()
    env.update(
        DDE_BACKEND="pytorch",
        MPLBACKEND="Agg",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(ROOT),
            str(ROOT / "comparisons"),
            str(ROOT / "comparisons/adapters"),
            str(ROOT / "third_party"),
        ]
    )
    if args.device == "cpu":
        env.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    command = [
        sys.executable,
        str(ROOT / "comparisons" / f"{runner}.py"),
        str(job_path),
    ]
    if runner == "ars" and args.smoke:
        command.append("--smoke")
    result = subprocess.run(command, env=env, cwd=ROOT)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
