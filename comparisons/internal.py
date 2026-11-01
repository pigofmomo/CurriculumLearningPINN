"""PINN, spatial curriculum, bridge, and C2 configuration runner."""
from pathlib import Path
import copy, dataclasses, json, sys, time, traceback
import numpy as np
import torch
import deepxde as dde
import pinn_pro
from experiments.ode1d.ode1d import OdeConfig_l, OdeConfig_h, Ode1D
from experiments.poisson_high_fre.poisson_high_fre import Poisson2dHFConfig, Poisson2dHF
from experiments.adr.adr import ADRConfig, ADR
from experiments.ns2d.ns2d import NS2DConfig, NS2D

job = json.loads(Path(sys.argv[1]).read_text())
out = Path(job["output"])
torch.set_num_threads(2)
ctype, modeltype = {
    "ode_low": (OdeConfig_l, Ode1D),
    "ode_high": (OdeConfig_h, Ode1D),
    "poisson": (Poisson2dHFConfig, Poisson2dHF),
    "adr": (ADRConfig, ADR),
    "ns": (NS2DConfig, NS2D),
}[job["case"]]
cfg = ctype()
for k, v in json.loads(Path(job["source_config"]).read_text()).items():
    if not hasattr(cfg, k):
        raise ValueError(f"Unknown configuration field: {k}")
    setattr(cfg, k, copy.deepcopy(v))
if job["validation_only"]:
    cfg.adam_iterations = 6
    cfg.lbfgs_iterations = 4
    cfg.num_domain = 100
    cfg.num_boundary = 2 if job["case"].startswith("ode") else 40
    cfg.num_test = 100
    cfg.net_width = 12
    cfg.net_depth = 2
    rw = cfg.reweight_config
    active = rw.get("reweight_adaptive_begin", 100000) < 100000
    rw.update(
        reweight_every=2,
        record_every=2,
        reweight_causal_begin=1,
        reweight_causal_end=4,
        reweight_adaptive_begin=4 if active else 100000,
        reweight_adaptive_end=8 if active else 100000,
    )
if job.get("gate_schedule") == "fixed_thirds":
    from hard_spatial import install

    # The separate fixed-thirds baseline starts at update zero and disables C2.
    cfg.reweight_config.update(
        reweight_causal_begin=100000,
        reweight_causal_end=100000,
        reweight_adaptive_begin=100000,
        reweight_adaptive_end=100000,
    )
    install(pinn_pro, cfg, out)

start = time.monotonic()
manifest = dict(
    job=job,
    config=dataclasses.asdict(cfg),
    status="running",
    deepxde_version=dde.__version__,
)
try:
    model = modeltype(cfg)
    trainer = (
        pinn_pro.PINN if job.get("trainer") == "plain" else pinn_pro.PINNWeightedSamples
    )(model, out)
    trainer.train_and_evaluate()
    x, y = model.reference
    prediction = model.model.predict(x)
    np.savez_compressed(
        out / "predictions.npz", x=x, reference=y, prediction=prediction
    )
    metrics = model.component_metrics(y, prediction)
    manifest.update(
        status="complete"
        if all(np.isfinite(v) for v in metrics.values())
        else "nonfinite",
        metrics=metrics,
    )
except Exception:
    manifest.update(status="failed", traceback=traceback.format_exc())
    traceback.print_exc()
finally:
    manifest["wall_seconds"] = time.monotonic() - start
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
sys.exit(0 if manifest["status"] == "complete" else 1)
