"""Check corrected ODE truth, stored arithmetic, and preset input paths."""
from pathlib import Path
import csv, json, os, sys

os.environ["DDE_BACKEND"] = "pytorch"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from scipy.integrate import quad
from experiments.ode1d.ode1d import Ode1D, OdeConfig_l, OdeConfig_h

ROOT = Path(__file__).resolve().parent
for cls in [OdeConfig_l, OdeConfig_h]:
    cfg = cls()
    exact = Ode1D(cfg).build_reference_fn()
    a, k = cfg.ALPHA_VALUE, cfg.K_VALUE

    def forcing(x):
        p = x - x * x
        return np.exp(a * x) * (
            (-2 + 2 * a * (1 - 2 * x) + (a * a - k * k) * p) * np.sin(k * x)
            + 2 * k * (1 - x - x * x + a * p) * np.cos(k * x)
        )

    integral = quad(lambda t: (1 - t) * forcing(t), 0, 1, epsabs=1e-10)[0]
    for x in np.linspace(0, 1, 19):
        direct = (
            quad(lambda t: (x - t) * forcing(t), 0, x, epsabs=1e-10)[0] - x * integral
        )
        assert np.isclose(exact(np.array([x]))[0], direct, atol=2e-10, rtol=1e-10)
    assert np.max(np.abs(exact(np.array([0.0, 1.0])))) < 1e-13
from experiments.adr.adr import ADRConfig
from experiments.ns2d.ns2d import NS2DConfig
from experiments.poisson_high_fre.poisson_high_fre import Poisson2dHFConfig

classes = {
    "ode_low": OdeConfig_l,
    "ode_high": OdeConfig_h,
    "adr": ADRConfig,
    "ns": NS2DConfig,
    "poisson": Poisson2dHFConfig,
}

presets = json.loads((ROOT / "configs/presets.json").read_text())
cases = json.loads((ROOT / "configs/cases.json").read_text())
for name, spec in presets.items():
    cfg = dict(cases.get(spec["case"], {"seed": 0}))
    cfg.update(spec.get("overrides", {}))
    if spec["case"] != "pb":
        obj = classes[spec["case"]]()
        assert all(hasattr(obj, k) for k in cfg), name
    if cfg.get("data_file"):
        assert (ROOT / cfg["data_file"]).is_file(), name
print(
    f"Passed: corrected ODE source and boundaries, {len(presets)} presets and reference inputs."
)
