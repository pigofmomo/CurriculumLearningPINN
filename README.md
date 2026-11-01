# CurriculumLearningPINN

Spatial curriculum PINNs with regional residual weighting and a polynomial information bridge. Examples cover low- and high-frequency ODEs, Poisson, ADR, and steady Navier–Stokes equations.

## Setup

Python 3.10:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python check.py
```

The runner selects the DeepXDE PyTorch backend. For GPU training, install a PyTorch wheel compatible with your CUDA driver.

## Run

```bash
python run.py --list
python run.py --preset main/ode_high --smoke --output outputs/check
CUDA_VISIBLE_DEVICES=0 python run.py --preset main/ode_high --seed 0 --device cuda --output outputs/ode_high
```

`--smoke` runs a few updates to check installation. Omit it for the configured training budget. Output directories must be new. Configuration, predictions, and metrics are saved with each run.

The main examples are `main/ode_low`, `main/ode_high`, `main/poisson`, `main/adr`, and `main/ns`. Their settings are in `configs/cases.json`.

For a sensitivity or ablation run:

```bash
python run.py --preset main/poisson --stage c1 --layers 3 --epsilon 0.5 --seed 0 --output outputs/poisson_c1
python run.py --preset main/poisson --bridge-weight 0 --seed 0 --output outputs/poisson_no_bridge
```

`--layers L` uses `2L-1` intervals per coordinate. `--degree` selects the bridge polynomial degree. `--stage c1` disables second-stage adaptation; `--stage plain` disables both curriculum stages and the bridge. These options define new runs from the supplied example configuration.

## Comparisons

The same entry point provides `causal/*` (cumulative spatial hard gate), `ars/*` (ARS-derived regional weighting after C1), `xpinn/*`, and `pb/outer`, `pb/inner`, `pb/uniform`.

FBPINN uses a separate JAX environment:

```bash
python -m venv .venv-fbpinn
source .venv-fbpinn/bin/activate
pip install -r requirements-fbpinn.txt
python run.py --preset fbpinn/poisson/rings --smoke --output outputs/fbpinn_check
```

The dependency file installs CPU JAX. GPU runs require the matching JAX CUDA plugin. FBPINN includes five-interval ODE examples and both `strips` and `rings` for 2D problems. XPINN uses five ODE intervals or three square rings.

## Layout

- `pinn_pro/`: main training and regional weighting.
- `experiments/`: PDE definitions and NS reference data.
- `comparisons/`: supplemental methods and equation adapters.
- `configs/`: example configurations and runner selection.
- `third_party/`: the FBPINN library and required notices.

This is a compact runnable implementation with example configurations. Full experiment batches, trained models, and publication figures are not included. See [implementation notes](docs/methods.md) and [third-party notices](docs/third_party.md).

## Citation


If this repository is useful for your research, please cite the accompanying paper once it is available.

```bibtex
@misc{chen2026curriculum,
  title         = {Curriculum Learning of Physics-Informed Neural Networks based on Spatial Correlation},
  author        = {Chen, Xujia and Hu, Xinyue and Chen, Letian and Shi, Daming and Fan, Wenhui},
  year          = {2026},
  eprint        = {2605.15254},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2605.15254}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
