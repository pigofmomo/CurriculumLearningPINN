# Curriculum Learning PINN

This repository contains the public research implementation for the arXiv paper **[Curriculum Learning of Physics-Informed Neural Networks based on Spatial Correlation](https://arxiv.org/abs/2605.15254)**.

The code studies curriculum strategies for Physics-Informed Neural Networks (PINNs). It is organized around spatially structured training, low-frequency bridge constraints, and region-adaptive reweighting for boundary value problems and spatially coupled PDE systems.

> **Repository status:** this is a research code release. The core experiment runners and reusable PINN components are included; APIs may still change while the paper artifacts are finalized.

---

## Highlights

- **Spatial curriculum learning**: partitions the computational domain into subregions and adjusts training emphasis according to spatial relationships.
- **Low-frequency information bridge**: uses sparse anchor information to stabilize global low-frequency behavior across separated regions.
- **Region-adaptive reweighting**: updates subregion-wise PDE losses based on local residual and optimization status.
- **Reproducible artifacts**: experiment runners save configurations, losses, metrics, checkpoints, logs, and result figures under per-experiment run directories.

---

## Repository layout

```text
.
├── pinn_pro/                    # Reusable PINN runners, weighted-sample model, data utilities, domain decomposition
├── experiments/
│   ├── ode1d/                   # One-dimensional ODE benchmark
│   ├── poisson_high_fre/        # High-frequency 2D Poisson benchmark
│   ├── adr/                     # 2D advection-diffusion-reaction benchmark
│   └── ns2d/                    # 2D lid-driven Navier-Stokes benchmark and reference data
└── README.md
```

Key modules:

- `pinn_pro/pinn.py`: training/evaluation wrappers for baseline PINN and weighted-sample curriculum PINN runs.
- `pinn_pro/model.py`: DeepXDE/PyTorch-compatible model implementation for weighted collocation groups.
- `pinn_pro/data.py`: data containers and loss construction for grouped residual, boundary, and supervised terms.
- `pinn_pro/domain_decomp.py`: spatial partitioning helpers used by curriculum weighting.
- `experiments/utils.py`: shared utilities for references, metrics, logging, plotting, and artifact saving.

---

## Installation

The implementation uses Python, the PyTorch backend, and code paths that reuse/debug DeepXDE internals. For paper reproduction, prefer a local editable DeepXDE checkout instead of installing DeepXDE directly from PyPI. The development environment used DeepXDE around the `1.11.x` series; newer versions may work if the reused APIs remain compatible.

```bash
# 1. Create and activate an environment, for example:
conda create -n clpinn python=3.10 -y
conda activate clpinn

# 2. Install numerical and plotting dependencies.
pip install torch numpy scipy matplotlib

# 3. Put DeepXDE in a local sibling directory so it can be inspected/debugged.
#    Replace the tag with the exact local version you want to reproduce.
git clone https://github.com/lululxvi/deepxde.git ../deepxde
cd ../deepxde
git checkout v1.11.0
pip install -e .
cd ../CurriculumLearningPINN

# 4. Select the PyTorch backend for DeepXDE.
export DDE_BACKEND=pytorch
```

If you already have a local DeepXDE source tree, install that tree with `pip install -e /path/to/deepxde` or ensure it is earlier on `PYTHONPATH` than any site-packages DeepXDE installation. If you use CUDA, install the PyTorch build that matches your driver and CUDA toolkit before running the experiments.

---

## Quick start

Run commands from the repository root.

```bash
export DDE_BACKEND=pytorch
python experiments/ode1d/ode1d.py
```

The ODE example is the smallest benchmark and is useful for checking that the environment works. Larger 2D experiments can be launched with:

```bash
python experiments/poisson_high_fre/poisson_high_fre.py
python experiments/adr/adr_cli.py --cuda-device 0
python experiments/ns2d/ns2d_cli.py --cuda-device 0
```

The CLI scripts expose curriculum hyperparameters such as the number of subdomains, reweighting interval, low-frequency bridge order, and data weights. Use `--help` to inspect available options:

```bash
python experiments/adr/adr_cli.py --help
python experiments/ns2d/ns2d_cli.py --help
```

---

## Outputs

Experiment outputs are written inside each experiment directory, for example:

- `runs_pinn/`: baseline PINN runs.
- `runs_pinn_weighted_samples/`: curriculum weighted-sample runs.
- `logs/`: timestamped CLI logs for experiments that use CLI runners.

A typical run directory contains:

- `config.json`: experiment configuration and device summary.
- `loss.dat`: training loss history.
- `metrics.json`: evaluation metrics and PDE residual summaries.
- `results.png`: visualization of prediction/reference fields where available.
- DeepXDE model checkpoint files.

Generated run artifacts are intentionally ignored by Git so that the repository remains lightweight.

---

## Benchmarks included

| Benchmark | Entry point | Notes |
| --- | --- | --- |
| 1D ODE | `python experiments/ode1d/ode1d.py` | Lightweight sanity-check case. |
| High-frequency Poisson | `python experiments/poisson_high_fre/poisson_high_fre.py` | 2D benchmark with localized high-frequency structure. |
| ADR | `python experiments/adr/adr_cli.py` | 2D advection-diffusion-reaction case with CLI hyperparameters. |
| Lid-driven NS2D | `python experiments/ns2d/ns2d_cli.py` | 2D Navier-Stokes case with included reference data. |

---

## Reproducing paper experiments

The current defaults are intended to document the main experimental setup used during development. For final paper reproduction:

1. Set `DDE_BACKEND=pytorch`.
2. Fix random seeds through the experiment configuration.
3. Run the desired benchmark entry point from the repository root.
4. Archive the generated `config.json`, `metrics.json`, `loss.dat`, and result figures for each run.

Additional plotting scripts and exact paper command presets may be added as the paper release is finalized.

---

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

A license file has not yet been added. Please contact the authors before redistributing or using this code in downstream projects.
