"""One-dimensional ODE benchmark for curriculum PINN experiments. / 一维ODE课程学习PINN基准实验。"""

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Tuple, Union

import torch
import deepxde as dde
import deepxde.config as ddeconfig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.utils import *
import pinn_pro

BASE_DIR = Path(__file__).resolve().parent

@dataclass
class OdeConfig_l:
    # pde parameters
    K_VALUE = 10 * np.pi
    ALPHA_VALUE = 2.0
    
    # geometry
    domain: Tuple[float, float] = (0.0, 1.0)
    
    # sampling points num 
    num_domain: int = 1000
    num_boundary: int = 2
    num_test: int = 1000
    
    # training configuration
    net_width: int = 100
    net_depth: int = 6
    learning_rate: float = 1e-3
    opt_decay = ['step', 2000, 0.9]
    adam_iterations: int = 10000
    lbfgs_iterations: int = 0
    train_distribution: str = "uniform"
    loss_weight_bc = 100
    loss_weights: Tuple[float, float, float] = (1, loss_weight_bc, loss_weight_bc)
    loss_headers = ["PDE Loss", "BC Loss left", "BC Loss right"]
    # random seed
    seed: int = 0
    # 每个实例独立的可变默认值
    reweight_config: dict = field(default_factory=lambda: {
        "decay_epsi": 1.0,
        "num_subdomains": 5,
        "reweight_every": 1000,
        "reweight_causal_begin": 1000,
        "reweight_causal_end": 5000,
        "reweight_adaptive_begin": 100000,
        "reweight_adaptive_end": 100000,
        "record_every": 1000,
        "low_fre_n": 2,
        "low_fre_data_weight": 0.0,
        "frame_data_weight": 0.0,
    })
    
@dataclass
class OdeConfig_h:
    # pde parameters 
    K_VALUE = 20 * np.pi
    ALPHA_VALUE = 3.0
    
    # geometry
    domain: Tuple[float, float] = (0.0, 1.0)
    
    # sampling points num 
    num_domain: int = 1000
    num_boundary: int = 2
    num_test: int = 1000
    
    # training configuration
    net_width: int = 100
    net_depth: int = 6
    learning_rate: float = 1e-3
    opt_decay = ['step', 2000, 0.9]
    adam_iterations: int = 10000
    lbfgs_iterations: int = 6000
    train_distribution: str = "uniform"
    loss_weight_bc = 1000
    loss_weights: Tuple[float, float, float] = (1, loss_weight_bc, loss_weight_bc)
    loss_headers = ["PDE Loss", "BC Loss left", "BC Loss right"]
    # random seed
    seed: int = 0
    # 每个实例独立的可变默认值
    reweight_config: dict = field(default_factory=lambda: {
        "decay_epsi": 1.0,
        "num_subdomains": 5,
        "reweight_every": 1000,
        "reweight_causal_begin": 1000,
        "reweight_causal_end": 8000,
        "reweight_adaptive_begin": 100000,
        "reweight_adaptive_end": 100000,
        "low_fre_n": 2,
        "low_fre_data_weight": 10.0,
        "frame_data_weight": 0.0,
    })

class Ode1D:
    def __init__(self, config: OdeConfig_l):
        self.config = config
        
        self.pde = None
        self.reference_fn = self.build_reference_fn()
        x = np.linspace(*config.domain, config.num_test)[:, None]
        self.reference = (x, self.reference_fn(x))
        self.geom = None
        self.data = None
        self.net = None
        self.model = None

    def output_dir(self) -> Path:
        run_name = (
            f"k_{self.config.K_VALUE:.1f}__alpha_{self.config.ALPHA_VALUE:.1f}_"
            f"weight_bc_{self.config.loss_weight_bc}_"
            # f"width_{self.config.net_width}_depth_{self.config.net_depth}_"
            # f"num_domain_{self.config.num_domain}_"
            # f"adam_{self.config.adam_iterations}_lbfgs_{self.config.lbfgs_iterations}_"
            f"random_seed_{self.config.seed}_"
            f"decay_epsi_{self.config.reweight_config['decay_epsi']}_"
            f"data_weight_{self.config.reweight_config['low_fre_data_weight']}"
        )
        return run_name
    
    def build_reference_fn(self):
        ALPHA_VALUE = self.config.ALPHA_VALUE
        K_VALUE = self.config.K_VALUE
        def exact_solution(x):
            return x * (1.0 - x) * np.exp(ALPHA_VALUE * x) * np.sin(K_VALUE * x)
        return exact_solution

    def build_pde(self):
        ALPHA_VALUE = self.config.ALPHA_VALUE
        K_VALUE = self.config.K_VALUE
        
        def ode(x, y):
            y_xx = dde.grad.hessian(y, x)

            exp_ax = dde.backend.exp(ALPHA_VALUE * x)

            g1 = 1.0 - x - x**2 + ALPHA_VALUE * (x - x**2)
            g2 = (
                -2.0
                + 2.0 * ALPHA_VALUE * (1.0 - 2.0 * x)
                + (ALPHA_VALUE**2 - K_VALUE**2) * (x - x**2)
            )

            f = exp_ax * (
                g2 * dde.backend.sin(K_VALUE * x)
                + 2.0 * K_VALUE * g1 * dde.backend.cos(K_VALUE * x)
            )

            return y_xx - f
        
        return ode
    
    def boundary_l(self, x, on_boundary):
        return on_boundary and dde.utils.isclose(x[0], 0.0)

    def boundary_r(self, x, on_boundary):
        return on_boundary and dde.utils.isclose(x[0], 1.0)

    def make_domain(self, config):
        return dde.geometry.Interval(*config.domain)

    def make_dataset_pinn(self, geom, reference_fn, config: OdeConfig_l):
        bc1 = dde.icbc.DirichletBC(geom, lambda x: 0, self.boundary_l)
        bc2 = dde.icbc.DirichletBC(geom, lambda x: 0, self.boundary_r)
        data = dde.data.PDE(
            geom,
            self.pde,
            [bc1, bc2],
            num_domain=config.num_domain,
            num_boundary=config.num_boundary,
            solution=reference_fn,
            num_test=config.num_test,
            train_distribution=config.train_distribution,
        )
        enforce_dataset_dtype(data, ddeconfig.real(np))
        return data

    def make_dataset_pinn_weighted_samples(self, geom, reference_fn, config: OdeConfig_l):
        bc1 = dde.icbc.DirichletBC(geom, lambda x: 0, self.boundary_l)
        bc2 = dde.icbc.DirichletBC(geom, lambda x: 0, self.boundary_r)
        
        data = pinn_pro.PDEWeightedSamples(
            geometry=geom,
            pde=self.pde,
            bcs=[bc1, bc2],
            num_domain=config.num_domain,
            num_boundary=config.num_boundary,
            train_distribution=config.train_distribution,
            solution=reference_fn,
            num_test=config.num_test,
            reweight_config=config.reweight_config,
        )
        
        enforce_dataset_dtype(data, ddeconfig.real(np))
        return data
    
    def make_dataset_function(self, geom, reference_fn, config: OdeConfig_l):
        num_train = config.num_domain + config.num_boundary
        data = dde.data.Function(
            geom,
            reference_fn,
            num_train=num_train,
            num_test=config.num_test,
            train_distribution=config.train_distribution,
        )
        enforce_dataset_dtype(data, ddeconfig.real(np))
        return data

    def make_network(self, config: OdeConfig_l):
        layers = [1] + [config.net_width] * config.net_depth + [1]
        return dde.nn.FNN(layers, "tanh", "Glorot uniform")


    def component_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        result: Dict[str, float] = {}
        result["u_mse"] = float(dde.metrics.mean_squared_error(y_true, y_pred))
        result["u_l2_relative"] = float(dde.metrics.l2_relative_error(y_true, y_pred))
        result["u_max_abs"] = max_absolute_error(y_true, y_pred)
        return result

    def plot_fields(self, x, y_true, y_pred, save_path: Path, loss_record: str = ""):

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(x, y_true, label="Exact", linewidth=2.0)
        axes[0].plot(x, y_pred, "--", label="Prediction")
        axes[0].set_xlabel(r"$x$")
        axes[0].set_ylabel(r"$y(x)$")
        axes[0].set_title(r"$y(x)$")
        axes[0].legend()

        axes[1].plot(x, np.abs(y_true - y_pred), color="tab:red")
        axes[1].set_xlabel(r"$x$")
        axes[1].set_ylabel(r"$|y_{\mathrm{true}} - y_{\mathrm{pred}}|$")
        axes[1].set_title("Absolute error")

        fig.suptitle("ODE 1D " + loss_record, fontsize=10)
        # fig.tight_layout()
        fig.subplots_adjust(wspace=0.25)
        fig.savefig(save_path)
        plt.close(fig)

    def plot_pde_loss_grad(self, loss_weight_list, loss_list, grad_list, save_path: Path):
        # 直方图展示每个子域的 PDE loss 与梯度强度
        save_path.parent.mkdir(parents=True, exist_ok=True)
        weight_arr = np.asarray(loss_weight_list, dtype=float)
        loss_arr = np.asarray(loss_list, dtype=float)
        grad_arr = np.asarray(grad_list, dtype=float)
        num_subdomains = len(loss_arr)

        fig, axes = plt.subplots(3, 1, figsize=(5, 6), sharex=True)

        axes[0].bar(np.arange(num_subdomains), weight_arr, color="tab:green")
        axes[0].set_ylabel("Loss weight")
        axes[0].set_title("Per-subdomain loss weight")

        axes[1].bar(np.arange(num_subdomains), loss_arr)
        axes[1].set_ylabel("PDE loss")
        axes[1].set_title("Per-subdomain PDE loss")

        axes[2].bar(np.arange(num_subdomains), grad_arr, color="tab:red")
        axes[2].set_xlabel("Subdomain index")
        axes[2].set_ylabel("Gradient norm")
        axes[2].set_title("Per-subdomain gradient norm")

        # fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)    

if __name__ == "__main__":
    device = torch.device("cuda:0")
    print(f"Using device: {device}")
    torch.cuda.set_device(device)
    
    # frequency = "low"  # "low" or "high"
    frequency = "high"
    num_runs = 3
    base_seed = 0
    seeds = [base_seed + i for i in range(num_runs)]
    # seeds = [2]

    batch_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = BASE_DIR / f"multi_runs_{frequency}_frequency" / f"batch_{batch_tag}"
    run_root.mkdir(parents=True, exist_ok=True)

    for run_idx, seed in enumerate(seeds):
        if frequency == "low":
            config = OdeConfig_l()
        else:
            config = OdeConfig_h()
        if seed is None:
            continue

        config.seed = int(seed)
        ODE_model = Ode1D(config)
        run_base_dir = run_root / f"run_{run_idx:02d}_seed_{config.seed}"
        run_base_dir.mkdir(parents=True, exist_ok=True)

        pinn_space_weight = pinn_pro.PINNWeightedSamples(ODE_model, run_base_dir)
        pinn_space_weight.train_and_evaluate()
        
    summarize_batch(run_root)
