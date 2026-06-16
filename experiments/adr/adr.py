from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence, Tuple

import deepxde as dde
import deepxde.config as ddeconfig
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata
from datetime import datetime

from experiments.utils import *
import pinn_pro


BASE_DIR = Path(__file__).resolve().parent


@dataclass
class ADRConfig:
    
    nu: float = 1e-4
    beta: float = 100.0
    cmin: float = 1.0
    cmax: float = 1e4
    xc: float = 0.65
    yc: float = 0.35
    sig_c: float = 0.10
    delta: float = 0.35
    x0: float = 0.60
    y0: float = 0.40
    sig_u: float = 0.12
    k: int = 8
    data_file: str = ""
    bbox: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
    
    num_domain: int = 10000
    num_boundary: int = 400
    num_test: int = 10000
    
    learning_rate: float = 5e-4
    opt_decay = ['step', 1000, 0.9]
    adam_iterations: int = 5000
    lbfgs_iterations: int = 5000
    net_width: int = 50
    net_depth: int = 5
    activation: str = "tanh"
    initializer: str = "Glorot normal"
    grid_resolution: int = 250
    loss_headers = ["PDE loss", "BC loss"]
    # loss_weights: Tuple[float, float] = (1.0, 100000.0)
    loss_weights: Tuple[float, float] = (1.0, 10000.0)
    train_distribution: str = "uniform"
    
    seed: int = 0
    
    reweight_config: dict = field(default_factory=lambda:{
        "decay_epsi": 0.5, # 0.5
        "num_subdomains": [5,5],
        "reweight_every": 500,
        # "reweight_causal_begin": 100000,
        # "reweight_causal_end": 100000,
        "reweight_adaptive_begin": 100000,
        "reweight_adaptive_end": 100000,
        "reweight_causal_begin": 1000,
        "reweight_causal_end": 5000,
        # "reweight_adaptive_begin": 5000,
        # "reweight_adaptive_end": 8000,
        "log": True,
        "scale": 5,
        "grad_norms_scale": 2,
        "low_fre_n": 2,
        "low_fre_data_weight": 1.0,
        "frame_data_weight": 0.0,
    })
    

class ADR:
    def __init__(self, config):
        self.config = config
        
        self.pde = None
        self.reference_fn = None
        # self.reference = None
        self.geom = None
        self.data = None
        self.net = None
        self.model = None
        
    def output_dir(self) -> Path:
        run_name = (
            # f"width_{self.config.net_width}_depth_{self.config.net_depth}_"
            # f"domain_{self.config.num_domain}_boundary_{self.config.num_boundary}_"
            f"adam_{self.config.adam_iterations}_lbfgs_{self.config.lbfgs_iterations}_"
            f"seed_{self.config.seed}_"
            f"decay_epsi_{self.config.reweight_config['decay_epsi']}_"
            f"data_weight_{self.config.reweight_config['low_fre_data_weight']}_"
            f"frame_weight_{self.config.reweight_config['frame_data_weight']}_"
            f"causal_begin_{self.config.reweight_config['reweight_causal_begin']}_"
            f"adaptive_begin_{self.config.reweight_config['reweight_adaptive_begin']}_"
        )
        return run_name
    
    def advection_field_np(self, points: np.ndarray):
        x = points[:, 0:1]
        y = points[:, 1:2]
        bx = self.config.beta * (-(y - 0.5))
        by = self.config.beta * (x - 0.5)
        return bx, by


    def advection_field_backend(self, points):
        xb = points[:, 0:1]
        yb = points[:, 1:2]
        bx = self.config.beta * (-(yb - 0.5))
        by = self.config.beta * (xb - 0.5)
        return bx, by


    def reaction_coeff_np(self, points: np.ndarray):
        x = points[:, 0:1]
        y = points[:, 1:2]
        r2 = (x - self.config.xc) ** 2 + (y - self.config.yc) ** 2
        return self.config.cmin + self.config.cmax * np.exp(-r2 / (self.config.sig_c ** 2))

    def reaction_coeff_backend(self, points):
        xb = points[:, 0:1]
        yb = points[:, 1:2]
        r2 = (xb - self.config.xc) ** 2 + (yb - self.config.yc) ** 2
        return self.config.cmin + self.config.cmax * dde.backend.exp(-r2 / (self.config.sig_c ** 2))
    
    def solution_components(self, points: np.ndarray):
        x = points[:, 0:1]
        y = points[:, 1:2]

        sx = np.sin(np.pi * x)
        cx = np.cos(np.pi * x)
        sy = np.sin(np.pi * y)
        cy = np.cos(np.pi * y)

        u_low = sx * sy
        u_low_x = np.pi * cx * sy
        u_low_y = np.pi * sx * cy
        u_low_xx = -(np.pi ** 2) * sx * sy
        u_low_yy = -(np.pi ** 2) * sx * sy

        s2 = self.config.sig_u ** 2
        invs2 = 1.0 / s2
        dx = x - self.config.x0
        dy = y - self.config.y0
        r2 = dx * dx + dy * dy
        G = np.exp(-r2 * invs2)

        C = np.cos(self.config.k * np.pi * x)
        S = np.sin(self.config.k * np.pi * x)

        Gx = G * (-2.0 * dx * invs2)
        Gy = G * (-2.0 * dy * invs2)
        Gxx = G * ((4.0 * dx * dx * (invs2 ** 2)) - 2.0 * invs2)
        Gyy = G * ((4.0 * dy * dy * (invs2 ** 2)) - 2.0 * invs2)

        Cx = -(self.config.k * np.pi) * S
        Cxx = -((self.config.k * np.pi) ** 2) * C

        u_packet = G * C
        u_packet_x = Gx * C + G * Cx
        u_packet_y = Gy * C
        u_packet_xx = Gxx * C + 2.0 * Gx * Cx + G * Cxx
        u_packet_yy = Gyy * C

        u = u_low + self.config.delta * u_packet
        u_x = u_low_x + self.config.delta * u_packet_x
        u_y = u_low_y + self.config.delta * u_packet_y
        lap_u = (u_low_xx + u_low_yy) + self.config.delta * (u_packet_xx + u_packet_yy)
        return u, u_x, u_y, lap_u
    

    def forcing_rhs(self, points):
        if isinstance(points, np.ndarray):
            u, u_x, u_y, lap_u = self.solution_components(points)
            bx, by = self.advection_field_np(points)
            c_val = self.reaction_coeff_np(points)
            return -self.config.nu * lap_u + bx * u_x + by * u_y + c_val * u

        points_np = points.detach().cpu().numpy()
        rhs_np = self.forcing_rhs(points_np)
        rhs_tensor = dde.backend.as_tensor(rhs_np, dtype=getattr(points, "dtype", None))
        point_device = getattr(points, "device", None)
        if point_device is not None and hasattr(rhs_tensor, "to"):
            rhs_tensor = rhs_tensor.to(point_device)
        return rhs_tensor

    def build_reference_fn(self):
        def exact_solution(points):
            if isinstance(points, np.ndarray):
                u, _, _, _ = self.solution_components(points)
                return u

            points_np = points.detach().cpu().numpy()
            values_np = exact_solution(points_np)
            tensor = dde.backend.as_tensor(values_np, dtype=getattr(points, "dtype", None))
            point_device = getattr(points, "device", None)
            if point_device is not None and hasattr(tensor, "to"):
                tensor = tensor.to(point_device)
            return tensor
        return exact_solution

    def build_pde(self):
        def pde(x, u):
            u_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_y = dde.grad.jacobian(u, x, i=0, j=1)
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, i=1, j=1)
            bx, by = self.advection_field_backend(x)
            c_val = self.reaction_coeff_backend(x)
            return -self.config.nu * (u_xx + u_yy) + bx * u_x + by * u_y + c_val * u - self.forcing_rhs(x)

        return pde

    def make_domain(self, config):
        return dde.geometry.Rectangle(xmin=[0.0, 0.0], xmax=[1.0, 1.0])

    def make_dataset_pinn(self, geom, reference_fn, config):
        bc = dde.icbc.DirichletBC(
            geom,
            lambda pts: reference_fn(pts),
            lambda _, on_boundary: on_boundary,
        )
        data = dde.data.PDE(
            geom,
            self.pde,
            [bc],
            num_domain=config.num_domain,
            num_boundary=config.num_boundary,
            train_distribution=config.train_distribution,
            solution=reference_fn,
        )
        enforce_dataset_dtype(data, ddeconfig.real(np))
        return data
    
    def make_dataset_pinn_weighted_samples(self, geom, reference_fn, config):
        bc = dde.icbc.DirichletBC(
            geom,
            lambda pts: reference_fn(pts),
            lambda _, on_boundary: on_boundary,
        )
        data = pinn_pro.PDEWeightedSamples(
            geom,
            self.pde,
            [bc],
            num_domain=config.num_domain,
            num_boundary=config.num_boundary,
            train_distribution=config.train_distribution,
            solution=reference_fn,
            reweight_config=config.reweight_config,
        )
        enforce_dataset_dtype(data, ddeconfig.real(np))
        return data
    
    def make_dataset_function(self, geom, reference_fn, config):
        num_train = config.num_domain + config.num_boundary
        num_test = config.eval_resolution ** 2
        data = dde.data.Function(
            geom,
            reference_fn,
            num_train=num_train,
            num_test=num_test,
            train_distribution=config.train_distribution,
        )
        enforce_dataset_dtype(data, ddeconfig.real(np))
        return data


    def make_network(self, config):
        layer_sizes = [2] + [config.net_width] * config.net_depth + [1]
        return dde.nn.FNN(layer_sizes, config.activation, config.initializer)

    def component_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        metrics["u_mse"] = float(dde.metrics.mean_squared_error(y_true, y_pred))
        metrics["u_l2_relative"] = float(dde.metrics.l2_relative_error(y_true, y_pred))
        metrics["u_max_abs"] = max_absolute_error(y_true, y_pred)
        return metrics

    def make_grid(self, config):
        xs = np.linspace(0.0, 1.0, config.grid_resolution)
        ys = np.linspace(0.0, 1.0, config.grid_resolution)
        grid_x, grid_y = np.meshgrid(xs, ys)
        coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        return grid_x, grid_y, coords

    def load_reference(self, ref_path: Path):
        grid_x, grid_y, coords = self.make_grid(self.config)
        reference_fn = self.build_reference_fn()
        values_true = reference_fn(coords)
        
        return coords, values_true
        
    
    def plot_fields(self, coords, values_true, grid_x, grid_y, grid_pred, save_path: Path, loss_record: str = ""):
        true_grid = griddata(coords, values_true.ravel(), (grid_x, grid_y), method="cubic")
        true_grid = np.nan_to_num(true_grid, nan=0.0)
        true_grid = true_grid.reshape(grid_x.shape)
        error_grid = np.abs(grid_pred - true_grid)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        extent = (self.config.bbox[0], self.config.bbox[1], self.config.bbox[2], self.config.bbox[3])
        titles = ["Predicted", "Reference", "|delta u|"]
        fields = [grid_pred, true_grid, error_grid]
        for ax, title, field in zip(axes, titles, fields):
            im = ax.imshow(field, cmap="viridis", extent=extent, origin="lower", aspect="auto")
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, shrink=0.8)
            
        fig.suptitle("Poisson2d VC" + loss_record, fontsize=10)
        # fig.tight_layout()
        fig.savefig(save_path)
        print(f"Saved field plots to {save_path}")
        plt.close(fig)

    def plot_pde_loss_grad(self, loss_weight_list, loss_list, grad_list, save_path: Path):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        n_x, n_y = self.config.reweight_config["num_subdomains"]
        weight_arr = np.asarray(loss_weight_list, dtype=float).reshape(n_x, n_y)
        loss_arr = np.asarray(loss_list, dtype=float).reshape(n_x, n_y)
        grad_arr = np.asarray(grad_list, dtype=float).reshape(n_x, n_y)

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

        def plot_grid(ax, data, title, cmap, fmt=".2e"):
            im = ax.imshow(data.T, origin="lower", cmap=cmap, aspect="equal")
            ax.set_xticks(np.arange(n_x))
            ax.set_yticks(np.arange(n_y))
            ax.set_xticklabels([f"{i}" for i in range(n_x)])
            ax.set_yticklabels([f"{j}" for j in range(n_y)])
            ax.set_xlabel("Subdomain i")
            ax.set_ylabel("Subdomain j")
            ax.set_title(title)
            for i in range(n_x):
                for j in range(n_y):
                    ax.text(i, j, format(data[i, j], fmt), ha="center", va="center", fontsize=6, color="k")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plot_grid(axes[0], weight_arr, "Loss weight per subdomain", "Greens", fmt=".2f")
        plot_grid(axes[1], loss_arr, "PDE loss per subdomain", "Blues")
        plot_grid(axes[2], grad_arr, "Grad norm per subdomain", "Reds")

        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        plt.close(fig)
        
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.cuda.set_device(device)
    
    num_runs = 3
    base_seed = 0
    seeds = [base_seed + i for i in range(num_runs)]
    
    batch_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = BASE_DIR / f"multi_runs" / f"batch_{batch_tag}"
    run_root.mkdir(parents=True, exist_ok=True)
    
    for run_idx, seed in enumerate(seeds):
        if seed is None:
            continue
        config = ADRConfig()
        config.seed = int(seed)
        adr_model = ADR(config)
        run_base_dir = run_root / f"run_{run_idx:02d}_seed_{config.seed}"
        run_base_dir.mkdir(parents=True, exist_ok=True)

        pinn_space_weight = pinn_pro.PINNWeightedSamples(adr_model, run_base_dir)
        pinn_space_weight.train_and_evaluate()
    
    # run_root = BASE_DIR / f"multi_runs" / f"batch_20260329_015121"
    summarize_batch(run_root)
