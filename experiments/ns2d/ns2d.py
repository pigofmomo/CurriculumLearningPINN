from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple
import torch
import deepxde as dde
import deepxde.config as ddeconfig
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from datetime import datetime
from experiments.utils import *
import pinn_pro

BASE_DIR = Path(__file__).resolve().parent

@dataclass
class NS2DConfig:
    # pde parameters
    reynolds: float = 100.0
    lid_coefficient: float = 8
    data_file: str = BASE_DIR / f"data/lid_driven_a{int(lid_coefficient)}.dat"
    
    # geometry
    spatial_domain = ([0.0, 0.0], [1.0, 1.0])
    
    # sampling points num 
    num_domain: int = 2500
    num_boundary: int = 400
    
    # training configuration
    learning_rate: float = 1e-3
    adam_iterations: int = 5000
    lbfgs_iterations: int = 3000
    opt_decay = ['step', 1000, 0.9]
    net_width: int = 60
    net_depth: int = 6
    grid_resolution: int = 120
    # loss_weights: Tuple[float, float, float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    loss_weights: Tuple[float, float, float, float, float, float, float, float] = (1.0, 1.0, 1.0, 100.0, 100.0, 100.0, 100.0, 100.0)
    loss_headers = ["loss momentum x", "loss momentum y", "loss continuity",
                    "loss BC u top", "loss BC v top",
                    "loss BC u other", "loss BC v other",
                    "loss BC p fix"]
    train_distribution: str = "uniform"
    
    # random seed
    seed: int = 0
    
    reweight_config: dict = field(default_factory=lambda: {
        "decay_epsi": 1.0, # 0.5
        "num_subdomains": [5,5],
        "reweight_every": 500,
        "reweight_causal_begin": 1000,
        "reweight_causal_end": 5000,
        "reweight_adaptive_begin": 100000,
        "reweight_adaptive_end": 100000,
        # "reweight_adaptive_begin": 6000,
        # "reweight_adaptive_end": 8000,
        "log": True,
        "scale": 2,
        "grad_norms_scale": 1,
        "low_fre_n": 3,
        "low_fre_data_weight": 0.0,
        "frame_data_weight": 0.0,
    })

  
class NS2D:
    def __init__(self, config: NS2DConfig):
        self.config = config
        
        self.pde = None
        self.reference_fn = None
        self.geom = None
        self.data = None
        self.net = None
        self.model = None
    
    def output_dir(self) -> Path:
        run_name = (
            # f"Re_{self.config.reynolds:.0f}_a_{self.config.lid_coefficient:.0f}_"
            # f"width_{self.config.net_width}_depth_{self.config.net_depth}_"
            f"adam_{self.config.adam_iterations}_lbfgs_{self.config.lbfgs_iterations}_"
            f"random_seed_{self.config.seed}_"
            f"decay_epsi_{self.config.reweight_config['decay_epsi']}_"
            f"data_weight_{self.config.reweight_config['low_fre_data_weight']}"
            f"frame_weight_{self.config.reweight_config['frame_data_weight']}_"
            f"reweight_causal_{self.config.reweight_config['reweight_causal_begin']}_{self.config.reweight_config['reweight_causal_end']}_"
            f"reweight_adaptive_{self.config.reweight_config['reweight_adaptive_begin']}_{self.config.reweight_config['reweight_adaptive_end']}"
        )
        return run_name
    
    def build_pde(self):
        reynolds = self.config.reynolds
        inv_re = 1.0 / reynolds

        def pde(x, u):
            u_vel, v_vel = u[:, 0:1], u[:, 1:2]
            u_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_y = dde.grad.jacobian(u, x, i=0, j=1)
            u_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

            v_x = dde.grad.jacobian(u, x, i=1, j=0)
            v_y = dde.grad.jacobian(u, x, i=1, j=1)
            v_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
            v_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

            p_x = dde.grad.jacobian(u, x, i=2, j=0)
            p_y = dde.grad.jacobian(u, x, i=2, j=1)

            momentum_x = u_vel * u_x + v_vel * u_y + p_x - inv_re * (u_xx + u_yy)
            momentum_y = u_vel * v_x + v_vel * v_y + p_y - inv_re * (v_xx + v_yy)
            continuity = u_x + v_y
            return [momentum_x, momentum_y, continuity]

        return pde


    def boundary_top(self, x, on_boundary):
        return on_boundary and np.isclose(x[1], 1.0)


    def boundary_not_top(self, x, on_boundary):
        return on_boundary and not np.isclose(x[1], 1.0)


    def lid_velocity(self, a: float):
        def velocity(x):
            return a * x[:, 0:1] * (1.0 - x[:, 0:1])
        return velocity

    def make_domain(self, config):
        geom = dde.geometry.Rectangle(config.spatial_domain[0], config.spatial_domain[1])
        return geom
    
    def make_dataset_pinn(self, geom, reference_fn, config: NS2DConfig):
        bc_u_top = dde.DirichletBC(geom, self.lid_velocity(config.lid_coefficient), self.boundary_top, component=0)
        bc_v_top = dde.DirichletBC(geom, lambda _: 0.0, self.boundary_top, component=1)
        bc_u_other = dde.DirichletBC(geom, lambda _: 0.0, self.boundary_not_top, component=0)
        bc_v_other = dde.DirichletBC(geom, lambda _: 0.0, self.boundary_not_top, component=1)
        bc_p = dde.PointSetBC(np.array([[0.0, 0.0]]), np.array([[0.0]]), component=2)
        data = dde.data.PDE(
            geom,
            self.pde,
            [bc_u_top, bc_v_top, bc_u_other, bc_v_other, bc_p],
            num_domain=config.num_domain,
            num_boundary=config.num_boundary,
            train_distribution=config.train_distribution,
            solution=reference_fn
        )
        enforce_dataset_dtype(data, ddeconfig.real(np))
        return data
    
    def make_dataset_pinn_weighted_samples(self, geom, reference_fn, config: NS2DConfig):
        bc_u_top = dde.DirichletBC(geom, self.lid_velocity(config.lid_coefficient), self.boundary_top, component=0)
        bc_v_top = dde.DirichletBC(geom, lambda _: 0.0, self.boundary_top, component=1)
        bc_u_other = dde.DirichletBC(geom, lambda _: 0.0, self.boundary_not_top, component=0)
        bc_v_other = dde.DirichletBC(geom, lambda _: 0.0, self.boundary_not_top, component=1)
        bc_p = dde.PointSetBC(np.array([[0.0, 0.0]]), np.array([[0.0]]), component=2)
        data = pinn_pro.PDEWeightedSamples(
            geom,
            self.pde,
            [bc_u_top, bc_v_top, bc_u_other, bc_v_other, bc_p],
            num_domain=config.num_domain,
            num_boundary=config.num_boundary,
            train_distribution=config.train_distribution,
            solution=reference_fn,
            reweight_config=config.reweight_config,
        )
        enforce_dataset_dtype(data, ddeconfig.real(np))
        return data
    
    def make_dataset_function(self, geom, reference_fn, config: NS2DConfig):
        num_train = config.num_domain + config.num_boundary
        num_test = config.grid_resolution ** 2
        data = dde.data.Function(
            geom,
            reference_fn,
            num_train=num_train,
            num_test=num_test,
            train_distribution=config.train_distribution,
        )
        enforce_dataset_dtype(data, ddeconfig.real(np))
        return data
    
    def make_network(self, config: NS2DConfig):
        layer_sizes = [2] + [config.net_width] * config.net_depth + [3]
        return dde.nn.FNN(layer_sizes, "sin", "Glorot normal")

    def load_reference(self, ref_path: Path):
        ref_path = Path(ref_path)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference data not found at {ref_path}")
        data = np.loadtxt(ref_path, comments="%")
        if data.shape[1] < 5:
            raise ValueError("Reference file must contain columns: x, y, u, v, p")
        coords = data[:, :2]
        values = data[:, 2:5]
        return coords, values

    def component_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        labels = ["u", "v", "p"]
        result: Dict[str, float] = {}
        for idx, label in enumerate(labels):
            true_comp = y_true[:, idx : idx + 1]
            pred_comp = y_pred[:, idx : idx + 1]
            result[f"{label}_mse"] = float(dde.metrics.mean_squared_error(true_comp, pred_comp))
            result[f"{label}_l2_relative"] = float(dde.metrics.l2_relative_error(true_comp, pred_comp))
            result[f"{label}_max_abs"] = max_absolute_error(true_comp, pred_comp)
        return result


    def make_grid(self, config: NS2DConfig):
        x_range = [config.spatial_domain[0][0], config.spatial_domain[1][0]]
        y_range = [config.spatial_domain[0][1], config.spatial_domain[1][1]]
        xs = np.linspace(x_range[0], x_range[1], config.grid_resolution)
        ys = np.linspace(y_range[0], y_range[1], config.grid_resolution)
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        return grid_x, grid_y, grid_points

    def plot_fields(self, coords, values_true, grid_x, grid_y, grid_pred, save_path: Path, loss_record: str = ""):
        labels = ["u", "v", "p"]

        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        ref_fields = []
        pred_fields = []
        
        def plot_heatmap(ax, grid_x, grid_y, values, title: str):
            im = ax.pcolormesh(grid_x, grid_y, values, shading="auto", cmap="RdBu_r")
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            ax.set_title(title)
            fig = ax.get_figure()
            fig.colorbar(im, ax=ax, shrink=0.8)
        
        def plot_streamlines(ax, grid_x, grid_y, u_field, v_field, title: str):
            u_clean = np.nan_to_num(u_field)
            v_clean = np.nan_to_num(v_field)
            x_axis = grid_x[0, :]
            y_axis = grid_y[:, 0]
            ax.streamplot(x_axis, y_axis, u_clean, v_clean, color="k", linewidth=1, density=3)
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            ax.set_title(title, fontsize=10)
            # add a thin, empty colorbar to match heatmap widths
            sm = plt.cm.ScalarMappable(cmap="Greys")
            sm.set_array([])
            cbar = ax.get_figure().colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_ticks([])
            cbar.outline.set_visible(False)
            
        for idx, label in enumerate(labels):
            ref_grid = np.nan_to_num(griddata(coords, values_true[:, idx], (grid_x, grid_y), method="cubic"))
            pred_grid = grid_pred[:, :, idx]
            error_grid = np.nan_to_num(np.abs(pred_grid - ref_grid))

            ref_fields.append(ref_grid)
            pred_fields.append(pred_grid)

            plot_heatmap(axes[idx, 0], grid_x, grid_y, ref_grid, f"True {label}")
            plot_heatmap(axes[idx, 1], grid_x, grid_y, pred_grid, f"Pred {label}")
            plot_heatmap(axes[idx, 2], grid_x, grid_y, error_grid, f"|Δ{label}|")

        plot_streamlines(axes[3, 0], grid_x, grid_y, ref_fields[0], ref_fields[1], "Reference velocity field")
        plot_streamlines(axes[3, 1], grid_x, grid_y, pred_fields[0], pred_fields[1], "Predicted velocity field")
        axes[3, 2].axis("off")

        fig.suptitle("2D Navier-Stokes Fields " + loss_record, fontsize=10)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.35, hspace=0.35)
        fig.savefig(save_path)
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
    
if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
        config = NS2DConfig()
        config.seed = int(seed)
        ns2d_model = NS2D(config)
        run_base_dir = run_root / f"run_{run_idx:02d}_seed_{config.seed}"
        run_base_dir.mkdir(parents=True, exist_ok=True)

        pinn_space_weight = pinn_pro.PINNWeightedSamples(ns2d_model, run_base_dir)
        pinn_space_weight.train_and_evaluate()
    
    # run_root = BASE_DIR / f"multi_runs_cli" / f"batch_20260509_020219"
    # summarize_batch(run_root)
