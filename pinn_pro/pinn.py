"""Training wrappers for baseline and curriculum PINN variants. / 基线与课程学习PINN的训练封装。"""

import deepxde as dde
import deepxde.config as ddeconfig
from experiments.utils import *
from dataclasses import asdict
import pinn_pro
import matplotlib.pyplot as plt
import numpy as np

class FunctionInterpolator:
    def __init__(self, PDE, BASE_DIR: Path):
        self.PDE = PDE
        self.RUN_ROOT = BASE_DIR / "runs_function"
        self.run_dir = self.RUN_ROOT / self.PDE.output_dir()
        
    def train_and_evaluate(self):
        ensure_run_root(self.run_dir)
        ddeconfig.set_random_seed(self.PDE.config.seed)
        self.PDE.pde = self.PDE.build_pde()
        if self.PDE.reference_fn is None:
            self.PDE.reference = self.PDE.load_reference(self.PDE.config.data_file)
            self.PDE.reference_fn = build_reference_function(
                self.PDE.reference[0], self.PDE.reference[1]
            )
        self.PDE.geom = self.PDE.make_domain(self.PDE.config)
        self.PDE.data = self.PDE.make_dataset_function(self.PDE.geom, 
                                                       self.PDE.reference_fn, self.PDE.config)
        self.PDE.net = self.PDE.make_network(self.PDE.config)
        self.PDE.model = dde.Model(self.PDE.data, self.PDE.net)
        
        
        losshistory, train_state = self.optimize()
        self.save_artifacts(losshistory, train_state)
        self.evaluate()
    
    def optimize(self):
        self.PDE.model.compile("adam", lr=self.PDE.config.learning_rate, decay=self.PDE.config.opt_decay)
        losshistory, train_state = self.PDE.model.train(iterations=self.PDE.config.adam_iterations,
                                                        display_every=500)
        if self.PDE.config.lbfgs_iterations > 0:
            dde.optimizers.set_LBFGS_options(maxiter=self.PDE.config.lbfgs_iterations)
            self.PDE.model.compile("L-BFGS-B")
            losshistory, train_state = self.PDE.model.train()
            
        return losshistory, train_state
    
    def evaluate(self):
        
        coords, values_true = self.PDE.reference
        values_pred = self.PDE.model.predict(coords)
        metrics = self.PDE.component_metrics(values_true, values_pred)
        save_metrics(self.run_dir / "metrics.json", metrics)

        if hasattr(self.PDE, "make_grid"):
            grid_x, grid_y, grid_points = self.PDE.make_grid(self.PDE.config)
            grid_pred = self.PDE.model.predict(grid_points)
            if grid_pred.shape[1] == 1:
                grid_pred = grid_pred.reshape(grid_x.shape)
            else:
                grid_pred = grid_pred.reshape(grid_x.shape + (grid_pred.shape[1],))
            self.PDE.plot_fields(coords, values_true, 
                             grid_x, grid_y, grid_pred, self.run_dir / "results.png")
        else:
            self.PDE.plot_fields(coords, values_true, values_pred,
                             self.run_dir / "results.png")
            
        prefix = "[Function]"
        for key, value in metrics.items():
            print(f"{prefix} {key}: {value:.6e}")
            
        
    def save_artifacts(self, losshistory, train_state):
        self.PDE.model.save(self.run_dir)
        cfg_dict = asdict(self.PDE.config)
        cfg_dict.update(device_summary())
        save_loss_history(losshistory, self.run_dir / "loss.dat")
        save_config(self.run_dir / "config.json", cfg_dict)
        
    
class PINN:
    def __init__(self, PDE, BASE_DIR: Path):
        self.PDE = PDE
        self.RUN_ROOT = BASE_DIR / "runs_pinn"
        self.run_dir = self.RUN_ROOT / self.PDE.output_dir()
    
    def train_and_evaluate(self):
        ensure_run_root(self.run_dir)
        ddeconfig.set_random_seed(self.PDE.config.seed)
        self.PDE.pde = self.PDE.build_pde()
        if self.PDE.reference_fn is None:
            self.PDE.reference = self.PDE.load_reference(self.PDE.config.data_file)
            self.PDE.reference_fn = build_reference_function(
                self.PDE.reference[0], self.PDE.reference[1]
            )
        self.PDE.geom = self.PDE.make_domain(self.PDE.config)
        self.PDE.data = self.PDE.make_dataset_pinn(self.PDE.geom, 
                                                       self.PDE.reference_fn, self.PDE.config)
        self.PDE.net = self.PDE.make_network(self.PDE.config)
        self.PDE.model = dde.Model(self.PDE.data, self.PDE.net)
        
        losshistory, train_state = self.optimize()
        self.save_artifacts(losshistory, train_state)
        self.evaluate()
    
    def optimize(self):
        reweight_every = self.PDE.config.reweight_every
        dde.optimizers.LBFGS_options["iter_per_step"] = reweight_every
        self.PDE.model.compile("adam", lr=self.PDE.config.learning_rate, 
                               loss_weights=self.PDE.config.loss_weights, decay=self.PDE.config.opt_decay)
        losshistory, train_state = self.PDE.model.train(iterations=self.PDE.config.adam_iterations,
                                                        display_every=500)
        if self.PDE.config.lbfgs_iterations > 0:
            dde.optimizers.set_LBFGS_options(maxiter=self.PDE.config.lbfgs_iterations)
            self.PDE.model.compile("L-BFGS-B", loss_weights=self.PDE.config.loss_weights)
            losshistory, train_state = self.PDE.model.train()
        return losshistory, train_state
    
    def evaluate(self):
        coords, values_true = self.PDE.reference
        values_pred = self.PDE.model.predict(coords)
        metrics = self.PDE.component_metrics(values_true, values_pred)
        residual = self.PDE.model.predict(coords, operator=self.PDE.pde)
        metrics["pde_residual"] = float(np.mean(np.abs(residual)))
        save_metrics(self.run_dir / "metrics.json", metrics)

        if hasattr(self.PDE, "make_grid"):
            grid_x, grid_t, grid_points = self.PDE.make_grid(self.PDE.config)
            grid_pred = self.PDE.model.predict(grid_points)
            if grid_pred.shape[1] == 1:
                grid_pred = grid_pred.reshape(grid_x.shape)
            else:
                grid_pred = grid_pred.reshape(grid_x.shape + (grid_pred.shape[1],))
            residual_grid = self.PDE.model.predict(grid_points, operator=self.PDE.pde)
            metrics["pde_residual_grid"] = float(np.mean(np.abs(residual_grid)))
            self.PDE.plot_fields(coords, values_true, 
                             grid_x, grid_t, grid_pred, self.run_dir / "results.png")
        
        else:
            self.PDE.plot_fields(coords, values_true, values_pred,
                             self.run_dir / "results.png")
            
        prefix = "[PINN]"
        for key, value in metrics.items():
            print(f"{prefix} {key}: {value:.6e}")
            
        
    
    def save_artifacts(self, losshistory, train_state):
        self.PDE.model.save(self.run_dir)
        cfg_dict = asdict(self.PDE.config)
        cfg_dict.update(device_summary())
        save_loss_history(losshistory, self.run_dir / "loss.dat")
        save_config(self.run_dir / "config.json", cfg_dict)

class PINNWeightedSamples:
    def __init__(self, PDE, BASE_DIR: Path):
        self.PDE = PDE
        self.RUN_ROOT = BASE_DIR / "runs_pinn_weighted_samples"
        self.run_dir = self.RUN_ROOT / self.PDE.output_dir()
    
    def train_and_evaluate(self):
        ensure_run_root(self.run_dir)
        ddeconfig.set_random_seed(self.PDE.config.seed)
        self.PDE.pde = self.PDE.build_pde()
        if self.PDE.reference_fn is None:
            self.PDE.reference = self.PDE.load_reference(self.PDE.config.data_file)
            self.PDE.reference_fn = build_reference_function(
                self.PDE.reference[0], self.PDE.reference[1]
            )
        self.PDE.geom = self.PDE.make_domain(self.PDE.config)
        self.PDE.data = self.PDE.make_dataset_pinn_weighted_samples(self.PDE.geom, 
                                                       self.PDE.reference_fn, self.PDE.config)
        self.PDE.net = self.PDE.make_network(self.PDE.config)
        self.PDE.model = pinn_pro.ModelWeightedSamples(self.PDE.data, self.PDE.net, self)
        
        losshistory, train_state = self.optimize()
        self.save_artifacts(losshistory, train_state)
        self.evaluate(step=losshistory.steps[-1])
    
    def optimize(self):
        reweight_every = self.PDE.config.reweight_config.get("reweight_every", 1000)
        dde.optimizers.LBFGS_options["iter_per_step"] = reweight_every
        self.PDE.model.compile("adam", lr=self.PDE.config.learning_rate, loss_weights=self.PDE.config.loss_weights)
        losshistory, train_state = self.PDE.model.train(iterations=self.PDE.config.adam_iterations, 
                                                        display_every=500)
        

        if self.PDE.config.lbfgs_iterations > 0:
            self.PDE.model.compile("L-BFGS-B", loss_weights=self.PDE.config.loss_weights)
            losshistory, train_state = self.PDE.model.train(self.PDE.config.lbfgs_iterations, 
                                                            display_every=500)
        return losshistory, train_state
    
    def evaluate(self, step=None):
        coords, values_true = self.PDE.reference
        values_pred = self.PDE.model.predict(coords)
        metrics = self.PDE.component_metrics(values_true, values_pred)
        residual = self.PDE.model.predict(coords, operator=self.PDE.pde)
        metrics["pde_residual"] = float(np.mean(np.abs(residual)))
        save_metrics(self.run_dir / "metrics.json", metrics, step=step)

        pic_name = "results.png" if step is None else f"results_step_{step}.png"
        if hasattr(self.PDE, "make_grid"):
            grid_x, grid_t, grid_points = self.PDE.make_grid(self.PDE.config)
            grid_pred = self.PDE.model.predict(grid_points)
            if grid_pred.shape[1] == 1:
                grid_pred = grid_pred.reshape(grid_x.shape)
            else:
                grid_pred = grid_pred.reshape(grid_x.shape + (grid_pred.shape[1],))
            residual_grid = self.PDE.model.predict(grid_points, operator=self.PDE.pde)
            metrics["pde_residual_grid"] = float(np.mean(np.abs(residual_grid)))
            
            print_str = ""
            for key, value in metrics.items():
                print_str += f"{key}: {value:.4e} "
            print(print_str)
            self.PDE.plot_fields(coords, values_true, 
                             grid_x, grid_t, grid_pred, self.run_dir / pic_name, loss_record=print_str)
        
        else:
            print_str = ""
            for key, value in metrics.items():
                print_str += f"{key}: {value:.4e} "
            print(print_str)
            self.PDE.plot_fields(coords, values_true, values_pred,
                             self.run_dir / pic_name, loss_record=print_str)
    
    def plot_pde_loss_grad(self, pde_loss_weight, pde_loss_list, grad_norms, step):
        self.PDE.plot_pde_loss_grad(pde_loss_weight, pde_loss_list, grad_norms, 
                                     self.run_dir / f"pde_loss_grad_step_{step}.png")
        
    
    def save_artifacts(self, losshistory, train_state, step=None):
        plot_loss_curve(losshistory, self.PDE.config.loss_headers, self.run_dir / "loss_curve.png")
        self.PDE.model.save(self.run_dir)
        cfg_dict = asdict(self.PDE.config)
        cfg_dict.update(device_summary())
        save_loss_history(losshistory, self.run_dir / "loss.dat")
        save_config(self.run_dir / "config.json", cfg_dict)

def plot_loss_curve(loss_history, headers, save_path: Path):
    # loss_history.loss_train/test 是列表，每个元素为长度=len(headers) 的数组
    steps = np.asarray(loss_history.steps, dtype=float)
    train_arr = np.vstack(loss_history.loss_train) if loss_history.loss_train else None

    plt.figure(figsize=(4, 4))
    for idx, header in enumerate(headers):
        if train_arr is not None:
            plt.semilogy(steps, train_arr[:, idx], label=f"{header}")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()