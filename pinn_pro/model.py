import deepxde as dde
import torch
from collections import OrderedDict
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from deepxde import config
from deepxde import display
from deepxde import gradients as grad
from deepxde import losses as losses_module
from deepxde import metrics as metrics_module
from deepxde import optimizers
from deepxde.callbacks import CallbackList
from deepxde import utils
from deepxde import backend as bkd
import experiments.utils as exp_utils

class ModelWeightedSamples:
    def __init__(self, data, net, pinn_pro=None):
        self.data = data
        self.net = net

        self.opt_name = None
        self.loss_weights = None
        self.metrics = None
        self.train_state = TrainState()
        self.losshistory = LossHistory()
        self.stop_training = False
        self.opt = None
        
        self.outputs = None
        self.outputs_losses_train = None
        self.outputs_losses_test = None
        
        self.f_partition = None
        self.loss_fn = None
        self.pinn_pro = pinn_pro
        
        # Low-frequency bridge order. / 低频桥接项的阶数。
        self.low_fre_n = self.data.reweight_config.get("low_fre_n", 2)
        
        self.data_loss_weight = None
        
    @utils.timing
    def compile(
        self,
        optimizer,
        lr=None,
        loss="MSE",
        metrics=None,
        decay=None,
        loss_weights=None
    ):
        self.opt_name = optimizer
        loss_fn = losses_module.get(loss)
        self.loss_fn = loss_fn
        self.loss_weights = loss_weights
        self._compile_pytorch(lr, loss_fn, decay)
        metrics = metrics or []
        self.metrics = [metrics_module.get(m) for m in metrics]
        
    def _compile_pytorch(self, lr, loss_fn, decay):
        # Prediction path without loss computation. / 预测路径，不计算损失。
        def outputs(training, inputs):
            self.net.train(mode=training)
            with torch.no_grad():
                if isinstance(inputs, tuple):
                    inputs = tuple(
                        map(lambda x: torch.as_tensor(x).requires_grad_(), inputs)
                    )
                else:
                    inputs = torch.as_tensor(inputs)
                    inputs.requires_grad_()
            # Clear cached Jacobians and Hessians.
            grad.clear()
            return self.net(inputs)

        # Forward pass and loss computation. / 前向传播并计算损失。
        def outputs_losses(training, x_col_group, x_bc, x_data, y_data, losses_fn):
            self.net.train(mode=training)
            
            # Convert sampled arrays to tensors. / 将采样点转为张量。
            inputs_col_group = tuple(
                torch.as_tensor(x).requires_grad_() for x in x_col_group
            )
            inputs_bcs = tuple(torch.as_tensor(x).requires_grad_() for x in x_bc)
            
            outputs_col_group = tuple(self.net(inputs) for inputs in inputs_col_group)
            outputs_bcs = tuple(self.net(inputs) for inputs in inputs_bcs)
            
            if x_data is not None and y_data is not None:
                inputs_data = torch.as_tensor(x_data)
                outputs_data_true = torch.as_tensor(y_data)
                outputs_data = self.net(inputs_data)
                if self.data_loss_weight is None:
                    loss_data = loss_fn(outputs_data, outputs_data_true)
                else:
                    data_loss_weight_tensor = torch.as_tensor(self.data_loss_weight)
                    outputs_data = outputs_data * data_loss_weight_tensor
                    outputs_data_true = outputs_data_true * data_loss_weight_tensor
                    loss_data = loss_fn(outputs_data, outputs_data_true)
                loss_data = self.data_loss_weight * loss_data
                loss_data = loss_data.unsqueeze(0)
            else:
                loss_data = torch.tensor([0.0])
            
            # Delegate PDE residual losses to the data object. / PDE残差损失由data对象计算。
            losses, f_partition = losses_fn(inputs_col_group, outputs_col_group, 
                                            inputs_bcs, outputs_bcs, loss_fn, mul_pde_weights=training)
            self.f_partition = f_partition
            if not isinstance(losses, list):
                losses = [losses]
            losses = torch.stack(losses)
            # Weighted losses
            if self.loss_weights is not None:
                losses *= torch.as_tensor(self.loss_weights)
            
            losses = torch.cat((losses, loss_data))
            
            grad.clear()
            return losses
        

        def outputs_losses_train(x_col_group, x_bc, x_data, y_data):
            return outputs_losses(
                True, x_col_group, x_bc, x_data, y_data, self.data.losses
            )

        def outputs_losses_test(x_col_group, x_bc, x_data, y_data):
            return outputs_losses(
                False, x_col_group, x_bc, x_data, y_data, self.data.losses
            )

        trainable_variables = list(self.net.parameters())
        
        
        self.opt, self.lr_scheduler = optimizers.get(
            trainable_variables, self.opt_name, learning_rate=lr, decay=decay
        )

        def train_step(x_col_group, x_bc, x_data, y_data):
            def closure():
                losses = outputs_losses_train(x_col_group, x_bc, x_data, y_data)
                total_loss = torch.sum(losses)
                self.opt.zero_grad()
                total_loss.backward()
                return total_loss

            self.opt.step(closure)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # Callables
        self.outputs = outputs
        self.outputs_losses_train = outputs_losses_train
        self.outputs_losses_test = outputs_losses_test
        self.train_step = train_step
    
    # NumPy-facing helper interfaces. / 面向NumPy的辅助接口。
    def _outputs(self, training, inputs):
        outs = self.outputs(training, inputs)
        return utils.to_numpy(outs)
    
    def _outputs_losses(self, training, x_col_group, x_bc, x_data, y_data):
        if training:
            outputs_losses = self.outputs_losses_train
        else:
            outputs_losses = self.outputs_losses_test
        self.net.requires_grad_(requires_grad=False)
        outs = outputs_losses(x_col_group, x_bc, x_data, y_data)
        self.net.requires_grad_()
        
        loss_all_np = utils.to_numpy(outs)

        return loss_all_np
    
    def _train_step(self, x_col_group, x_bc, x_data, y_data):
        self.train_step(x_col_group, x_bc, x_data, y_data)
        
    @utils.timing
    def train(
        self,
        iterations=None,
        display_every=1000,
        model_restore_path=None,
        model_save_path=None
    ):
        if model_restore_path is not None:
            self.restore(model_restore_path, verbose=1)

        print("Training model...\n")
        self.stop_training = False
        if self.train_state.step == 0:
            self.data.init_data(savepath=self.pinn_pro.run_dir)
        
        # Training batch groups: collocation, boundary, and optional data anchors. / 训练批次包含配点、边界点和可选数据锚点。
        self.train_state.x_train_col = self.data.train_x_col
        self.train_state.x_train_group = self.data.train_x_group
        self.train_state.x_train_bc = self.data.train_x_bc
        self.train_state.x_train_data = self.data.train_x_data
        self.train_state.y_train_data = self.data.train_y_data
        self.train_state.x_test = self.data.test_x
        self.train_state.y_test = self.data.test_y
        
        self._test()
        
        if optimizers.is_external_optimizer(self.opt_name):
            self._train_pytorch_lbfgs(iterations)
        else:
            self._train_sgd(iterations, display_every)

        return self.losshistory, self.train_state

    def get_low_fre_anchor_values(self):
        # Interior anchors are stored as a list by subdomain. / 内部锚点按子域存为列表。
        x_anchors_np = self.data.geometry_partitions.inside_anchors
        y_anchors = [None for _ in range(len(x_anchors_np))]
        weights = [None for _ in range(len(x_anchors_np))]
        for i in range(len(x_anchors_np)):
            x_anchors_tenser = torch.as_tensor(x_anchors_np[i])
            y_anchors[i] = self.net(x_anchors_tenser).detach().cpu().numpy()
            weights[i] = self.data.pde_loss_weights[i]**2 * np.ones((y_anchors[i].shape[0], 1))
        
        x_anchors_all = np.vstack(x_anchors_np)
        y_anchors_all = np.vstack(y_anchors)
        weights_all = np.vstack(weights)
        coef, powers = exp_utils.weighted_polyfit_nd(x_anchors_all, y_anchors_all, deg=self.low_fre_n, w=weights_all)
        y_low = exp_utils.polyval_nd(x_anchors_all, coef, powers)

        exp_utils.visualize_polyfit_nd(
            x_anchors_all, y_anchors_all, coef, powers,
            out_path=self.pinn_pro.run_dir / f"anchors_polyfit_step_{self.train_state.step}.png"
        )
        
        self.train_state.x_train_data = x_anchors_all
        self.train_state.y_train_data = y_low
        self.train_state.weights_train_data = np.exp(-self.data.decay_epsi * weights_all)
        print("Low-fre anchors values updated.")
        pass
        
    def get_frame_anchor_values(self):
        # Frame points are stored as one NumPy array. / 框架点存为一个NumPy数组。
        x_anchors_np = self.data.geometry_partitions.frame_points
        x_anchors_tenser = torch.as_tensor(x_anchors_np)
        y_anchors = self.net(x_anchors_tenser).detach().cpu().numpy()
        self.train_state.x_train_data = x_anchors_np
        self.train_state.y_train_data = y_anchors
        self.train_state.weights_train_data = None
        print("Frame anchors values updated.")
    
    def reset_reweight_and_anchors(self):
        self.data.pde_loss_weights = [1 for _ in range(len(self.data.geometry_partitions.distance_to_boundary))]
        self.train_state.x_train_data = None
        self.train_state.y_train_data = None
        self.train_state.weights_train_data = None
        print("Reset PDE loss weights reset to 1 for each part and anchors reset to None.")
        
    def _train_sgd(self, iterations, display_every):
        reweight_every = self.data.reweight_config.get("reweight_every", 1000)
        reweight_causal_begin = self.data.reweight_config.get("reweight_causal_begin", 1000)
        reweight_causal_end = self.data.reweight_config.get("reweight_causal_end", 1000)
        reweight_adaptive_begin = self.data.reweight_config.get("reweight_adaptive_begin", 100000)
        reweight_adaptive_end = self.data.reweight_config.get("reweight_adaptive_end", 100000)
        
        for i in range(iterations):
            self._train_step(
                self.train_state.x_train_group,
                self.train_state.x_train_bc,
                self.train_state.x_train_data,
                self.train_state.y_train_data,
            )

            self.train_state.step += 1
            
            if self.train_state.step % display_every == 0 or i + 1 == iterations:
                self._test()
                
            if self.train_state.step % reweight_every == 0:
                self.record_and_plot()
                
                if self.train_state.step >= reweight_causal_begin and self.train_state.step < reweight_causal_end:
                    self.data_loss_weight = self.data.reweight_config.get("low_fre_data_weight", 0)
                    pde_loss_list = utils.to_numpy(self.pde_loss_for_each_part()[0])
                    self.data.pde_loss_weights = self.data.calculate_pde_weights_causal(pde_loss_list)
                    print(f"Step {self.train_state.step}: updated causal PDE loss weights for each part: {self.data.pde_loss_weights}")
                    self.get_low_fre_anchor_values()
                elif self.train_state.step >= reweight_causal_end and self.train_state.step < reweight_adaptive_begin:
                    self.reset_reweight_and_anchors()
                elif self.train_state.step >= reweight_adaptive_begin and self.train_state.step < reweight_adaptive_end:
                    self.data_loss_weight = self.data.reweight_config.get("frame_data_weight", 0)
                    pde_loss_for_each_part = self.pde_loss_for_each_part()
                    pde_loss_list = utils.to_numpy(pde_loss_for_each_part[0])
                    grad_norms = pde_loss_for_each_part[1]
                    self.data.pde_loss_weights = self.data.calculate_pde_weights_adaptive(pde_loss_list, log=self.data.reweight_config.get("log", False),
                                                                                      scale=self.data.reweight_config.get("scale", 10), 
                                                                                      grad_norms=grad_norms, grad_norms_scale=self.data.reweight_config.get("grad_norms_scale", 0))
                    print(f"Step {self.train_state.step}: updated adaptive PDE loss weights for each part: {self.data.pde_loss_weights}")
                    self.get_frame_anchor_values()
                elif self.train_state.step >= reweight_adaptive_end:
                    self.reset_reweight_and_anchors()
            
            if self.stop_training:
                break
    
    def _train_pytorch_lbfgs(self, iterations, display_every=None):
        reweight_every = self.data.reweight_config.get("reweight_every", 1000)
        reweight_causal_begin = self.data.reweight_config.get("reweight_causal_begin", 1000)
        reweight_causal_end = self.data.reweight_config.get("reweight_causal_end", 1000)
        reweight_adaptive_begin = self.data.reweight_config.get("reweight_adaptive_begin", 100000)
        reweight_adaptive_end = self.data.reweight_config.get("reweight_adaptive_end", 100000)
        
        prev_n_iter = 0
        
        while prev_n_iter < iterations:
            self._train_step(
                self.train_state.x_train_group,
                self.train_state.x_train_bc,
                self.train_state.x_train_data,
                self.train_state.y_train_data,
            )
            n_iter = self.opt.state_dict()["state"][0]["n_iter"]
            if prev_n_iter == n_iter - 1:
                break
            self.train_state.step += n_iter - prev_n_iter
            prev_n_iter = n_iter
            
            self._test()
            
            self.record_and_plot()
            
            if self.train_state.step >= reweight_causal_begin and self.train_state.step < reweight_causal_end:
                self.data_loss_weight = self.data.reweight_config.get("low_fre_data_weight", 0)
                pde_loss_list = utils.to_numpy(self.pde_loss_for_each_part()[0])
                self.data.pde_loss_weights = self.data.calculate_pde_weights_causal(pde_loss_list)
                print(f"Step {self.train_state.step}: updated causal PDE loss weights for each part: {self.data.pde_loss_weights}")
                self.get_low_fre_anchor_values()
            elif self.train_state.step >= reweight_causal_end and self.train_state.step < reweight_adaptive_begin:
                self.reset_reweight_and_anchors()
            elif self.train_state.step >= reweight_adaptive_begin and self.train_state.step < reweight_adaptive_end:
                self.data_loss_weight = self.data.reweight_config.get("frame_data_weight", 0)
                pde_loss_for_each_part = self.pde_loss_for_each_part()
                pde_loss_list = utils.to_numpy(pde_loss_for_each_part[0])
                grad_norms = pde_loss_for_each_part[1]
                self.data.pde_loss_weights = self.data.calculate_pde_weights_adaptive(pde_loss_list, log=self.data.reweight_config.get("log", False),
                                                                                      scale=self.data.reweight_config.get("scale", 10), 
                                                                                      grad_norms=grad_norms, grad_norms_scale=self.data.reweight_config.get("grad_norms_scale", 0))
                print(f"Step {self.train_state.step}: updated adaptive PDE loss weights for each part: {self.data.pde_loss_weights}")
                self.get_frame_anchor_values()
            elif self.train_state.step >= reweight_adaptive_end:
                self.reset_reweight_and_anchors()
                    
            if self.stop_training:
                break
            
    def _test(self):
        
        loss_all_np = \
            self._outputs_losses(
                False,
                self.train_state.x_train_group,
                self.train_state.x_train_bc,
                self.train_state.x_train_data,
                self.train_state.y_train_data,
            )

        self.train_state.loss_train = loss_all_np
        self.train_state.loss_test = loss_all_np
        
        self.losshistory.append(
            self.train_state.step,
            self.train_state.loss_train,
            self.train_state.loss_test,
            self.train_state.metrics_test,
        )

        if (
            np.isnan(self.train_state.loss_train).any()
            or np.isnan(self.train_state.loss_test).any()
        ):
            self.stop_training = True
        
        display.training_display(self.train_state)
        

    def predict(self, x, operator=None):
        if isinstance(x, tuple):
            x = tuple(np.asarray(xi, dtype=config.real(np)) for xi in x)
        else:
            x = np.asarray(x, dtype=config.real(np))
        
        if operator is None:
            y = self._outputs(False, x)
            return y
        
        self.net.eval()
        if isinstance(x, tuple):
            inputs = tuple(map(lambda x: torch.as_tensor(x).requires_grad_(), x))
        else:
            inputs = torch.as_tensor(x).requires_grad_()
        outputs = self.net(inputs)
        y = operator(inputs, outputs)
        grad.clear()
        y = utils.to_numpy(y)
        return y
    
    def state_dict(self):
        return self.net.state_dict()

    def save(self, save_path, protocol="backend", verbose=0):
        save_path = f"{save_path}-{self.train_state.step}"
        if protocol == "pickle":
            save_path += ".pkl"
            with open(save_path, "wb") as f:
                pickle.dump(self.state_dict(), f)
        elif protocol == "backend":
            save_path += ".pt"
            checkpoint = {
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
            }
            torch.save(checkpoint, save_path)
            
        print(
            "Epoch {}: saving model to {} ...\n".format(
                self.train_state.step, save_path
            )
        )
        return save_path

    def restore(self, save_path, device=None):
        print("Restoring model from {} ...\n".format(save_path))
        if device is not None:
            checkpoint = torch.load(save_path, map_location=torch.device(device))
        else:
            checkpoint = torch.load(save_path)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Reduce partitioned residuals to one PDE loss per subdomain. / 将分区残差汇总为每个子域的PDE损失。
    def pde_loss_for_each_part(self):
        f_partition_mse = [[self.loss_fn(bkd.zeros_like(error), error) for error in parts] for parts in self.f_partition]
        pde_loss_list = [sum(part) for part in f_partition_mse]
        grads_norms = loss_magnitude(pde_loss_list, self.net)
        return pde_loss_list, grads_norms
    
    def record_and_plot(self):
        self.pinn_pro.evaluate(step=self.train_state.step)
        
        losses = self.outputs_losses_train(
            self.train_state.x_train_group, self.train_state.x_train_bc, self.train_state.x_train_data, self.train_state.y_train_data)
        
        losses_for_each_term = loss_magnitude(losses, self.net)
        print(f"Step {self.train_state.step}: losses grad for each term: {losses_for_each_term}")
        
        pde_loss_list, grads_norms = self.pde_loss_for_each_part()
        pde_loss_weight = self.data.pde_loss_weights
        self.pinn_pro.plot_pde_loss_grad(pde_loss_weight, utils.to_numpy(pde_loss_list), grads_norms, self.train_state.step)
        
        return pde_loss_list, grads_norms

def loss_magnitude(losses, net):
    grad_norms = []
    for li in losses:  # li 是标量 loss
        g = torch.autograd.grad(
            li,
            net.parameters(),
            retain_graph=True,    # 逐项求梯度需要保留计算图
            allow_unused=True,
        )
        # L2 范数聚合各参数梯度
        total = torch.sqrt(sum((gi.norm() ** 2 for gi in g if gi is not None)))
        grad_norms.append(total.item())
    
    return grad_norms 
        
class TrainState:
    def __init__(self):
        self.step = 0

        self.x_train_col = None
        self.x_train_group = None
        self.x_train_bc = None
        self.x_train_data = None
        self.y_train_data = None
        self.x_test = None
        self.y_test = None

        self.weights_train_data = None
        
        self.loss_train = []
        self.loss_test = []
        self.metrics_test = []

class LossHistory:
    def __init__(self):
        self.steps = []
        self.loss_train = []
        self.loss_test = []
        self.metrics_test = []

    def append(self, step, loss_train, loss_test, metrics_test):
        self.steps.append(step)
        self.loss_train.append(loss_train)
        if loss_test is None:
            loss_test = self.loss_test[-1]
        if metrics_test is None:
            metrics_test = self.metrics_test[-1]
        self.loss_test.append(loss_test)
        self.metrics_test.append(metrics_test)

