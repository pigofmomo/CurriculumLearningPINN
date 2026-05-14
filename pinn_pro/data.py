import numpy as np
import deepxde as dde
from deepxde import backend as bkd
from deepxde import config
from domain_decomp import Pointset1D, Pointset2D
import torch

class PDEWeightedSamples:
    def __init__(
        self,
        geometry,
        pde,
        bcs,
        num_domain=0,
        num_boundary=0,
        train_distribution="uniform",
        solution=None,
        num_test=None,
        reweight_config=None,
    ):
        
        self.decay_epsi = float(reweight_config["decay_epsi"]) if reweight_config and "decay_epsi" in reweight_config else 0.0
        self.num_subdomains = reweight_config["num_subdomains"] if reweight_config and "num_subdomains" in reweight_config else None
        self.reweight_config = reweight_config
        
        self.geometry_partitions = None
        self.pde_loss_list = None
        self.pde_loss_weights = None
        self.pde_loss_weights_vec = None
        
        self.geom = geometry
        self.pde = pde
        self.bcs = bcs if isinstance(bcs, (list, tuple)) else [bcs]

        self.num_domain = num_domain
        self.num_boundary = num_boundary
        self.train_distribution = train_distribution
        
        self.soln = solution
        self.num_test = num_test

        # 3类采样点
        self.num_bcs = None
        self.train_x_group = None
        self.train_x_col = None
        self.train_x_bc = None
        self.train_x_data = None
        self.train_y_data = None
        
        self.train_x_frame = None


        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None

    def init_data(self, savepath=None):
        self.set_geometry_partitions(self.num_subdomains, savepath=savepath)
        self.train_next_batch()
        self.geometry_partitions.split(savepath=savepath, col_points=self.train_x_group)
        self.test()
        
        
    def set_geometry_partitions(self, num_subdomains, savepath=None):
        if self.geom.dim == 1:
            pointset = Pointset1D(self.geom, num_subdomains)
            pointset.split()
            pointset.gen_inside_anchors()
            pointset.gen_frame_points(num_each_part=2, savepath=savepath)
            self.geometry_partitions = pointset
        elif self.geom.dim == 2:
            pointset = Pointset2D(self.geom, num_subdomains)
            pointset.split()
            pointset.gen_inside_anchors()
            pointset.gen_frame_points(num_each_part=16, savepath=savepath)
            self.geometry_partitions = pointset
        
        self.pde_loss_weights = [1 for _ in range(len(pointset.distance_to_boundary))]
    
    def calculate_pde_weights_causal(self, pde_loss_list):
        distance_to_boundary = np.asarray(self.geometry_partitions.distance_to_boundary)
        pde_loss_arr = np.asarray(pde_loss_list)

        # 归一化 loss，避免尺度过大/过小影响权重
        sum_loss = np.sum(pde_loss_arr)
        if sum_loss > 0:
            pde_loss_arr = pde_loss_arr / sum_loss

        unique_dist = np.sort(np.unique(distance_to_boundary))
        group_sum = {
            d: float(np.sum(pde_loss_arr[distance_to_boundary == d])) for d in unique_dist
        }
        cumulative_sum = {-1: 0.0}
        running = 0.0
        for d in unique_dist:
            running += group_sum[d]
            cumulative_sum[d] = running

        epsi = self.decay_epsi
        weights = [np.exp(-epsi * cumulative_sum[d-1]) for d in distance_to_boundary]
        self.pde_loss_weights = weights
        
        return weights
    
    def calculate_pde_weights_adaptive(self, pde_loss_list, log=False, scale=10, grad_norms=None, grad_norms_scale=0):
        
        if grad_norms is not None and log is True:
            grad_norms_arr = np.asarray(grad_norms)
            grad_norms_arr = np.log(grad_norms_arr + 1e-12)
            grad_norms_arr = grad_norms_arr * grad_norms_scale
        else:
            grad_norms_arr = np.zeros_like(pde_loss_list, dtype=float)
        
        # 按区域 loss 大小分配权重；
        # 若 log=True，则取log
        # 归一到1~scale的绝对值之间
        # 如果scale<0，则反转权重，loss大的权重小，loss小的权重大
        scale_abs = abs(scale)
        if scale < 0:
            reverse = True
        else:            
            reverse = False
        
        pde_loss_arr = np.asarray(pde_loss_list, dtype=float)
        if log:
            pde_loss_arr = np.log(pde_loss_arr + 1e-12)

        loss_arr = pde_loss_arr - grad_norms_arr
        
        min_v, max_v = float(np.min(loss_arr)), float(np.max(loss_arr))
        if max_v - min_v < 1e-12:
            weights = np.ones_like(loss_arr)
        else:
            norm = (loss_arr - min_v) / (max_v - min_v)  # [0,1]
            if reverse:
                norm = 1.0 - norm
            weights = 1.0 + norm * (scale_abs - 1.0)  # [1, scale_abs]
        self.pde_loss_weights = weights.tolist()
        return self.pde_loss_weights
        
    def losses(self, inputs_col_group, outputs_col_group, inputs_bcs, outputs_bcs, loss_fn, 
               mul_pde_weights=True):
        
        f_bc = []
        for i in range(len(inputs_bcs)):
            pde_loss_bc = self.pde(inputs_bcs[i], outputs_bcs[i])
            if not isinstance(pde_loss_bc, (list, tuple)):
                pde_loss_bc = [pde_loss_bc]
            f_bc.append(pde_loss_bc)
        f_bc_stack = [torch.vstack([f_bc[i][j] for i in range(len(f_bc))]) for j in range(len(f_bc[0]))]
        # 是一个tensor，补充到每一个pde term的前面就行
        
        # print(f_bc_stack)
        num_partitions = len(inputs_col_group)
        f_partition = [None for i in range(num_partitions)]
        for i in range(num_partitions):
            f_i = self.pde(inputs_col_group[i], outputs_col_group[i]) # 是长度等于点数的tensor
            if not isinstance(f_i, (list, tuple)):
                f_i = [f_i]
            if mul_pde_weights:
                f_partition[i] = [fi * self.pde_loss_weights[i] for fi in f_i]
            else:
                f_partition[i] = f_i
            
        num_pde_terms = len(f_partition[0])
        f = [
            torch.vstack([f_partition[i][j] for i in range(num_partitions)])
            for j in range(num_pde_terms)
        ]
        # print(f[0][0:5,:])
        f = [torch.vstack((f_bc_stack[j], f[j])) for j in range(num_pde_terms)]
        # print(f[0][0:5,:])
        losses = [
            loss_fn(bkd.zeros_like(error), error) for error in f
        ]
        # print(losses)
        for i, bc in enumerate(self.bcs):
            beg, end = 0, self.num_bcs[i]
            error = bc.error(self.train_x_bc[i], inputs_bcs[i], outputs_bcs[i], beg, end)
            losses.append(loss_fn(bkd.zeros_like(error), error))
        
        # print(losses)
        return losses, f_partition

    def train_next_batch(self):
        self.train_points()
        self.bc_points()
        self.train_x = np.vstack((np.vstack(self.train_x_bc), self.train_x_col))
        self.train_y = self.soln(self.train_x) if self.soln else None
        pass

    def train_points(self):
        # 计算出col points
        X = np.empty((0, self.geom.dim), dtype=config.real(np))
        if self.num_domain > 0:
            if self.train_distribution == "uniform":
                X = self.geom.uniform_points(self.num_domain, boundary=False)
            else:
                X = self.geom.random_points(
                    self.num_domain, random=self.train_distribution
                )
        if self.num_boundary > 0:
            if self.train_distribution == "uniform":
                tmp = self.geom.uniform_boundary_points(self.num_boundary)
            else:
                tmp = self.geom.random_boundary_points(
                    self.num_boundary, random=self.train_distribution
                )
            X = np.vstack((tmp, X))
        
        X = X.astype(config.real(np))
        self.train_x_col = X
        self.train_x_group = self.geometry_partitions.filter_points(X)
        
        pass
        

    def bc_points(self):
        x_bcs = [bc.collocation_points(self.train_x_col) for bc in self.bcs]
        self.num_bcs = list(map(len, x_bcs))
        self.train_x_bc = (
            [np.asarray(x_bc, dtype=config.real(np)) for x_bc in x_bcs]
            if x_bcs
            else np.empty([0, self.train_x_col.shape[-1]], dtype=config.real(np))
        )
        
    def test(self):
        if self.num_test is None:
            self.test_x = self.train_x
        else:
            x = self.geom.uniform_points(self.num_test, boundary=False)
            train_x_bc_stacked = np.vstack(self.train_x_bc)
            self.test_x = np.vstack((train_x_bc_stacked, x))
        self.test_y = self.soln(self.test_x) if self.soln else None