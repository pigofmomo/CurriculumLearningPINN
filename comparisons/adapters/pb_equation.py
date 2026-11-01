import deepxde as dde
import torch


def pde(x, u, mu1, mu2, k, A):

    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    u_yy = dde.grad.hessian(u, x, i=1, j=1)

    def f(xy):
        x, y = xy[:, 0:1], xy[:, 1:2]
        return (
            A
            * (mu1**2 + x**2 + mu2**2 + y**2)
            * torch.sin(mu1 * torch.pi * x)
            * torch.sin(mu2 * torch.pi * y)
        )

    return -(u_xx + u_yy) + k**2 * u - f(x)


def output_transform(x, y):
    return (x[:, 0:1] ** 2 - 1) * (x[:, 1:2] ** 2 - 1) * y + 0.2
