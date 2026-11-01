"""Three square-ring subnetworks, preserving frozen PDEs and XPINN interface terms."""
import torch
from torch import nn


def install(base):
    class RingNet(nn.Module):
        def __init__(self, width, depth, outputs):
            super().__init__()
            self.weights = nn.ParameterList()
            self.biases = nn.ParameterList()
            self.scales = nn.ParameterList()
            sizes = [2] + [width] * depth + [outputs]
            for k, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
                v = torch.empty(3, a, b)
                for i in range(3):
                    nn.init.xavier_uniform_(v[i])
                self.weights.append(nn.Parameter(v))
                self.biases.append(nn.Parameter(torch.zeros(3, 1, b)))
                if k < depth:
                    self.scales.append(nn.Parameter(torch.full((3, 1, 1), 0.05)))
            self.register_buffer("halfwidth", torch.tensor([0.5, 0.3, 0.1]))

        def forward(self, x, ids):
            h = (x - 0.5) / self.halfwidth[ids, None, None]
            for k, (w, b) in enumerate(zip(self.weights, self.biases)):
                h = torch.bmm(h, w[ids]) + b[ids]
                if k < len(self.scales):
                    h = 20 * self.scales[k][ids] * h
                    a = (ids % 3)[:, None, None]
                    h = torch.where(
                        a == 0,
                        torch.tanh(h),
                        torch.where(a == 1, torch.sin(h), torch.cos(h)),
                    )
            return h

    def labels(x):
        d = torch.minimum(x, 1 - x).amin(-1)
        return (d >= 0.2).long() + (d >= 0.4).long()

    def contour(a, n, device):
        t = (
            a
            + (1 - 2 * a)
            * (torch.arange(n, device=device, dtype=torch.float32) + 0.5)
            / n
        )
        return torch.cat(
            [
                torch.stack([t, torch.full_like(t, a)], -1),
                torch.stack([t, torch.full_like(t, 1 - a)], -1),
                torch.stack([torch.full_like(t, a), t], -1),
                torch.stack([torch.full_like(t, 1 - a), t], -1),
            ]
        )[None]

    def prepare(cfg, case, device, smoke=False):
        side = 10 if smoke else round(cfg["num_domain"] ** 0.5)
        axis = (torch.arange(side, device=device, dtype=torch.float32) + 0.5) / side
        xx, yy = torch.meshgrid(axis, axis, indexing="ij")
        x = torch.stack([xx.flatten(), yy.flatten()], -1)
        ids = labels(x)
        pts = [x[ids == k][None].detach().requires_grad_() for k in range(3)]
        d = dict(
            inside=pts,
            ids=[torch.tensor([k], device=device) for k in range(3)],
            interface=[
                contour(0.2, 3 if smoke else 150, device).requires_grad_(),
                contour(0.4, 3 if smoke else 50, device).requires_grad_(),
            ],
            boundary=contour(0, 4 if smoke else cfg["num_boundary"] // 4, device),
            bids=torch.tensor([0], device=device),
            pressure=torch.zeros(1, 1, 2, device=device),
            pid=torch.tensor([0], device=device),
        )
        d["top"] = d["boundary"][..., 1] == 1
        if case != "ns":
            fn = (
                base.equations.poisson_forcing
                if case == "poisson"
                else base.equations.adr_forcing
            )
            exact = (
                base.equations.poisson_exact
                if case == "poisson"
                else base.equations.adr_exact
            )
            for key in ["inside", "interface"]:
                d[key + "_forcing"] = [
                    fn(z.detach().reshape(-1, 2)).reshape(1, -1, 1).detach()
                    for z in d[key]
                ]
            d["boundary_target"] = (
                exact(d["boundary"].reshape(-1, 2)).reshape(1, -1, 1).detach()
            )
        return d

    def losses(net, d, cfg, case, flux_weight=0.0):
        total = sum(z.shape[1] for z in d["inside"])
        pde = 0
        for k, x in enumerate(d["inside"]):
            _, res = base.field_residual(
                net, x, d["ids"][k], case, d.get("inside_forcing", [None] * 3)[k]
            )
            pde = pde + res.square().mean((0, 1)).sum() * x.shape[1] / total
        b = net(d["boundary"], d["bids"])
        weights = cfg["loss_weights"]
        if case == "ns":
            target = (
                cfg["lid_coefficient"]
                * d["boundary"][..., 0]
                * (1 - d["boundary"][..., 0])
            )
            top = d["top"]
            bc = (
                weights[3] * (b[..., 0][top] - target[top]).square().mean()
                + weights[4] * b[..., 1][top].square().mean()
                + weights[5] * b[..., 0][~top].square().mean()
                + weights[6] * b[..., 1][~top].square().mean()
                + weights[7] * net(d["pressure"], d["pid"])[..., 2].square().mean()
            )
        else:
            bc = weights[1] * (b - d["boundary_target"]).square().mean()
        ui = 0
        ri = 0
        for k, x in enumerate(d["interface"]):
            force = d.get("interface_forcing", [None] * 2)[k]
            a, ra = base.field_residual(net, x, d["ids"][k], case, force)
            b, rb = base.field_residual(net, x, d["ids"][k + 1], case, force)
            ui = ui + 10 * (a - b).square().mean()
            ri = ri + 2 * (ra - rb).square().mean((0, 1)).sum()
        return dict(pde=pde, boundary=bc, interface_value=ui, interface_residual=ri)

    def predict(net, x):
        ids = labels(x)
        out = []
        # Evaluate each network only on its own points; restore original ordering.
        result = torch.zeros(
            (len(x), net.biases[-1].shape[-1]), device=x.device, dtype=x.dtype
        )
        for k in range(3):
            ix = torch.where(ids == k)[0]
            if len(ix):
                result[ix] = net(x[ix][None], torch.tensor([k], device=x.device))[0]
        return result

    def jumps(net, d, case):
        values = []
        res = []
        for k, x in enumerate(d["interface"]):
            force = d.get("interface_forcing", [None] * 2)[k]
            a, ra = base.field_residual(net, x, d["ids"][k], case, force)
            b, rb = base.field_residual(net, x, d["ids"][k + 1], case, force)
            values.append((a - b).reshape(-1))
            res.append((ra - rb).reshape(-1))
        v = torch.cat(values)
        r = torch.cat(res)
        return dict(
            value_rms=float(v.square().mean().sqrt().detach()),
            value_max=float(v.abs().max().detach()),
            residual_rms=float(r.square().mean().sqrt().detach()),
        )

    base.LocalEnsemble = RingNet
    base.prepare_data = prepare
    base.loss_terms = losses
    base.pointwise_prediction = predict
    base.ring_jumps = jumps
    base.ring_labels = labels
