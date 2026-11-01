"""Two-region Poisson-Boltzmann order diagnostic (10:1 followed by equal weights)."""
import os

os.environ["DDE_BACKEND"] = "pytorch"
import sys, time, json, csv, hashlib, ast, argparse, random
from pathlib import Path
import numpy as np
import torch

P = Path(__file__).resolve().parent
R = P.parent
import deepxde as dde

job = json.loads(Path(sys.argv[1]).read_text())
smoke = job["validation_only"]
args = argparse.Namespace(
    order=job["order"], seed=job["seed"], adam=5 if smoke else 1500, lbfgs=0
)
priority_end = 3 if smoke else 1000
device = torch.device(job.get("device", "cpu"))
O = Path(job["output"])
O.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from adapters.pb_equation import pde, output_transform

namespace = {"pde": pde, "output_transform": output_transform}
par = dict(mu1=2, mu2=4, k=5, A=5)
mu1, mu2, k, A = [par[n] for n in ["mu1", "mu2", "k", "A"]]
from adapters.pb_reference import generate

reference_path = O / "reference.npz"
generate(reference_path)
info = {
    "source_sha256": hashlib.sha256(
        (P / "adapters/pb_equation.py").read_bytes()
    ).hexdigest(),
    "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
}
net = dde.nn.FNN([2] + [30] * 4 + [1], "sin", "Glorot normal").to(device)
net.apply_output_transform(output_transform)
initial = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
torch.save(initial, O / "initial.pt")
hash_initial = hashlib.sha256(
    b"".join(v.numpy().tobytes() for v in initial.values())
).hexdigest()
xy = dde.geometry.Rectangle([-1, -1], [1, 1]).uniform_points(
    100 if smoke else 5000, boundary=False
)
a = np.sqrt(0.5)
inner = np.all(np.abs(xy) <= a, axis=1)
x = torch.tensor(xy, device=device, requires_grad=True)
mask = torch.tensor(inner, device=device)


def forcing(z):
    return (
        A
        * (mu1**2 + z[:, 0:1] ** 2 + mu2**2 + z[:, 1:2] ** 2)
        * torch.sin(mu1 * torch.pi * z[:, 0:1])
        * torch.sin(mu2 * torch.pi * z[:, 1:2])
    )


rhs = forcing(x).detach()


def residual(z, f):
    u = net(z)
    g = torch.autograd.grad(u, z, torch.ones_like(u), create_graph=True)[0]
    lap = sum(
        torch.autograd.grad(
            g[:, j : j + 1], z, torch.ones_like(g[:, j : j + 1]), create_graph=True
        )[0][:, j : j + 1]
        for j in range(2)
    )
    return -lap + k * k * u - f


# Assert residual implementation is identical to the source PDE at arbitrary inputs.
z = x[:37].detach().clone().requires_grad_(True)
rr = residual(z, forcing(z))
orig = namespace["pde"](z, net(z), mu1, mu2, k, A)
check = float((rr - orig).abs().max().detach())
assert check < 1e-5, check
dde.grad.clear()
with np.load(reference_path) as f:
    eval_xy = f["xy"].copy()
    truth = f["truth"].copy()
ev = torch.tensor(eval_xy, device=device, dtype=torch.float32)
im = np.all(np.abs(eval_xy) <= a, axis=1)
inside = np.all(np.abs(eval_xy) < 1, axis=1)
om = (~im) & inside
q = (np.arange(64) + 0.5) * 2 / 64 - 1
xx, yy = np.meshgrid(q, q)
res_xy = np.c_[xx.ravel(), yy.ravel()].astype("float32")
rx = torch.tensor(res_xy, device=device, requires_grad=True)
rf = forcing(rx).detach()
rm = np.all(np.abs(res_xy) <= a, axis=1)
t = np.linspace(-1, 1, 129, dtype="float32")
bc = np.concatenate(
    [np.c_[t, t * 0 - 1], np.c_[t, t * 0 + 1], np.c_[t * 0 - 1, t], np.c_[t * 0 + 1, t]]
)
b = torch.tensor(bc, device=device)
np.savez(O / "points.npz", pde=xy, inner_mask=inner, boundary=bc)
config = {
    **vars(args),
    "parameters": par,
    "seed": args.seed,
    "network": [2] + [30] * 4 + [1],
    "activation": "sin",
    "dtype": "float32",
    "hard_dirichlet": 0.2,
    "points": len(xy),
    "inner_points": int(inner.sum()),
    "outer_points": int((~inner).sum()),
    "inner_half_side": float(a),
    "warmup": 0,
    "equal_weight_from": priority_end,
    "priority_ratio": 10,
    "layer_coefficients": [20 / 11, 2 / 11],
    "loss_definition": ".5*(outer_coefficient*outer_residual_mse+inner_coefficient*inner_residual_mse)",
    "adam_lr": 0.001,
    "initial_sha256": hash_initial,
    "points_sha256": hashlib.sha256(xy.tobytes()).hexdigest(),
    "source_sha256": info["source_sha256"],
    "source_residual_difference": check,
    "reference_sha256": info["reference_sha256"],
    "GPU": str(device),
    "torch": torch.__version__,
    "bridge": False,
    "C2": False,
    "notes": "Same CL-PINN equation, hard boundary, activation, width/depth and uniform sampler; shorter diagnostic optimizer budget and spatial loss introduced. All layers remain active.",
}
(O / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
rows = []
start = time.perf_counter()
closure_calls = 0


def evaluate(stage, step, save=False):
    with torch.no_grad():
        pred = torch.cat([net(v) for v in ev.split(8192)]).cpu().numpy().ravel()
        be = (net(b) - 0.2).abs()
        bcmax = float(be.max())
    rr = residual(rx, rf).detach().cpu().numpy().ravel()
    rx.grad = None
    err = pred - truth
    row = {
        "stage": stage,
        "step": step,
        "seconds": time.perf_counter() - start,
        "relative_l2": float(np.linalg.norm(err) / np.linalg.norm(truth)),
        "outer_rmse": float(np.sqrt(np.mean(err[om] ** 2))),
        "inner_rmse": float(np.sqrt(np.mean(err[im] ** 2))),
        "pde_mse": float(np.mean(rr**2)),
        "outer_pde_mse": float(np.mean(rr[~rm] ** 2)),
        "inner_pde_mse": float(np.mean(rr[rm] ** 2)),
        "bc_max_abs": bcmax,
        "lbfgs_closure_calls": closure_calls,
    }
    assert np.isfinite(pred).all() and np.isfinite(rr).all()
    rows.append(row)
    with (O / "metrics.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerows(rows)
    if save:
        np.savez_compressed(
            O / f"field_{stage}_{step}.npz",
            xy=eval_xy,
            truth=truth,
            prediction=pred,
            error=err,
        )
        torch.save(net.state_dict(), O / f"model_{stage}_{step}.pt")
        torch.save(
            {"net": net.state_dict(), "optimizer": opt.state_dict()},
            O / f"checkpoint_{stage}_{step}.pt",
        )
    print(json.dumps(row), flush=True)
    return row


def loss(co=1, ci=1):
    rr = residual(x, rhs).ravel()
    return 0.5 * (co * rr[~mask].square().mean() + ci * rr[mask].square().mean())


opt = torch.optim.Adam(net.parameters(), lr=0.001)
for step in range(args.adam + 1):
    if step % 100 == 0 or step == args.adam or step == priority_end:
        evaluate("adam", step, step in [0, 500, priority_end, 1200, args.adam])
    if step == args.adam:
        break
    co, ci = 1.0, 1.0
    if step < priority_end:
        if args.order == "outer":
            co, ci = 20 / 11, 2 / 11
        elif args.order == "inner":
            co, ci = 2 / 11, 20 / 11
    opt.zero_grad(set_to_none=True)
    x.grad = None
    l = loss(co, ci)
    if not torch.isfinite(l):
        raise RuntimeError("Non-finite Adam loss")
    l.backward()
    opt.step()
adam_seconds = time.perf_counter() - start
torch.cuda.synchronize() if device.type == "cuda" else None
summary = {
    **rows[-1],
    "status": "complete",
    "adam_seconds": adam_seconds,
    "total_seconds": time.perf_counter() - start,
    "lbfgs_actual_iterations": 0,
    "lbfgs_stop": "not_used",
}
(O / "summary.json").write_text(json.dumps(summary, indent=2))
torch.save(
    {"net": net.state_dict(), "optimizer": opt.state_dict()}, O / "final_checkpoint.pt"
)
print("COMPLETE", json.dumps(summary), flush=True)
