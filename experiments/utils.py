"""Shared experiment utilities for metrics, references, plots, and summaries. / 实验指标、参考解、绘图与汇总工具。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Union
import deepxde as dde
import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
from statistics import mean, pvariance
import re


def device_summary() -> Dict[str, object]:
	if not torch.cuda.is_available():
		return {"cuda_available": False}
	props = torch.cuda.get_device_properties(torch.cuda.current_device())
	return {
		"cuda_available": True,
		"cuda_device_name": props.name,
		"cuda_device_memory_gb": round(props.total_memory / (1024 ** 3), 2),
	}


def save_loss_history(loss_history, path: Path) -> None:
	print(f"Saving loss history to {path} ...")
	num_rows = len(loss_history.steps)
	data = np.hstack(
		(
			np.array(loss_history.steps, dtype=float)[:, None],
			_pad_records(loss_history.loss_train, num_rows),
			_pad_records(loss_history.loss_test, num_rows),
			_pad_records(loss_history.metrics_test, num_rows),
		)
	)
	np.savetxt(path, data, header="step, loss_train, loss_test, metrics_test")


def save_config(path: Path, config_dict: Dict[str, Any]) -> None:
	def _default(obj):
		if isinstance(obj, np.generic):
			return obj.item()
		return str(obj)

	with path.open("w", encoding="utf-8") as fh:
		json.dump(config_dict, fh, indent=2, sort_keys=True, default=_default)


def save_metrics(path: Path, metrics: Dict[str, float], step=None) -> None:
	# 若提供 step 且文件已存在，则追加 {step: metrics}；否则直接写入 metrics。
	if step is None:
		with path.open("w", encoding="utf-8") as fh:
			json.dump(metrics, fh, indent=2, sort_keys=True)
		return

	data: Dict[str, Any] = {}
	if path.exists():
		try:
			with path.open("r", encoding="utf-8") as fh:
				loaded = json.load(fh)
			if isinstance(loaded, dict):
				data.update(loaded)
		except Exception:
			# 若旧文件损坏或格式不符，直接覆盖
			data = {}

	data[str(step)] = metrics
	with path.open("w", encoding="utf-8") as fh:
		json.dump(data, fh, indent=2, sort_keys=True)


def max_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	return float(np.max(np.abs(y_true - y_pred)))


def _pad_records(records: Iterable, num_rows: int):
	if records is None or len(records) == 0:
		return np.empty((num_rows, 0))

	arrays = [np.atleast_1d(np.asarray(rec, dtype=float).reshape(-1)) for rec in records]
	width = max(arr.size for arr in arrays)
	padded = np.full((len(arrays), width), np.nan)
	for idx, arr in enumerate(arrays):
		padded[idx, : arr.size] = arr
	return padded

def ensure_run_root(RUN_ROOT: Path) -> None:
	if RUN_ROOT.exists():
		shutil.rmtree(RUN_ROOT)
	RUN_ROOT.mkdir(parents=True, exist_ok=True)
    
    
def enforce_dataset_dtype(data, dtype):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.astype(dtype, copy=False)
        if isinstance(value, (list, tuple)):
            converted = [convert(item) for item in value]
            return type(value)(converted)
        return value

    for attr in (
        "train_x",
        "train_y",
        "train_aux_vars",
        "test_x",
        "test_y",
        "test_aux_vars",
    ):
        if hasattr(data, attr):
            current = getattr(data, attr)
            if current is not None:
                setattr(data, attr, convert(current))

def build_reference_function(coords, values, dtype=dde.config.real(np)):
	if values.shape[1] == 1:
		interpolator = LinearNDInterpolator(coords, values.ravel())

		def reference_fn(points: np.ndarray):
			vals = interpolator(points)
			if vals is None:
				vals = np.zeros(len(points), dtype=dtype)
			vals = np.nan_to_num(vals, nan=0.0)
			return vals.reshape(-1, 1).astype(dtype, copy=False)

	else:
		interpolators = [LinearNDInterpolator(coords, values[:, idx]) for idx in range(values.shape[1])]

		def reference_fn(points: np.ndarray):
			comps = []
			for interp in interpolators:
				vals = interp(points)
				if vals is None:
					vals = np.zeros(len(points), dtype=dtype)
				vals = np.nan_to_num(vals, nan=0.0)
				comps.append(vals.reshape(-1, 1))
			stacked = np.hstack(comps)
			return stacked.astype(dtype, copy=False)

	return reference_fn

def weighted_polyfit_nd(X: np.ndarray, y: np.ndarray, deg: int, w: np.ndarray | None = None):
    """
    Weighted polynomial regression for d-dimensional inputs.
    Fits y ≈ A(X) @ coef where A is monomial basis (total degree <= deg).

    Parameters
    ----------
    X : ndarray, shape (N, d)
        Input coordinates.
    y : ndarray, shape (N,), (N,1) or (N,C)
        Target values (C outputs supported).
    deg : int
        Total polynomial degree.
    w : ndarray, shape (N,) or (N,1), optional
        Non-negative weights.

    Returns
    -------
    coef : ndarray, shape (M, C)
        Coefficients for each output channel. If C=1, still returns (M,1).
    powers : ndarray, shape (M, d)
        Exponent vectors for each monomial term.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N,d), got shape {X.shape}")
    N, d = X.shape

    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        if y.shape[0] != N:
            raise ValueError(f"y length must match N={N}, got {y.shape[0]}")
        y = y.reshape(N, 1)  # (N,1)
    elif y.ndim == 2:
        if y.shape[0] != N:
            raise ValueError(f"y first dim must match N={N}, got {y.shape[0]}")
        # keep as (N,C)
    else:
        raise ValueError(f"y must be 1D or 2D, got shape {y.shape}")

    if w is None:
        w = np.ones((N,), dtype=float)
    else:
        w = np.asarray(w, dtype=float).reshape(-1)
        if w.shape[0] != N:
            raise ValueError(f"w must have length N={N}, got {w.shape[0]}")
        if np.any(w < 0):
            raise ValueError("w must be non-negative")

    if deg < 0:
        raise ValueError("deg must be >= 0")

    # Build monomial exponent list for total degree <= deg
    if d == 1:
        powers = np.arange(deg + 1).reshape(-1, 1)
    elif d == 2:
        pows = []
        for i in range(deg + 1):
            for j in range(deg + 1 - i):
                pows.append((i, j))
        powers = np.array(pows, dtype=int)
    else:
        # Generic recursion for d>2 (term count grows quickly)
        def gen_powers(total_deg, dim):
            if dim == 1:
                return [(total_deg,)]
            out = []
            for a0 in range(total_deg + 1):
                for rest in gen_powers(total_deg - a0, dim - 1):
                    out.append((a0,) + rest)
            return out

        pows = []
        for tdeg in range(deg + 1):
            pows.extend(gen_powers(tdeg, d))
        powers = np.array(pows, dtype=int)

    # Design matrix A: A[i, j] = prod_k X[i,k]**powers[j,k]
    M = powers.shape[0]
    A = np.ones((N, M), dtype=float)
    for k in range(d):
        A *= X[:, [k]] ** powers[:, k]

    # Weighted least squares: min ||sqrt(W)(A C - Y)||_F
    Wsqrt = np.sqrt(w)                      # (N,)
    Aw = A * Wsqrt[:, None]                 # (N,M)
    Yw = y * Wsqrt[:, None]                 # (N,C)

    coef, *_ = np.linalg.lstsq(Aw, Yw, rcond=None)  # coef: (M,C)
    return coef, powers

def polyval_nd(X: np.ndarray, coef: np.ndarray, powers: np.ndarray):
    """
    Evaluate the polynomial defined by (coef, powers) at X.

    Parameters
    ----------
    X : ndarray, shape (N, d)
    coef : ndarray, shape (M,) or (M,1) or (M,C)
        Polynomial coefficients. If (M,C), returns C outputs.
    powers : ndarray, shape (M, d)
        Exponent vectors for each monomial term.

    Returns
    -------
    yhat : ndarray
        If coef is (M,), returns shape (N,)
        If coef is (M,1), returns shape (N,1)
        If coef is (M,C), returns shape (N,C)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N,d), got shape {X.shape}")
    N, d = X.shape

    powers = np.asarray(powers, dtype=int)
    if powers.ndim != 2:
        raise ValueError(f"powers must be 2D (M,d), got shape {powers.shape}")
    if powers.shape[1] != d:
        raise ValueError(f"powers expects d={powers.shape[1]}, but X has d={d}")

    coef = np.asarray(coef, dtype=float)
    if coef.ndim == 1:
        # (M,)
        coef2 = coef.reshape(-1, 1)   # (M,1)
        squeeze_1d = True
    elif coef.ndim == 2:
        # (M,C)
        coef2 = coef
        squeeze_1d = False
    else:
        raise ValueError(f"coef must be 1D or 2D, got shape {coef.shape}")

    M = powers.shape[0]
    if coef2.shape[0] != M:
        raise ValueError(f"coef first dim must match M={M}, got {coef2.shape[0]}")

    # Design matrix A: (N,M)
    A = np.ones((N, M), dtype=float)
    for k in range(d):
        A *= X[:, [k]] ** powers[:, k]

    Yhat = A @ coef2  # (N,C)

    if squeeze_1d:
        return Yhat.reshape(-1)  # (N,)
    return Yhat

def visualize_polyfit_nd(
    X,              # (N, d)
    y,              # (N,) or (N,1) or (N,C)
    coef,           # (M,) or (M,1) or (M,C)
    powers,         # (M, d)
    out_path,       # Path or str
    title="Low-frequency fit on anchors",
    grid_res=50     # for 2D visualization
):
    """
    Visualize polynomial fit for 1D or 2D.
    If y is multi-output (N,C), plot C subplots (one per channel).

    Parameters
    ----------
    X : (N,d) array
    y : (N,) or (N,1) or (N,C)
    coef : (M,) or (M,1) or (M,C)
    powers : (M,d)
    out_path : str or Path
    title : str
    grid_res : int
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N,d), got {X.shape}")
    N, d = X.shape

    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        if y.shape[0] != N:
            raise ValueError(f"y length must match N={N}, got {y.shape[0]}")
        Y = y.reshape(N, 1)  # (N,1)
    elif y.ndim == 2:
        if y.shape[0] != N:
            raise ValueError(f"y first dim must match N={N}, got {y.shape[0]}")
        Y = y               # (N,C)
    else:
        raise ValueError(f"y must be 1D or 2D, got {y.shape}")

    # Evaluate polyfit on anchors (same X)
    Ylow = polyval_nd(X, coef, powers)
    if np.ndim(Ylow) == 1:
        Ylow = Ylow.reshape(N, 1)
    elif Ylow.ndim == 2:
        if Ylow.shape[0] != N:
            raise ValueError(f"polyval result N mismatch: {Ylow.shape[0]} vs {N}")
    else:
        raise ValueError(f"polyval_nd returned unexpected shape {Ylow.shape}")

    C = Y.shape[1]
    if Ylow.shape[1] != C:
        raise ValueError(f"Channel mismatch: y has C={C}, but polyfit gives {Ylow.shape[1]}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Layout: one row if few channels, otherwise grid
    ncols = min(3, C)
    nrows = int(np.ceil(C / ncols))

    # ---------- 1D ----------
    if d == 1:
        x = X[:, 0]
        order = np.argsort(x)

        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), squeeze=False)
        for c in range(C):
            ax = axes[c // ncols][c % ncols]
            ax.plot(x[order], Ylow[order, c], linewidth=2, label="polyfit")
            ax.scatter(x, Y[:, c], s=15, alpha=0.7, label="anchors")
            ax.set_xlabel("x")
            ax.set_ylabel(f"y[ch={c}]")
            ax.set_title(f"{title} (ch={c})")
            ax.legend()

        # hide unused subplots
        for k in range(C, nrows*ncols):
            axes[k // ncols][k % ncols].axis("off")

        # plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)

    # ---------- 2D ----------
    elif d == 2:
        x = X[:, 0]
        y_coord = X[:, 1]

        # grid for contour
        xg = np.linspace(x.min(), x.max(), grid_res)
        yg = np.linspace(y_coord.min(), y_coord.max(), grid_res)
        Xg, Yg = np.meshgrid(xg, yg)
        grid = np.stack([Xg.ravel(), Yg.ravel()], axis=1)

        Zg_all = polyval_nd(grid, coef, powers)
        if np.ndim(Zg_all) == 1:
            Zg_all = Zg_all.reshape(-1, 1)

        if Zg_all.shape[1] != C:
            raise ValueError(f"Grid poly channel mismatch: {Zg_all.shape[1]} vs C={C}")

        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows), squeeze=False)
        for c in range(C):
            ax = axes[c // ncols][c % ncols]

            Zg = Zg_all[:, c].reshape(grid_res, grid_res)
            contour = ax.contourf(Xg, Yg, Zg, levels=30, cmap="viridis", alpha=0.85)
            fig.colorbar(contour, ax=ax, shrink=0.85)

            sc = ax.scatter(
                x, y_coord,
                c=Y[:, c],
                cmap="viridis",
                edgecolor="k",
                s=25,
                label="anchors"
            )

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"{title} (ch={c})")
            ax.legend(loc="upper right")

        # hide unused subplots
        for k in range(C, nrows*ncols):
            axes[k // ncols][k % ncols].axis("off")

        # plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)

    else:
        raise ValueError(f"Visualization only supports d=1 or d=2, got d={d}")
    
# Summarize final metrics by experiment group. The table reports mean and variance.
def summarize_batch(dir: Union[str, Path], output_filename: str = "final_metrics_summary.txt") -> Path:
    batch_dir = Path(dir)
    if not batch_dir.exists() or not batch_dir.is_dir():
        raise ValueError(f"Invalid batch directory: {batch_dir}")

    metrics_paths = sorted(batch_dir.rglob("metrics.json"))
    if not metrics_paths:
        raise ValueError(f"No metrics.json found under: {batch_dir}")

    def canonical_group_key(path: Path) -> str:
        # Group only by experiment path pattern, independent of each run's max_step.
        rel = path.relative_to(batch_dir).as_posix()
        rel = re.sub(r"^run_\d+_seed_\d+/", "", rel)
        # Normalize both naming styles: `random_seed_x` and `seed_x`.
        rel = re.sub(r"random_seed_\d+|seed_\d+", "seed_*", rel)
        return rel

    groups: Dict[str, list] = {}
    for metrics_path in metrics_paths:
        with metrics_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        steps = [int(k) for k in data.keys() if str(k).isdigit()]
        if not steps:
            continue

        max_step = max(steps)
        final_metrics = data[str(max_step)]

        group_key = canonical_group_key(metrics_path)
        groups.setdefault(group_key, []).append(
            {
                "file": metrics_path,
                "max_step": max_step,
                "metrics": final_metrics,
            }
        )

    if not groups:
        raise ValueError(f"No valid numeric-step metrics found under: {batch_dir}")

    out_path = batch_dir / output_filename
    lines = []
    lines.append(f"Batch directory: {batch_dir}")
    lines.append(
        "Aggregation rule: for each run, use metrics at the maximum training step in metrics.json."
    )
    lines.append("Grouping rule: runs are grouped by experiment path; different max_step values stay in the same group.")
    lines.append("Variance definition: population variance (ddof=0).")
    lines.append("")

    for group_key in sorted(groups.keys()):
        items = groups[group_key]
        metric_names = sorted({k for it in items for k in it["metrics"].keys()})

        lines.append("=" * 100)
        lines.append(f"Experiment group: {group_key}")
        lines.append(f"Run count: {len(items)}")
        lines.append("Run details:")
        for it in items:
            rel_path = it["file"].relative_to(batch_dir).as_posix()
            lines.append(f"  - {rel_path} | max_step={it['max_step']}")

        lines.append("Final-step metrics stats:")
        header = f"{'metric':<20} {'mean':>18} {'variance':>18}"
        lines.append(header)
        lines.append("-" * len(header))

        for metric in metric_names:
            values = []
            for it in items:
                value = it["metrics"].get(metric)
                if isinstance(value, (int, float)):
                    values.append(float(value))

            if not values:
                continue

            metric_mean = mean(values)
            metric_var = pvariance(values) if len(values) > 1 else 0.0
            lines.append(f"{metric:<20} {metric_mean:>18.10g} {metric_var:>18.10g}")

        lines.append("Raw values by run:")
        for metric in metric_names:
            values = []
            for it in items:
                value = it["metrics"].get(metric)
                if isinstance(value, (int, float)):
                    values.append(float(value))
            if values:
                joined = ", ".join(f"{v:.10g}" for v in values)
                lines.append(f"  {metric}: [{joined}]")

        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path