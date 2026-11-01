"""Complete, local trusted C1 branch checkpoints; original training files untouched.

The boundary is after diagnostics at adaptive_begin, BEFORE its C2 weight update.
load() restores the state; continuation must explicitly apply the next phase's
boundary transition. It does not silently reset or advance the optimizer.
"""
import copy, hashlib, os, random
from pathlib import Path
import numpy as np
import torch

DATA_KEYS = (
    "num_bcs",
    "train_x_group",
    "train_x_col",
    "train_x_bc",
    "train_x_data",
    "train_y_data",
    "train_x_frame",
    "train_x",
    "train_y",
    "test_x",
    "test_y",
    "pde_loss_weights",
    "pde_loss_weights_vec",
    "pde_loss_list",
)


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state().cpu(),
        cuda=[x.cpu() for x in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from checkpoint")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def save(path, model, identity, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(
        format_version=1,
        identity=identity,
        metadata=metadata or {},
        boundary="after_record_before_C2_update",
        step=model.train_state.step,
        model_state_dict=model.net.state_dict(),
        optimizer_state_dict=model.opt.state_dict(),
        optimizer_name=model.opt_name,
        scheduler_state=(
            model.lr_scheduler.state_dict() if model.lr_scheduler else None
        ),
        train_state=copy.deepcopy(vars(model.train_state)),
        loss_history=copy.deepcopy(vars(model.losshistory)),
        data={key: copy.deepcopy(getattr(model.data, key)) for key in DATA_KEYS},
        geometry={
            key: copy.deepcopy(value)
            for key, value in vars(model.data.geometry_partitions).items()
            if key != "geom"
        },
        data_loss_weight=model.data_loss_weight,
        network_training=model.net.training,
        loss_weights=copy.deepcopy(model.loss_weights),
        rng=rng_state(),
    )
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, model, identity):
    # Only checkpoints generated locally by this runner are accepted.
    state = torch.load(
        path, weights_only=False, map_location=next(model.net.parameters()).device
    )
    if state["format_version"] != 1 or state["identity"] != identity:
        raise ValueError("checkpoint configuration/code identity mismatch")
    if model.opt_name != state["optimizer_name"]:
        raise ValueError("compile the saved optimizer before restoration")
    model.net.load_state_dict(state["model_state_dict"])
    model.opt.load_state_dict(state["optimizer_state_dict"])
    if state["scheduler_state"] is not None:
        model.lr_scheduler.load_state_dict(state["scheduler_state"])
    vars(model.train_state).clear()
    vars(model.train_state).update(copy.deepcopy(state["train_state"]))
    vars(model.losshistory).clear()
    vars(model.losshistory).update(copy.deepcopy(state["loss_history"]))
    for k, v in state["data"].items():
        setattr(model.data, k, copy.deepcopy(v))
    vars(model.data.geometry_partitions).update(copy.deepcopy(state["geometry"]))
    model.data_loss_weight = state["data_loss_weight"]
    model.loss_weights = state["loss_weights"]
    model.net.train(state["network_training"])
    model.stop_training = False
    model.f_partition = None
    restore_rng(state["rng"])
    return state
