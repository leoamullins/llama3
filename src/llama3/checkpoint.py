import json
import os

import torch


def unwrap(model):
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def save_checkpoint(
    path, model, optimizer, step, val_loss, loader, train_cfg, model_cfg
):
    """
    call the master process only
    """
    raw = unwrap(model)
    ckpt = {
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "val_loss": float(val_loss),
        # master's position; other ranks re-derive theirs on load
        "loader_pos": loader.current_position,
        "train_cfg": train_cfg,  # pass asdict(cfg)
        "model_cfg": model_cfg,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        },
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer, loader, device, rank=0):
    """call on every rank"""
    ckpt = torch.load(path, map_location=device, weights_only=True)

    unwrap(model).load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    loader.current_position = ckpt["loader_pos"] + loader.B * loader.T * rank

    torch.set_rng_state(ckpt["rng"]["torch"].cpu())
    if ckpt["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(ckpt["rng"]["cuda"].cpu())

    start_step = ckpt["step"] + 1
    best_val = ckpt["val_loss"]
    del ckpt
    return start_step, best_val
