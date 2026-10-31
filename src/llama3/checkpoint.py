import json
import os

import torch
from safetensors.torch import save_model


def unwrap(model):
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def save_checkpoint(
    path, model, optimizer, step, best_val, loader, train_cfg, model_cfg
):
    """
    call the master process only
    """
    raw = unwrap(model)
    ckpt = {
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val": float(best_val),
        # master's shard and position; other ranks re-derive theirs on load
        "loader": loader.state_dict(),
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


def load_checkpoint(path, model, loader, device):
    """
    call on every rank, returns the checkpoint
    the optimizer doesn't exist yet at this point, so load ckpt["optimizer"] later
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)

    unwrap(model).load_state_dict(ckpt["model"])
    loader.load_state_dict(ckpt["loader"])

    torch.set_rng_state(ckpt["rng"]["torch"].cpu())
    if ckpt["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(ckpt["rng"]["cuda"].cpu())

    return ckpt


def export_safetensors(ckpt_path, model, export_dir, model_cfg):
    """
    call the master process only, overwrites the weights of model with the checkpoint's
    """
    raw = unwrap(model)
    device = next(raw.parameters()).device
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    raw.load_state_dict(ckpt["model"])

    os.makedirs(export_dir, exist_ok=True)
    # save_model rather than save_file, wte and lm_head share a tensor
    save_model(raw, os.path.join(export_dir, "model.safetensors"))
    with open(os.path.join(export_dir, "config.json"), "w") as f:
        json.dump(model_cfg, f, indent=2)
