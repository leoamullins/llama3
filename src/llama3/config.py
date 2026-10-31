import argparse
import os
from dataclasses import dataclass, fields


@dataclass
class TrainConfig:
    # run / io
    run_name: str = "test"
    runs_root: str = "runs"  # on RunPod: /workspace/runs (network volume)
    data_path: str = "edu_fineweb10B"  # directory of train and val .npy shards
    resume: bool = False

    # batch
    B: int = 8  # micro-batch size per GPU
    T: int = 1024  # sequence length
    total_batch: int = 8192  # tokens per optimizer step (B*T*grad_accum*world_size)

    # optimisation
    max_lr: float = 6e-4
    min_lr_ratio: float = 0.1  # min_lr = max_lr * min_lr_ratio
    warmup_steps: int = 20
    max_steps: int = 300
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # eval / logging / saving
    eval_every: int = 25
    val_steps: int = 3
    gen_every: int = 100
    save_every: int = 100

    # misc
    seed: int = 1337
    compile: bool = False

    @property
    def run_dir(self):
        return os.path.join(self.runs_root, self.run_name)

    @property
    def latest_path(self):
        return os.path.join(self.run_dir, "latest.pt")

    @property
    def best_path(self):
        return os.path.join(self.run_dir, "best.pt")

    @property
    def export_dir(self):
        return os.path.join(self.run_dir, "export")

    @property
    def min_lr(self):
        return self.max_lr * self.min_lr_ratio


@dataclass
class LlamaConfig:
    vocab_size: int = 50304  # GPT-2 tokenizer, padded to a multiple of 64
    block_size: int = 1024  # max context; RoPE tables are built to this length
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12  # head_dim = n_embd // n_head
    n_groups: int = 4  # KV heads; = n_head gives plain multi-head attention
    theta: float = 10_000.0  # RoPE base
    eps: float = 1e-5  # RMSNorm epsilon
    sdpa: bool = True  # F.scaled_dot_product_attention; --no-sdpa for the manual path


def parse_configs(*classes):
    parser = argparse.ArgumentParser()
    seen = set()
    for cls in classes:
        for f in fields(cls):
            assert f.name not in seen, f"duplicate config field: {f.name}"

            seen.add(f.name)
            if f.type is bool:
                parser.add_argument(
                    f"--{f.name}",
                    action=argparse.BooleanOptionalAction,
                    default=f.default,
                )
            else:
                parser.add_argument(f"--{f.name}", type=f.type, default=f.default)

    args = vars(parser.parse_args())
    return [cls(**{f.name: args[f.name] for f in fields(cls)}) for cls in classes]
