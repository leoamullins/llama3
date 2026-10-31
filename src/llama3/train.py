import math
import os
import time

import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from llama3.config import LlamaConfig, TrainConfig, parse_configs
from llama3.data import DataLoaderLite
from llama3.model import LLaMA3

train_cfg, model_cfg = parse_configs(TrainConfig, LlamaConfig)

run_dir = os.path.join("runs", train_cfg.run_name)
latest_path = os.path.join(run_dir, "latest.pt")
best_path = os.path.join(run_dir, "best.pt")
export_dir = os.path.join(run_dir, "export")

# setup DDP
# torchrun command sets the env variables RANK, LOCAL_RANK, WORLD_SIZE
ddp = int(os.environ.get(key="RANK", default=-1)) != -1
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_world_size = 1
    ddp_local_rank = 0
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524_288  # 2^19 approx 0.5M tokens
B = 16  # micro batch size
T = 1024  # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"desired batch size {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")


train_loader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size
)

torch.set_float32_matmul_precision("high")

model = LLaMA3(model_cfg)
model.to("cuda")


start_step, best_val = 0, float("inf")
ckpt = None
if train_cfg.resume and os.path.exists(latest_path):
    ckpt = torch.load(latest_path, map_location=device)
    model.load_state_dict(ckpt["model"])

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 20
max_steps = 300


def get_lr(it):
    # linear warmup for warmup_steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


optimizer = raw_model.configure_optimizer(
    weight_decay=0.1, learning_rate=6e-4, device=device
)
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)

        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.requires_backward_grad_sync = micro_step == grad_accum_steps - 1
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # get lr
    lr = get_lr(step)
    # set lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = (
        train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    )
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(
            f"step{step}, loss={loss_accum.item():.6f}, lr = {lr:.4e} norm={norm:.4f}, dt={dt * 1000:.2f}ms, tok/s={tokens_per_sec}"
        )


if ddp:
    destroy_process_group()
