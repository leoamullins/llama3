import csv
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict

import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from llama3.checkpoint import (
    export_safetensors,
    load_checkpoint,
    save_checkpoint,
    unwrap,
)
from llama3.config import LlamaConfig, TrainConfig, parse_configs
from llama3.data import DataLoaderFineWeb
from llama3.model import LLaMA3

CSV_FIELDS = [
    "step",
    "train_loss",
    "val_loss",
    "lr",
    "grad_norm",
    "dt_ms",
    "tokens_per_sec",
]


class Trainer:
    def __init__(self, train_cfg, model_cfg, callbacks=()):
        self.cfg = train_cfg
        self.model_cfg = model_cfg
        self.callbacks = list(callbacks)

        self._setup_distributed()

        torch.manual_seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.cfg.seed)
        torch.set_float32_matmul_precision("high")

        self._setup_batch_sizes()
        self.train_loader = self.build_loader("train")
        self.val_loader = self.build_loader("val")

        self.start_step, self.best_val = 0, float("inf")
        self._build_model()
        self._build_optimizer()
        self._write_config()
        self._open_log()

    # ----------------------------------------------------------
    # setup
    def _setup_distributed(self):
        # torchrun command sets the env variables RANK, LOCAL_RANK, WORLD_SIZE
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        if self.ddp:
            assert torch.cuda.is_available()
            init_process_group(backend="nccl")
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.device = f"cuda:{self.local_rank}"
            torch.cuda.set_device(self.device)
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            print(f"using device {self.device}")
        self.master_process = self.rank == 0
        # autocast takes "cuda", not "cuda:0"
        self.device_type = self.device.split(":")[0]

    def _setup_batch_sizes(self):
        tokens_per_micro_step = self.cfg.B * self.cfg.T * self.world_size
        assert self.cfg.total_batch % tokens_per_micro_step == 0, (
            f"total_batch {self.cfg.total_batch} must divide evenly by "
            f"B * T * world_size = {tokens_per_micro_step}"
        )
        self.grad_accum_steps = self.cfg.total_batch // tokens_per_micro_step
        if self.master_process:
            print(f"desired batch size {self.cfg.total_batch}")
            print(f"=> calculated gradient accumulation steps: {self.grad_accum_steps}")

    def build_loader(self, split):
        # override to train on another dataset, e.g. DataLoaderLite
        return DataLoaderFineWeb(
            B=self.cfg.B,
            T=self.cfg.T,
            process_rank=self.rank,
            num_processes=self.world_size,
            data_path=self.cfg.data_path,
            split=split,
        )

    def _build_model(self):
        model = LLaMA3(self.model_cfg)
        model.to(self.device)

        self._resume_ckpt = None
        if self.cfg.resume and os.path.exists(self.cfg.latest_path):
            self._resume_ckpt = load_checkpoint(
                self.cfg.latest_path, model, self.train_loader, self.device
            )
            self.start_step = self._resume_ckpt["step"] + 1
            self.best_val = self._resume_ckpt["best_val"]
            if self.master_process:
                print(f"resumed from {self.cfg.latest_path} at step {self.start_step}")

        if self.cfg.compile:
            model = torch.compile(model)
        if self.ddp:
            model = DDP(model, device_ids=[self.local_rank])

        self.model = model
        self.raw_model = unwrap(model)

    def _build_optimizer(self):
        self.optimizer = self.raw_model.configure_optimizer(
            weight_decay=self.cfg.weight_decay,
            learning_rate=self.cfg.max_lr,
            device=self.device,
        )
        if self._resume_ckpt is not None:
            self.optimizer.load_state_dict(self._resume_ckpt["optimizer"])
            self._resume_ckpt = None

    def _write_config(self):
        if not self.master_process:
            return
        os.makedirs(self.cfg.run_dir, exist_ok=True)
        # rewritten on resume, so it describes the latest invocation
        path = os.path.join(self.cfg.run_dir, "config.json")
        with open(path, "w") as f:
            json.dump(
                {"train": asdict(self.cfg), "model": asdict(self.model_cfg)},
                f,
                indent=2,
            )

    def _open_log(self):
        self.log_file, self.log_writer = None, None
        if not self.master_process:
            return
        os.makedirs(self.cfg.run_dir, exist_ok=True)
        resumed = self.start_step > 0
        path = os.path.join(self.cfg.run_dir, "metrics.csv")
        self.log_file = open(path, "a" if resumed else "w", newline="")
        self.log_writer = csv.DictWriter(self.log_file, fieldnames=CSV_FIELDS)
        if not resumed:
            self.log_writer.writeheader()

    # ----------------------------------------------------------
    # each step
    def get_lr(self, it):
        # linear warmup for warmup_steps
        if it < self.cfg.warmup_steps:
            return self.cfg.max_lr * (it + 1) / self.cfg.warmup_steps
        if it >= self.cfg.max_steps:
            return self.cfg.min_lr
        decay_ratio = (it - self.cfg.warmup_steps) / (
            self.cfg.max_steps - self.cfg.warmup_steps
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.cfg.min_lr + coeff * (self.cfg.max_lr - self.cfg.min_lr)

    def _autocast(self):
        return torch.amp.autocast(device_type=self.device_type, dtype=torch.bfloat16)

    def _all_reduce(self, tensor):
        if self.ddp:
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)

    def _train_step(self, step):
        t0 = time.time()

        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        loss_accum = torch.tensor(0.0, device=self.device)
        for micro_step in range(self.grad_accum_steps):
            x, y = self.train_loader.next_batch()
            x, y = x.to(self.device), y.to(self.device)

            # only sync gradients across GPUs on the last micro step
            sync = micro_step == self.grad_accum_steps - 1
            with nullcontext() if sync or not self.ddp else self.model.no_sync():
                with self._autocast():
                    _, loss = self.model(x, y)
                loss = loss / self.grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip
        )
        self.optimizer.step()
        self.optimizer.zero_grad()

        self._all_reduce(loss_accum)

        if self.device_type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        tokens_processed = (
            self.cfg.B * self.cfg.T * self.grad_accum_steps * self.world_size
        )
        return {
            "step": step,
            "train_loss": loss_accum.item(),
            "val_loss": None,
            "lr": lr,
            "grad_norm": norm.item(),
            "dt_ms": dt * 1000,
            "tokens_per_sec": tokens_processed / dt,
        }

    # ----------------------------------------------------------
    # periodically
    def evaluate(self):
        self.model.eval()
        self.val_loader.reset()
        loss_accum = torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            for _ in range(self.cfg.val_steps):
                x, y = self.val_loader.next_batch()
                x, y = x.to(self.device), y.to(self.device)
                with self._autocast():
                    _, loss = self.model(x, y)
                loss_accum += loss / self.cfg.val_steps
        self._all_reduce(loss_accum)
        self.model.train()
        return loss_accum.item()

    def _save(self, path, step):
        if self.master_process:
            save_checkpoint(
                path,
                self.model,
                self.optimizer,
                step,
                self.best_val,
                self.train_loader,
                asdict(self.cfg),
                asdict(self.model_cfg),
            )
        if self.ddp:
            dist.barrier()

    def _log(self, metrics):
        if not self.master_process:
            return
        val_loss = metrics["val_loss"]
        val_str = "" if val_loss is None else f" val={val_loss:.6f}"
        print(
            f"step {metrics['step']}, loss={metrics['train_loss']:.6f}{val_str}, "
            f"lr={metrics['lr']:.4e}, norm={metrics['grad_norm']:.4f}, "
            f"dt={metrics['dt_ms']:.2f}ms, tok/s={metrics['tokens_per_sec']:.2f}"
        )
        self.log_writer.writerow(metrics)
        self.log_file.flush()

    def _fire(self, hook, *args):
        for callback in self.callbacks:
            getattr(callback, hook)(self, *args)

    # ----------------------------------------------------------
    # main loop
    def fit(self):
        self._fire("on_train_start")

        for step in range(self.start_step, self.cfg.max_steps):
            done = step + 1
            last_step = done == self.cfg.max_steps

            metrics = self._train_step(step)

            # the first step evaluates so the val curve has a starting point
            # the last step always evaluates and saves, so the final state is kept
            if step == 0 or last_step or done % self.cfg.eval_every == 0:
                val_loss = self.evaluate()
                metrics["val_loss"] = val_loss
                if val_loss < self.best_val:
                    self.best_val = val_loss
                    self._save(self.cfg.best_path, step)
                self._fire("on_eval_end", step, val_loss)

            if last_step or done % self.cfg.save_every == 0:
                self._save(self.cfg.latest_path, step)

            self._log(metrics)
            self._fire("on_step_end", step, metrics)

        self._finish()

    def _finish(self):
        self._fire("on_train_end")

        if self.master_process:
            self.log_file.close()
            if os.path.exists(self.cfg.best_path):
                export_safetensors(
                    self.cfg.best_path,
                    self.model,
                    self.cfg.export_dir,
                    asdict(self.model_cfg),
                )
        if self.ddp:
            dist.barrier()
            destroy_process_group()


if __name__ == "__main__":
    train_cfg, model_cfg = parse_configs(TrainConfig, LlamaConfig)
    Trainer(train_cfg, model_cfg).fit()
