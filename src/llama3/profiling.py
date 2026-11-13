from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule

from llama3.config import LlamaConfig, TrainConfig, parse_configs
from llama3.data import DataLoaderLite
from llama3.train import Trainer

# profile on the small text file in the repo, so no FineWeb shards are needed
INPUT_TXT = Path(__file__).resolve().parents[2] / "data" / "input.txt"


class ProfilingTrainer(Trainer):
    def build_loader(self, split):
        return DataLoaderLite(
            B=self.cfg.B,
            T=self.cfg.T,
            process_rank=self.rank,
            num_processes=self.world_size,
            data_path=str(INPUT_TXT),
            split=split,
        )


train_cfg, model_cfg = parse_configs(TrainConfig, LlamaConfig)
trainer = ProfilingTrainer(train_cfg, model_cfg)

# warmup outside profiler (also triggers torch.compile when --compile is set)
for s in range(3):
    trainer._train_step(s)

# timed steps without the profiler, whose overhead would skew the numbers
torch.cuda.reset_peak_memory_stats()
timed = [trainer._train_step(s) for s in range(10)]
dt = sum(m["dt_ms"] for m in timed) / len(timed)
tok_s = sum(m["tokens_per_sec"] for m in timed) / len(timed)
peak_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"[bench] {train_cfg.run_name}: {dt:.1f} ms/step, {tok_s:,.0f} tok/s, peak {peak_gb:.1f} GB")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=2, active=5),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for s in range(8):
        trainer._train_step(s)
        prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
trace_path = Path(train_cfg.run_dir) / f"trace_{train_cfg.run_name}.json"
trace_path.parent.mkdir(parents=True, exist_ok=True)
prof.export_chrome_trace(str(trace_path))
print(f"[trace] wrote {trace_path}")
