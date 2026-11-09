from pathlib import Path

from torch.profiler import profile, schedule, ProfilerActivity

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

# warmup outside profiler
for s in range(3):
    trainer._train_step(s)


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
prof.export_chrome_trace("trace.json")
