from torch.profiler import profile, schedule, ProfilerActivity

from llama3.config import LlamaConfig, TrainConfig, parse_configs
from llama3.train import Trainer

train_cfg, model_cfg = parse_configs(TrainConfig, LlamaConfig)
trainer = Trainer(train_cfg, model_cfg)

# warmup outside profiler
for s in range(3):
    trainer._train_step(s)


with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=2, active=5),
    record_shapes=True,
) as prof:
    for s in range(8):
        trainer._train_step(s)
        prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
prof.export_chrome_trace("trace.json")
