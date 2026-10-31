from dataclasses import dataclass


@dataclass
class LLMConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_groups: int = 4
    n_embd: int = 768
    theta: int = 500_000
    eps: float = 1e-5
