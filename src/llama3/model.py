import math
import inspect

import torch
from torch import nn
from torch.nn import functional as F

# --------------------------------------------------------------
# CREATING LLaMA 3 Model Architecture (Meta LLaMA)
# --------------------------------------------------------------


# --------------------------------------------------------------
# RoPE functions rather than in class, so i can reuse cos and sin
def precompute_rope(head_dim, max_seq_len, theta):
    # RoPE operates on pairs of dimensions, so we calc frequencies for half the head_dim
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

    # precompute the cos and sin freq ip to max_seq_len
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqss = torch.outer(t, inv_freq)

    # duplicate each frequency to match head dim
    emb = torch.cat((freqss, freqss), dim=1)

    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    # broadcast: (T, head_dim) -> (B, nh, T, head_dim)
    return x * cos + rotate_half(x) * sin


# --------------------------------------------------------------
class SwiGLU(nn.Module):
    """
    SWiGLU implementation, SwiGLU(x) = (xW2) * SiLU(xW1)
    """

    def __init__(self, d_model, hidden_dim):
        super().__init__()

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3._is_residual = True

    def forward(self, x):
        # first linear projection for the gate, activated by SiLU
        gate = F.silu(self.w1(x))
        # second linear projected
        data = self.w2(x)
        # element wise multiplicated, followed by final projection
        return self.w3(gate * data)


class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.rmsnorm = nn.RMSNorm(normalized_shape=(config.n_embd,), eps=config.eps)

    def forward(self, x):
        return self.rmsnorm(x)


class MLP(nn.Module):
    """
    MLP with SwiGLU hidden size of 3.5 * d_model, matching Llama 3 8B and 70B
    """

    def __init__(self, config):
        super().__init__()

        self.swiglu = SwiGLU(config.n_embd, int(3.5 * config.n_embd))

    def forward(self, x):
        return self.swiglu(x)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.n_embd
        self.n_heads = config.n_head
        self.n_groups = config.n_groups
        self.group_size = self.n_heads // self.n_groups
        self.head_dim = self.d_model // self.n_heads

        assert self.n_heads % self.n_groups == 0, "n_head must be divisible by n_groups"

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_groups * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_groups * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj._is_residual = True

        # Causal mask: lower triangular is 1, upper is 0
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x, cos, sin):
        B, T, C = x.size()

        # Project queries, keys, and values -> (B, num_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # k, v share heads within a group: (B, n_groups, T, head_dim)
        k = self.k_proj(x).view(B, T, self.n_groups, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_groups, self.head_dim).transpose(1, 2)

        # RoPE
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Expand K and V heads to match Q heads along head dimension
        # (B, n_groups, T, head_dim) -> (B, n_heads, T, head_dim)
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # Scaled dot-product attention
        # (B, n_heads, T, head_dim) @ (B, n_heads, head_dim, T) -> (B, n_heads, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # broadcast: (1, 1, T, T) -> (B, n_heads, T, T)
        attn = scores.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        # Aggregate values and restore dimensions
        # (B, n_heads, T, head_dim) -> (B, T, C)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(y)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rms_1 = RMSNorm(config=config)
        self.attn = GroupedQueryAttention(config=config)
        self.rms_2 = RMSNorm(config=config)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.rms_1(x), cos, sin)
        x = x + self.mlp(self.rms_2(x))
        return x


class LLaMA3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "rms_f": RMSNorm(config),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing (diverging from the Meta for efficiency for my own compute)
        self.transformer.wte.weight = self.lm_head.weight

        # precalculate the rope sin and cos
        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rope(head_dim, config.block_size, config.theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # initialise weights with Kaiming Init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # handle residual/output layers
            if getattr(module, "_is_residual", False):
                std = 1.0 / math.sqrt(2 * self.config.n_layer * self.config.n_embd)
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            # handle input / gate layers
            else:
                torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx of shape (B, T)
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        )

        # forward token
        tok_emb = self.transformer.wte(idx)
        x = tok_emb

        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for block in self.transformer.h:
            x = block(x, cos, sin)

        x = self.transformer.rms_f(x)
        # (B, T, n_embd) -> (B, T, vocab_size)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def configure_optimizer(self, weight_decay, learning_rate, device):
        # start with all of candidate params (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any 2D params are weight decayed, otherwise no.
        # tensors in matmuls + embeddings decay, all biases and layer norms dont.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nondecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nondecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nondecay = sum(p.numel() for p in nondecay_params)
        print(
            f"number of decayed param tensors {len(decay_params)} with {num_decay_params:,} parameters"
        )
        print(
            f"number of nondecayed param tensors {len(nondecay_params)} with {num_nondecay:,} parameters"
        )
        # create AdamW
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer
