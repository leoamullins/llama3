import math

import torch
from torch import nn
from torch.nn import functional as F

# --------------------------------------------------------------
# CREATING LLaMA 3 Model Architecture (Meta LLaMA)
# --------------------------------------------------------------


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=8192, theta=500_000):
        super().__init__()
        self.head_dim = head_dim

        # RoPE operates on pairs of dimensions, so we calc frequencies for half the head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # precompute the cos and sin freq ip to max_seq_len
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqss = torch.outer(t, self.inv_freq)

        # duplicate each frequency to match head dim
        emb = torch.cat((freqss, freqss), dim=1)

        # Cache cos and sin as [1, 1, max_seq_len, head_dim]
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    def _rotate_half(self, x):
        """
        split tensor in half along last dim and rotates one half
        """
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, seq_len):
        # slice the precomputed cache to match the current seq length
        cos = self.cos_cached[:, :, :seq_len, :]  # (1,1,seq_len,head_dim)
        sin = self.sin_cached[:, :, :seq_len, :]  # (1,1,seq_len,head_dim)

        # apply rotation formula
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)

        return q_rotated, k_rotated


class SwiGLU(nn.Module):
    """
    SWiGLU implementation, SwiGLU(x) = (xW1) * SiLU(xW2)
    """

    def __init__(self, d_model, hidden_dim):
        super().__init__()

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model)
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

        self.rmsnorm = nn.RMSNorm(normalized_shape=(config.n_embd,))

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


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # k,q,v proj
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output proj
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # RoPE
        self.rope = RotaryPositionEmbedding(
            self.n_embd // self.n_head, theta=config.theta
        )

        # the mask for causality
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()
        # get qkv
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)

        # nh no of heads and hs is head size and C is number of channels
        # k,q,v have shape (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # RoPE
        q, k = self.rope(q, k, T)

        # attention
        # att size (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.n_embd
        self.n_heads = config.n_head
        self.n_groups = config.n_groups
        self.group_size = self.n_heads // self.n_groups
        self.head_dim = self.d_model // self.n_heads

        assert self.n_heads % self.n_groups == 0, "n_head must be divisible by n_groups"

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim)
        self.k_proj = nn.Linear(self.d_model, self.n_groups * self.head_dim)
        self.v_proj = nn.Linear(self.d_model, self.n_groups * self.head_dim)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj._is_residual = True

        self.rope = RotaryPositionEmbedding(self.head_dim, theta=config.theta)

        # Causal mask: lower triangular is 1, upper is 0
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()

        # Project queries, keys, and values -> (B, num_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_groups, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_groups, self.head_dim).transpose(1, 2)

        # RoPE
        q, k = self.rope(q, k, T)

        # Expand K and V heads to match Q heads along head dimension
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = scores.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        # Aggregate values and restore dimensions
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

    def forward(self, x):
        x = x + self.attn(self.rms_1(x))
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

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.rms_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
