import math

import torch
from torch import nn
from torch.nn import functional as F

# --------------------------------------------------------------
# CREATING LLaMA 3 Model Architecture (Meta LLaMA)
# --------------------------------------------------------------


class SwiGLU(nn.Module):
    """
    SWiGLU implementation, SwiGLU(x) = (xW1) * SiLU(xW2)
    """

    def __init__(self, d_model, d_ffn):
        super().__init__()

        hidden_dim = int(2 * d_ffn / 3)

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model)

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
    MLP with d_ffn as 3.5 * d_model as per paper
    """

    def init(self, config):
        super().__init__()

        self.c_fc = nn.Linear(config.n_embd, 3.5 * config.n_embd)
        self.swiglu = SwiGLU(config.n_embd, 3.5 * config.n_embd)
        self.c_proj = nn.Linear(3.5 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.swiglu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # k,q,v proj
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output proj
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

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
        # attention
        # att size (B, nh, T, T)
        att = (q @ k.tranpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
