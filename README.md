# LLaMA 3

A from-scratch Llama 3-style model as per https://arxiv.org/abs/2407.21783
## Changes from GPT2

### RoPE

This model has RoPE as apposed the absolute positional encodings:
- RoPE rotates q and k inside every attention layer by an angle proportional to position, so attnetion scores depend only on the relative offeset $m-n$
- No position parameters, no position information in the residual stream, and it generalises better beyond the trained length

### RMSNorm (replaces LayerNorm)

- LayerNorm subtracts the mean, divides by the standard deviation, then applies a learned scale and bias.
- RMSNorm skips the mean subtract and bias: $y =  \frac{x}{g \times \sqrt{\text{mean}(x^2) + \epsilon}} \times \gamma$
- Fewer operations and params, and in practice is just as stable.

### SwiGLU (replaces the GELU MLP)

- GPT-2: down(GELU(up(x))), with a hidden size of $4d$
- SwiGLU: down(SiLU(gate(x))) $\odot$ up(x), one branch acts as learned gate on other
- This uses three matrices instead of two, so the hidden size drops to about 8/3*d to keep the parameter count the same.

### Grouped Query Attention

- In multihead attention every query heads has its own key and value heads
- In GQA, groups of query heads shard one K/V head (e.g. 12 query heads, 4KV Heads)
- Next i will add a KV Cache to make this even more efficient and inline with the paper

## Architecture

Below see the architecture I have implemented (minus the KV cache):

![LLaMA 3 architecture](figures/llama3.png)