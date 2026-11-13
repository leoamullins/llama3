# Kernel benchmarks

## Cumulative ablation: RTX 4090, 2026-11-13

Start from FP32 eager and add one optimisation at a time. Each config is a real 500-step training run:
- `B=8, T=1024`, 65,536 tokens/step (8 grad-accum micro-steps), ~33M tokens of FineWeb-Edu
- 50 warmup steps, eval every 25 steps (20 val batches), seed 1337
- torch 2.15.0+cu132, Triton 3.9.0, commit `1d1502f` + the ablation flags (`--precision`, `--fused_adamw`, `--save`)

Script: `ablation.sh` (not committed). ms/step and tok/s are medians over steps ≥10, excluding eval steps.
Peak memory is `torch.cuda.max_memory_allocated`.

| Run | Adds | ms/step | tok/s | vs C0 | Step gain | Peak mem | Final val | Δ val vs C0 |
|---|---|---|---|---|---|---|---|---|
| c0_fp32 | baseline (fp32, manual attn, for-loop AdamW) | 2621 | 25.0k | 1.00× | — | 20.30 GB | 5.268 | — |
| c1_tf32 | TF32 matmuls | 2175 | 30.1k | 1.20× | +20% | 20.30 GB | 5.242 | −0.026 |
| c2_bf16 | bf16 autocast | 1654 | 39.6k | 1.58× | +32% | 18.77 GB | 5.242 | −0.026 |
| c3_sdpa | SDPA attention | 701 | 93.4k | 3.74× | **+136%** | 11.31 GB | 5.236 | −0.032 |
| c4_fused_adamw | fused AdamW | 697 | 94.0k | 3.76× | +0.6% | 11.31 GB | 5.238 | −0.030 |
| c5_triton | Triton SwiGLU + RMSNorm | 669 | 97.9k | 3.92× | +4.2% | 10.78 GB | 5.240 | −0.028 |
| c6_fused_ce | fused linear cross-entropy | 638 | 102.8k | 4.11× | +5.0% | **6.37 GB** | 5.248 | −0.020 |
| c7_compile | `torch.compile` | **566** | **115.9k** | **4.64×** | +12.8% | 6.09 GB | 5.243 | −0.025 |
| c0_fp32_seed2 | C0, `--seed 2` (noise floor) | 2621 | 25.0k | — | — | 20.30 GB | 5.243 | −0.025 |

**Stability:** all 9 runs had 0 non-finite losses or grad norms. None had a loss spike (train loss > 1.5× the rolling
median of the previous 20 steps). Grad norm divided by C0's at the same step has a median of 1.000–1.008 for every config.
The largest per-step outlier is 1.6×, against 3.6× between the two C0 seeds.

**Takeaways**
- **Quality is unchanged.** Every config ends within 0.032 val loss of C0. Changing only the seed moves it by 0.025 at
  the end, and by up to 0.061 at earlier evals. C0 itself is the slight outlier: any numerical change, such as fp32 → tf32,
  sends training down a different path, the same way a new seed does.
- **SDPA is the biggest win** (2.4× on its own; 18.8 → 11.3 GB). The manual path stores every T×T attention matrix.
- **Fused CE is the biggest memory win after SDPA** (10.8 → 6.4 GB) because the `B·T×V` logits are never stored.
- **Compile still adds 12.8%** on top of the Triton kernels. The profiler, though, showed `_rmsnorm_fwd` and
  `_ce_rows_kernel` running slower under compile. The leave-one-out runs (`--compile --no-triton_*`) are the next check.
- **Fused AdamW is ~0.6%** because the optimizer step runs once per 8 micro-steps.
- **The overall result is 4.64× the throughput with 70% less memory**, at the same loss.

## RTX 4090 — 2026-11-13

Setup: 124M-param LLaMA-3 config (12 layers, d=768), `B=8, T=1024`, bf16 autocast, `data/input.txt`.
torch 2.13.0+cu129, Triton 3.7.1, CUDA 13.2 host. Run with `python -m llama3.profiling`;
ms/step is the mean of 10 steps after 3 warmup steps, without the profiler.

| Run | Flags | ms/step | vs eager | tok/s | Peak mem |
|---|---|---|---|---|---|
| eager | `--no-triton_swiglu --no-triton_rmsnorm --no-fused_ce` | 90.4 | — | 90,620 | 10.8 GB |
| compile | eager flags + `--compile` | **72.7** | −19.6% | 112,677 | 7.3 GB |
| swiglu | `--no-triton_rmsnorm --no-fused_ce` | 87.7 | −3.0% | 93,415 | 10.3 GB |
| fused_ce | `--no-triton_swiglu --no-triton_rmsnorm` | 86.6 | −4.2% | 94,542 | 6.4 GB |
| swiglu_ce | `--no-triton_rmsnorm` | 83.7 | −7.4% | 97,821 | 5.9 GB |
| all | *(defaults: all kernels on)* | 83.0 | −8.2% | 98,674 | 5.9 GB |
| all_compile | `--compile` | 77.6 | −14.2% | 105,551 | **5.6 GB** |

### Custom kernel time, 5 profiled steps

| Kernel | Launches | all | all_compile |
|---|---|---|---|
| `_swiglu_fwd` | 60 | 8.29 ms | 8.39 ms |
| `_swiglu_bwd` | 60 | 13.70 ms | 13.91 ms |
| `_rmsnorm_fwd` | 125 | 2.06 ms | 10.26 ms |
| `_rmsnorm_bwd` | 125 | 9.60 ms | 9.49 ms |
| `_ce_rows_kernel` | 40 | 8.96 ms | 25.01 ms |

### Takeaways
- Fused CE gives the largest single win and roughly halves peak memory (logits are never materialised).
- RMSNorm saves only ~0.7 ms/step: it is a small share of the step.
- `torch.compile` alone still beats all custom kernels by ~10 ms/step; it fuses the ops not yet covered (residual adds, RoPE, small elementwise).
- Under `--compile`, `_rmsnorm_fwd` (5×) and `_ce_rows_kernel` (2.8×) run slower than in eager mode, which explains most of all_compile vs compile. Hypothesis: lost divisibility/alignment specialisation on the stride-indexed kernels when called from Inductor-generated code. SwiGLU (flat offsets) is unaffected.
- `_rmsnorm_bwd` is ~77 µs/call vs ~16 µs for the forward (expected 1.5–2×). Next: sweep `n_programs` × `num_warps`.

### Earlier A40 results (2026-11-10, traces deleted)
Eager 163.7 → SwiGLU + fused CE 152.8 ms/step (−6.7%); peak memory 10.3 → 5.8 GB. `torch.compile` SwiGLU 7.46 vs Triton 7.58 ms/step.
