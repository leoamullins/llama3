# Kernel benchmarks

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
