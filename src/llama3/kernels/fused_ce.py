import torch
import triton
import triton.language as tl

@triton.jit
def _ce_rows_kernel(logits_ptr, stride_row, targets_ptr, loss_ptr, n_cols, scale, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) # stopping overflow of row * stride

    logits_ptr += row * stride_row
    target = tl.load(targets_ptr + row)

    # pass 1 : online softmax -> log-sum-exp
    m = float("-inf")
    s = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        x = tl.load(logits_ptr + cols, mask = cols < n_cols, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new))
        m = m_new

    lse = m + tl.log(s)

    # loss
    x_target = tl.load(logits_ptr + target).to(tl.float32)
    tl.store(loss_ptr + row, lse - x_target)

    # pass 2 overwite the row iwht its gradient
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start  + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(logits_ptr + cols, mask=mask).to(tl.float32)

        p = tl.exp(x - lse)
        p = tl.where(cols == target, p - 1, p )
        p *= scale
        tl.store(logits_ptr + cols, p.to(logits_ptr.dtype.element_ty), mask=mask)

def _ce_rows_triton(logits, targets, scale):
    assert logits.is_cuda and logits.dim() == 2
    assert logits.stride(1) == 1

    targets = targets.contiguous()

    n_rows, n_cols = logits.shape

    loss = torch.empty(n_rows, device=logits.device, dtype=torch.float32)

    _ce_rows_kernel[(n_rows,)](
        logits,
        logits.stride(0),
        targets,
        loss,
        n_cols,
        scale,
        BLOCK_SIZE=4096,
        num_warps = 8,
    )

    return loss


def _ce_rows_torch(logits, targets, scale):
    loss = F.cross_entropy(logits, targets)
    return loss

class FusedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, targets, chunk_size, row_fn):
        N = x.shape[0]
        xb = x.to(torch.bfloat16)
        wb = weight.to(torch.bfloat16)

        dx   = torch.empty_like(xb)                                          # (N, C)
        dw   = torch.zeros(weight.shape, device=x.device, dtype=torch.float32)  # (V, C) fp32 accumulator
        loss = torch.empty(N, device=x.device, dtype=torch.float32)

        for i in range(0, N, chunk_size):
            sl = slice(i, i+chunk_size)
            xc = xb[sl]

            logits = xc @ wb.T

            loss[sl] = row_fn(logits, targets[sl], 1.0 / N)
            dlogits = logits

            dx[sl] = dlogits @ wb
            dw += (dlogits.T @ xc).float()

        ctx.save_for_backward(dx,dw)
        ctx.dtypes = (x.dtype, weight.dtype)
        return loss.sum() / N

    @staticmethod
    def backward(ctx, grad_out):
        # grad_out: d(final loss)/d(our output). 1.0 normally,
        # 1/grad_accum_steps because train.py does `loss / grad_accum_steps`
        dx, dw = ctx.saved_tensors
        x_dtype, w_dtype = ctx.dtypes
        grad_x = (dx * grad_out).to(x_dtype)           # (N, C)
        grad_w = (dw * grad_out).to(w_dtype)           # (V, C)
        # one entry per forward input: x, weight, targets, chunk_size, row_fn
        return grad_x, grad_w, None, None, None



def fused_linear_cross_entropy(x, weight, targets, chunk_size=1024, row_fn=None):
    if row_fn is None:
        row_fn = _ce_rows_triton

    C = x.shape[-1]
    x = x.reshape(-1,C)

    targets = targets.reshape(-1)

    assert x.shape[0] == targets.shape[0], "one target per token"
    assert weight.shape[1] == C, "weight must be (vocab, n_embd)"

    return FusedLinearCrossEntropy.apply(x, weight, targets, chunk_size, row_fn)


