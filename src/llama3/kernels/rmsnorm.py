import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd(
    x_ptr, w_ptr, y_ptr, rstd_ptr, stride_row, D, eps, BLOCK_D: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)

    mask = cols < D

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    rstd = tl.math.rsqrt(tl.sum(x * x) / D + eps)

    # here the y is only correct if it shares x's stride. So use something like empty_like(x2d)
    tl.store(
        y_ptr + row * stride_row + cols,
        (x * rstd * w).to(y_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _rmsnorm_bwd(
    dy_ptr,
    x_ptr,
    w_ptr,
    rstd_ptr,
    dx_ptr,
    dw_partial_ptr,
    stride_row,
    N,
    D,
    rows_per_program,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    dw_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    row_start = pid * rows_per_program
    for row in range(row_start, tl.minimum(row_start + rows_per_program, N)):
        x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        dy = tl.load(dy_ptr + row * stride_row + cols, mask=mask, other=0.0).to(
            tl.float32
        )

        rstd = tl.load(rstd_ptr + row)

        x_hat = x * rstd
        g = dy * w
        dx = rstd * (g - x_hat * (tl.sum(g * x_hat) / D))
        tl.store(
            dx_ptr + row * stride_row + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask
        )

        dw_acc += dy * x_hat

    tl.store(dw_partial_ptr + pid * D + cols, dw_acc, mask=mask)


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def foward(ctx, x, w, eps):
        # B, T, D
        shape = x.shape

        # (B * T, D), so N = BT
        x2d = x.reshape(-1, shape[-1]).contiguous()
        N, D = x2d.shape

        y = torch.empty_like(x2d)  # smae stride as x2d

        rstd = torch.empty(N, dtype=torch.float32)

        BLOCK_D = triton.next_power_of_2(D)
        _rmsnorm_fwd[(N,)](
            x2d, w, y, rstd, x2d.stride(0), D, eps, BLOCK_D=BLOCK_D, num_warps=4
        )

        ctx.save_for_backward(x2d, w, rstd)
        ctx.shape = shape
        ctx.BLOCK_D = BLOCK_D
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        x2d, w, rstd = ctx.saved_tensors
        N, D = x2d.shape

        dy2d = dy.reshape(-1, D).contiguous()
        dx = torch.empty_like(dy2d)

        n_programs = torch.cuda.get_device_properties(x2d.device).multi_process_count
        rows_per_program = torch.cdiv(N, n_programs)
        dw_partial = torch.empty((n_programs, D), dtype=torch.float32)

        _rmsnorm_bwd[(n_programs,)](
            dy2d,
            x2d,
            w,
            rstd,
            dx,
            dw_partial,
            x2d.stride(0),
            N,
            D,
            rows_per_program,
            BLOCK_D=ctx.BLOCK_D,
            num_warps=4,
        )
        dw = dw_partial.sum(0).to(w.dtype)
        return dx.view(ctx.shape), dw, None


class TritonRMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return RMSNormFunction.apply(x, self.weight, self.eps)
