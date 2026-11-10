import torch
import triton
import triton.language as tl

@triton.jit
def _swiglu_fwd(a_ptr, b_ptr, h_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask).to(tl.float32)

    h = a * tl.sigmoid(a) * b

    tl.store(h_ptr + offsets, h.to(h_ptr.dtype.element_ty), mask=mask)

@triton.jit
def _swiglu_bwd(a_ptr, b_ptr, dh_ptr, da_ptr, db_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask).to(tl.float32)
    dh = tl.load(dh_ptr + offsets, mask=mask).to(tl.float32)

    db = dh * a * tl.sigmoid(a)
    da = dh * b * tl.sigmoid(a) * (1 + a * (1 - tl.sigmoid(a)))

    tl.store(da_ptr + offsets, da.to(da_ptr.type.element_ty), mask=mask)
    tl.store(db_ptr + offsets, db.to(db_ptr.type.element_ty), mask=mask)


class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        a, b = a.contiguous(), b.contiguous()
        h = torch.empty_like(a)
        n = a.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _swiglu_fwd[grid](a, b, h, n, BLOCK_SIZE=1024)
        ctx.save_for_backward(a, b)

        return h

    @staticmethod
    def backward(ctx, dh):
        a, b = ctx.saved_tensors
        dh = dh.contiguous()
        da = torch.empty_like(a)
        db = torch.empty_like(b)

        n = a.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

        _swiglu_bwd[grid](a, b, dh, da, db, n, BLOCK_SIZE=1024)
        return da, db


def swiglu(a, b):
    return SwiGLUFunction.apply(a,b)
