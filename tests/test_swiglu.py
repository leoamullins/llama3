import pytest
import torch
import torch.nn.functional as F

from llama3.kernels.swiglu import swiglu

CASES = [
    # shape,            dtype,          tolerance
    ((1000,),           torch.float32,  1e-5),   # not a multiple of BLOCK_SIZE: tests the mask
    ((4, 128, 2688),    torch.float32,  1e-5),   # tight check of the math
    ((8, 1024, 2688),   torch.bfloat16, 2e-2),   # your real training shape and dtype
    ((8, 1024, 2688),   torch.float16,  1e-2),
]

@pytest.mark.parametrize("shape,dtype,tol", CASES)
def test_swiglu_matches_pytorch(shape, dtype, tol):
    torch.manual_seed(0)
    a = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    b = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()

    # forward
    h = swiglu(a, b)
    h_ref = F.silu(a_ref) * b_ref
    torch.testing.assert_close(h, h_ref, atol=tol, rtol=tol)

    # backward, with a random upstream gradient
    dh = torch.randn_like(h)
    h.backward(dh)
    h_ref.backward(dh)
    torch.testing.assert_close(a.grad, a_ref.grad, atol=tol, rtol=tol)
    torch.testing.assert_close(b.grad, b_ref.grad, atol=tol, rtol=tol)

def test_non_contiguous_grad():
    # the gradient from w3's backward isn't guaranteed contiguous
    a = torch.randn(64, 256, device="cuda", requires_grad=True)
    b = torch.randn(64, 256, device="cuda", requires_grad=True)
    dh = torch.randn(256, 64, device="cuda").t()       # transposed, so not contiguous
    swiglu(a, b).backward(dh)
    a_ref = a.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    (F.silu(a_ref) * b_ref).backward(dh)
    torch.testing.assert_close(a.grad, a_ref.grad)
    torch.testing.assert_close(b.grad, b_ref.grad)
