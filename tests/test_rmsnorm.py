import pytest
import torch

from llama3.kernels.rmsnorm import rms_norm

CASES = [
    # shape,           dtype,          tolerance
    ((37, 1000),       torch.float32,  1e-4),   # D not a power of two: tests the column mask
    ((4, 128, 768),    torch.float32,  1e-4),   # tight check of the math
    ((8, 1024, 768),   torch.bfloat16, 2e-2),   # real training shape and dtype
    ((3, 1),           torch.float32,  1e-4),   # fewer rows than programs in the backward
]


@pytest.mark.parametrize("shape,dtype,tol", CASES)
def test_rmsnorm_matches_pytorch(shape, dtype, tol):
    torch.manual_seed(0)
    D = shape[-1]
    ref = torch.nn.RMSNorm(D, eps=1e-5, device="cuda", dtype=dtype)
    with torch.no_grad():
        ref.weight.copy_(torch.randn(D, device="cuda"))  # non-trivial weight, not all ones
    w = ref.weight.detach().clone().requires_grad_()

    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()

    # forward
    y = rms_norm(x, w, 1e-5)
    y_ref = ref(x_ref)
    torch.testing.assert_close(y, y_ref, atol=tol, rtol=tol)

    # backward, with a random upstream gradient
    dy = torch.randn_like(y)
    y.backward(dy)
    y_ref.backward(dy)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=tol, rtol=tol)
    # dw sums over every row, so compare relative to its scale
    dw_err = (w.grad.float() - ref.weight.grad.float()).abs().max()
    assert dw_err <= tol * ref.weight.grad.float().abs().max()


def test_non_contiguous_grad():
    # the gradient arriving from the next layer is not guaranteed contiguous
    x = torch.randn(64, 256, device="cuda", requires_grad=True)
    w = torch.randn(256, device="cuda", requires_grad=True)
    dy = torch.randn(256, 64, device="cuda").t()  # transposed, so not contiguous
    rms_norm(x, w, 1e-5).backward(dy)

    ref = torch.nn.RMSNorm(256, eps=1e-5, device="cuda")
    with torch.no_grad():
        ref.weight.copy_(w)
    x_ref = x.detach().clone().requires_grad_()
    ref(x_ref).backward(dy)
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(w.grad, ref.weight.grad)
