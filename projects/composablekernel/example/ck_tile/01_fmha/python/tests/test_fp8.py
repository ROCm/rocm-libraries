# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest
import torch

from ck_tile_fmha import fmha_fp8_func, fmha_func


def make_fp8_input(shape, qscale, block_size=128, is_value=False):
    """Quantized input, FP32 descale, and dequantized reference with varying scales."""
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    dtype = torch.float8_e4m3fnuz if arch.startswith("gfx94") else torch.float8_e4m3fn
    b, s, h, _ = shape
    x = torch.randn(shape, device="cuda") * 0.5
    if qscale == "no_scale":
        quant = x.to(dtype)
        return quant, None, quant.float()
    if qscale == "pertensor":
        descale = torch.rand(1, device="cuda") * 0.03 + 0.01
        broadcast = descale
    elif qscale == "perhead":
        descale = (torch.rand(h * 2, b, device="cuda") * 0.03 + 0.01).t()[:, ::2]
        broadcast = descale[:, None, :, None]
    else:
        blocks = (s + block_size - 1) // block_size
        descale = (torch.rand(h, b, blocks, device="cuda") * 0.03 + 0.01).transpose(0, 1)
        if is_value and arch.startswith("gfx1250"):
            descale = torch.exp2(
                torch.randint(-6, -3, (h, b, blocks), device="cuda").float()
            ).transpose(0, 1)
        broadcast = descale.repeat_interleave(block_size, -1)[..., :s].transpose(1, 2)[..., None]
    # Different physical values and quantization grids across heads, batches and blocks.
    x = x * (broadcast / 0.02)
    quant = (x / broadcast).to(dtype)
    return quant, descale, quant.float() * broadcast


def check_fp8_output(out, q, k, v, scale, causal=False, window=(-1, -1)):
    # Compare the same dequantized FP8 data, rounded to BF16 for the baseline kernel.
    q, k, v = (x.to(torch.bfloat16) for x in (q, k, v))
    kwargs = dict(softmax_scale=scale, causal=causal, window_size=window)
    ref, lse, _ = fmha_func(q, k, v, **kwargs, return_attn_probs=True)
    magnitude = fmha_func(q, k, v.abs(), **kwargs).float()
    ref = ref.float()
    valid = torch.isfinite(lse)
    # Use the C++ example's magnitude and gain checks with a BF16 baseline.
    # Attention(Q, K, abs(V)) estimates P @ abs(V) without materializing P.
    u_p, u_o = 2**-3, 2**-7
    error = out.float() - ref
    bound = u_p * magnitude + (u_o * ref.abs()).clamp_min(u_o * torch.finfo(torch.bfloat16).tiny)
    assert torch.all(error.abs() <= bound), (
        f"FP8 rounding bound exceeded: {(error.abs() - bound).max().item()}"
    )
    # Also detect a systematic gain per (batch, head), which the bound alone can miss.
    denominator = ref.square().sum(dim=(1, 3))
    alpha = (error * ref).sum(dim=(1, 3)) / denominator.clamp_min(1e-30)
    rows = valid.sum(dim=-1).clamp_min(1)
    alpha_tol = 2 * (u_p**2 / (12 * rows) + u_o**2 / (12 * rows * v.shape[-1])) ** 0.5
    assert torch.all(alpha.abs() <= alpha_tol), (
        f"FP8 gain error: {alpha.abs().max().item()} > {alpha_tol}"
    )
    return lse, valid


@pytest.mark.parametrize("qscale", ["no_scale", "pertensor", "perhead", "blockscale"])
@pytest.mark.parametrize("d", [32, 59, 64, 111, 128])
@pytest.mark.parametrize("mask", ["none", "causal", "local"])
@pytest.mark.parametrize("sq,sk,h,hk", [(97, 203, 4, 4), (1, 257, 6, 2), (193, 113, 6, 1)])
@pytest.mark.parametrize("return_lse", [False, True])
def test_fp8_fixedlen(qscale, d, mask, sq, sk, h, hk, return_lse):
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1250"):
        pytest.skip("This FP8 mode/head-dimension matrix requires gfx1250")
    torch.manual_seed(0)
    q, qs, qref = make_fp8_input((2, sq, h, d), qscale)
    k, ks, kref = make_fp8_input((2, sk, hk, d), qscale)
    v, vs, vref = make_fp8_input((2, sk, hk, d), qscale, is_value=True)
    causal = mask == "causal"
    window = (37, 11) if mask == "local" else (-1, -1)
    scale = 0.7 / d**0.5
    actual = fmha_fp8_func(
        q,
        k,
        v,
        qs,
        ks,
        vs,
        qscale=qscale,
        causal=causal,
        window_size=window,
        softmax_scale=scale,
        return_lse=return_lse,
    )
    out = actual[0] if return_lse else actual
    lse_ref, valid = check_fp8_output(out, qref, kref, vref, scale, causal, window)
    assert out.dtype == torch.bfloat16
    assert out.shape == q.shape
    if return_lse:
        lse = actual[1]
        assert lse.dtype == torch.float32
        assert lse.shape == (2, h, sq)
        torch.testing.assert_close(lse[valid], lse_ref[valid], atol=0.01, rtol=0.001)
        assert torch.isneginf(lse[~valid]).all()


@pytest.mark.parametrize("block_size_q,block_size_kv", [(128, 32), (128, 64), (256, 128)])
def test_fp8_block_sizes(block_size_q, block_size_kv):
    qscale = "blockscale"

    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1250"):
        pytest.skip("Fine KV block scales require gfx1250")
    torch.manual_seed(1)
    q, qs, qr = make_fp8_input((2, 273, 6, 128), qscale, block_size_q)
    k, ks, kr = make_fp8_input((2, 397, 2, 128), qscale, block_size_kv)
    v, vs, vr = make_fp8_input((2, 397, 2, 128), qscale, block_size_kv, is_value=True)
    out = fmha_fp8_func(
        q, k, v, qs, ks, vs, qscale=qscale, block_size_q=block_size_q, block_size_kv=block_size_kv
    )
    check_fp8_output(out, qr, kr, vr, 128**-0.5)


@pytest.mark.parametrize("qscale", ["no_scale", "pertensor"])
@pytest.mark.parametrize("d", [160, 192, 256])
def test_fp8_large_head(qscale, d):
    torch.manual_seed(2)
    q, qs, qr = make_fp8_input((2, 97, 4, d), qscale)
    k, ks, kr = make_fp8_input((2, 203, 2, d), qscale)
    v, vs, vr = make_fp8_input((2, 203, 2, d), qscale)
    if torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1250"):
        with pytest.raises(RuntimeError, match="No FMHA forward kernel"):
            fmha_fp8_func(q, k, v, qs, ks, vs, qscale=qscale)
    else:
        out = fmha_fp8_func(q, k, v, qs, ks, vs, qscale=qscale)
        check_fp8_output(out, qr, kr, vr, d**-0.5)


@pytest.mark.parametrize("qscale", ["no_scale", "pertensor", "perhead", "blockscale"])
@pytest.mark.parametrize("return_lse", [False, True])
def test_fp8_operator_and_compile(qscale, return_lse):
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1250"):
        pytest.skip("All FP8 modes require gfx1250")
    q, qs, _ = make_fp8_input((2, 32, 4, 64), qscale)
    k, ks, _ = make_fp8_input((2, 67, 2, 64), qscale)
    v, vs, _ = make_fp8_input((2, 67, 2, 64), qscale, is_value=True)
    args = (
        q,
        k,
        v,
        0.0,
        0.125,
        False,
        -1,
        -1,
        0.0,
        None,
        False,
        None,
        qs,
        ks,
        vs,
        qscale,
        return_lse,
    )
    torch.library.opcheck(
        torch.ops.ck_tile_fmha._fmha_forward,
        args,
        test_utils=("test_faketensor",),
    )
    # PyTorch's schema checker calls allclose on FP8, which ROCm does not implement.
    # Check input mutation explicitly using exact FP32 representations instead.
    inputs = [x for x in (q, k, v, qs, ks, vs) if x is not None]
    before = [x.float().clone() for x in inputs]
    compiled = torch.compile(fmha_fp8_func, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(
        compiled(q, k, v, qs, ks, vs, qscale=qscale, return_lse=return_lse),
        fmha_fp8_func(q, k, v, qs, ks, vs, qscale=qscale, return_lse=return_lse),
    )
    for x, saved in zip(inputs, before):
        assert torch.equal(x.float(), saved)


@pytest.mark.parametrize(
    "invalid",
    [
        "missing",
        "dtype",
        "device",
        "shape",
        "grad",
        "mode",
        "no_scale",
        "block_size",
        "block_shape",
        "block_stride",
        "fp8_dtype",
        "qkv_dtype",
    ],
)
def test_fp8_validation(invalid):
    q, qs, _ = make_fp8_input((2, 32, 4, 64), "pertensor")
    k, ks, _ = make_fp8_input((2, 97, 2, 64), "pertensor")
    v, vs, _ = make_fp8_input((2, 97, 2, 64), "pertensor")
    kwargs = {}
    if invalid == "missing":
        qs, message = None, "required"
    elif invalid == "dtype":
        qs, message = qs.half(), "float32"
    elif invalid == "device":
        qs, message = qs.cpu(), "same device"
    elif invalid == "shape":
        qs, message = qs.expand(2), "one scale"
    elif invalid == "grad":
        qs, message = qs.requires_grad_(), "forward only"
    elif invalid == "mode":
        kwargs["qscale"], message = "unknown", "Unsupported FP8 qscale"
    elif invalid == "no_scale":
        kwargs["qscale"], message = "no_scale", "does not accept descales"
    elif invalid == "block_size":
        kwargs.update(qscale="blockscale", block_size_kv=0)
        message = "block sizes must be positive"
    elif invalid == "block_shape":
        kwargs["qscale"], message = "blockscale", "must have shape"
    elif invalid == "block_stride":
        kwargs.update(qscale="blockscale", block_size_q=16)
        qs = torch.ones(2, 4, 4, device="cuda")[..., ::2]
        message = "contiguous block dimension"
    elif invalid == "fp8_dtype":
        other = torch.float8_e4m3fnuz if q.dtype == torch.float8_e4m3fn else torch.float8_e4m3fn
        q, k, v = (x.float().to(other) for x in (q, k, v))
        message = "dtype does not match the GPU"
    else:
        k, message = k.half(), "same dtype"
    with pytest.raises((RuntimeError, ValueError), match=message):
        fmha_fp8_func(q, k, v, qs, ks, vs, **kwargs)
