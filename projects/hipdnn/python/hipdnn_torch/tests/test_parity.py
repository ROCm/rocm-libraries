# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""GPU parity tests: each new override must (a) actually route through the catalog
engine and (b) match native PyTorch within the datatype's tolerance.

Every test here is marked ``gpu`` and is auto-skipped by ``conftest.py`` unless a
wired provider + HIP device is present (see ``../LOCAL_DEV.md``). The pattern for
each op is: install just that override, ``reset()`` its census, run the *patched*
functional, then assert the census shows ``aot>0`` (it routed, not fell back) and
the result matches a native run on the same inputs.
"""

import pytest

pytestmark = pytest.mark.gpu

# bf16 carries ~8 bits of mantissa; f16 ~11. These tolerances are loose enough for
# the accumulation-order differences between the WMMA kernels and native PyTorch but
# tight enough to catch a genuinely wrong result.
_TOL = {
    "f16": dict(rtol=2e-2, atol=2e-2),
    "bf16": dict(rtol=4e-2, atol=4e-2),
}


def _tok(torch, dtype):
    return {torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]


def _run_routed(op, fn):
    """Install ``op``, reset its census, run ``fn`` (which calls the patched
    functional), and return ``(result, aot, native)``. Always uninstalls."""
    import hipdnn_torch

    ov = hipdnn_torch.overrides()[op]
    hipdnn_torch.install([op])
    try:
        ov.reset()
        out = fn()
        aot, native = ov.totals()
        return out, aot, native
    finally:
        hipdnn_torch.uninstall([op])


def _assert_parity(torch, got, want, dtype):
    tol = _TOL[_tok(torch, dtype)]
    torch.testing.assert_close(got.float(), want.float(), **tol)


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


@pytest.mark.parametrize("dtype_name", ["f16", "bf16"])
def test_layernorm_parity(torch, dtype_name):
    import torch.nn.functional as F

    dtype = {"f16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    x = torch.randn(8, 256, device="cuda", dtype=dtype)
    w = torch.randn(256, device="cuda", dtype=dtype)
    b = torch.randn(256, device="cuda", dtype=dtype)
    want = F.layer_norm(x, (256,), w, b, 1e-5)
    got, aot, native = _run_routed(
        "layernorm", lambda: F.layer_norm(x, (256,), w, b, 1e-5)
    )
    assert aot > 0, f"layernorm did not route (aot={aot}, native={native})"
    _assert_parity(torch, got, want, dtype)


@pytest.mark.xfail(
    strict=False,
    reason="hipDNN coverage gap: no loaded engine (AOT/hipblaslt/MIOpen) serves "
    "weightless 3-D layer_norm at true rank -- backend returns "
    "get_ranked_engine_ids: No engine configurations available for the graph. "
    "Accepted as a current coverage limit; remove xfail when an engine lands it.",
)
@pytest.mark.parametrize("dtype_name", ["f16", "bf16"])
def test_layernorm_weightless_parity(torch, dtype_name):
    import torch.nn.functional as F

    dtype = {"f16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    x = torch.randn(4, 8, 128, device="cuda", dtype=dtype)
    want = F.layer_norm(x, (128,))
    got, aot, native = _run_routed("layernorm", lambda: F.layer_norm(x, (128,)))
    assert aot > 0, f"weightless layernorm did not route (aot={aot}, native={native})"
    _assert_parity(torch, got, want, dtype)


@pytest.mark.parametrize("dtype_name", ["f16", "bf16"])
def test_silu_parity(torch, dtype_name):
    import torch.nn.functional as F

    dtype = {"f16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    x = torch.randn(64, 512, device="cuda", dtype=dtype)
    want = F.silu(x)
    got, aot, native = _run_routed("silu", lambda: F.silu(x))
    assert aot > 0, f"silu did not route (aot={aot}, native={native})"
    _assert_parity(torch, got, want, dtype)


@pytest.mark.parametrize("dtype_name", ["f16", "bf16"])
def test_gelu_tanh_parity(torch, dtype_name):
    import torch.nn.functional as F

    dtype = {"f16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    x = torch.randn(64, 512, device="cuda", dtype=dtype)
    want = F.gelu(x, approximate="tanh")
    got, aot, native = _run_routed("gelu", lambda: F.gelu(x, approximate="tanh"))
    assert aot > 0, f"tanh-gelu did not route (aot={aot}, native={native})"
    _assert_parity(torch, got, want, dtype)


def test_gelu_erf_parity(torch):
    """Exact (erf) GELU is built and submitted like any other mode -- whether a
    loaded engine serves it or it falls back to native, the result must match native
    and must never raise. (Pure passthrough: no pre-judgement of the mode's support,
    so this asserts parity + no-exception, not a specific aot/native split.)"""
    import torch.nn.functional as F

    x = torch.randn(64, 512, device="cuda", dtype=torch.float16)
    want = F.gelu(x)  # approximate="none" (default)
    got, aot, native = _run_routed("gelu", lambda: F.gelu(x))
    assert aot + native > 0, "erf-gelu was not intercepted at all"
    _assert_parity(torch, got, want, torch.float16)


@pytest.mark.xfail(
    strict=False,
    reason="hipDNN coverage gap: no loaded engine (AOT/hipblaslt/MIOpen) serves "
    "N-D linear+bias at true rank ([B,T,K]@W^T+b) -- backend returns "
    "get_ranked_engine_ids: No engine configurations available for the graph. "
    "Accepted as a current coverage limit; remove xfail when an engine lands it.",
)
@pytest.mark.parametrize("dtype_name", ["f16", "bf16"])
def test_linear_nd_parity(torch, dtype_name):
    """N-D activation routed at true rank (no flatten): [B, T, K] @ W^T + b."""
    import torch.nn.functional as F

    dtype = {"f16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    x = torch.randn(4, 16, 128, device="cuda", dtype=dtype) * 0.1
    w = torch.randn(256, 128, device="cuda", dtype=dtype) * 0.1
    b = torch.randn(256, device="cuda", dtype=dtype) * 0.1
    want = F.linear(x, w, b)
    got, aot, native = _run_routed("linear", lambda: F.linear(x, w, b))
    assert aot > 0, f"N-D linear did not route (aot={aot}, native={native})"
    assert tuple(got.shape) == (4, 16, 256)
    _assert_parity(torch, got, want, dtype)


@pytest.mark.parametrize("dtype_name", ["f16", "bf16"])
def test_conv2d_grouped_parity(torch, dtype_name):
    """Grouped conv: groups inferred from the weight's own Cin/groups channel count,
    no group attribute, no pre-decline."""
    import torch.nn.functional as F

    dtype = {"f16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    x = torch.randn(1, 32, 28, 28, device="cuda", dtype=dtype) * 0.1
    w = torch.randn(64, 16, 3, 3, device="cuda", dtype=dtype) * 0.1  # groups=2
    want = F.conv2d(x, w, None, stride=1, padding=1, groups=2)
    got, aot, native = _run_routed(
        "conv2d", lambda: F.conv2d(x, w, None, stride=1, padding=1, groups=2)
    )
    assert aot > 0, f"grouped conv did not route (aot={aot}, native={native})"
    _assert_parity(torch, got, want, dtype)


@pytest.mark.parametrize("dtype_name", ["f16", "bf16"])
def test_conv2d_same_padding_parity(torch, dtype_name):
    """``padding='same'`` translated to pre/post padding (stride 1)."""
    import torch.nn.functional as F

    dtype = {"f16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    x = torch.randn(1, 32, 28, 28, device="cuda", dtype=dtype) * 0.1
    w = torch.randn(64, 32, 3, 3, device="cuda", dtype=dtype) * 0.1
    want = F.conv2d(x, w, None, stride=1, padding="same")
    got, aot, native = _run_routed(
        "conv2d", lambda: F.conv2d(x, w, None, stride=1, padding="same")
    )
    assert aot > 0, f"'same'-padding conv did not route (aot={aot}, native={native})"
    assert tuple(got.shape[-2:]) == (28, 28)  # 'same' preserves spatial size
    _assert_parity(torch, got, want, dtype)


@pytest.mark.xfail(
    strict=False,
    reason="hipDNN coverage gap: no loaded engine serves causal SDPA at this head "
    "config (H=8) -- the AOT fmha kernels are non-causal H64_HQ32_HK32 and MIOpen/"
    "hipblaslt offer no config, so the backend returns get_ranked_engine_ids: No "
    "engine configurations available. Accepted as a current coverage limit.",
)
@pytest.mark.parametrize("dtype_name", ["f16", "bf16"])
def test_sdpa_causal_parity(torch, dtype_name):
    """Causal SDPA on permuted (non-contiguous) Q/K/V, built from actual strides."""
    import torch.nn.functional as F

    dtype = {"f16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    # [B, S, H, D] -> transpose to [B, H, S, D] views (the usual permuted layout).
    q = torch.randn(2, 128, 8, 64, device="cuda", dtype=dtype).transpose(1, 2)
    k = torch.randn(2, 128, 8, 64, device="cuda", dtype=dtype).transpose(1, 2)
    v = torch.randn(2, 128, 8, 64, device="cuda", dtype=dtype).transpose(1, 2)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    got, aot, native = _run_routed(
        "sdpa",
        lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
    )
    assert aot > 0, f"causal SDPA did not route (aot={aot}, native={native})"
    _assert_parity(torch, got, want, dtype)


def test_sdpa_out_layout_match_q_returns_packed_bshd(torch, monkeypatch):
    """HIPDNN_TORCH_SDPA_OUT_LAYOUT=match_q allocates O in Q's packed BSHD layout.

    A kernel with no stride arguments reads AND writes packed BSHD, so it can serve a
    permuted Q/K/V but not a contiguous BHSD O -- that graph is mixed and gets declined.
    The opt-in makes O match Q. Allocation happens before engine selection, so the
    contract holds whichever engine ends up serving it.
    """
    import torch.nn.functional as F

    dtype = torch.float16
    q = torch.randn(2, 128, 8, 64, device="cuda", dtype=dtype).transpose(1, 2)
    k = torch.randn(2, 128, 8, 64, device="cuda", dtype=dtype).transpose(1, 2)
    v = torch.randn(2, 128, 8, 64, device="cuda", dtype=dtype).transpose(1, 2)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    monkeypatch.setenv("HIPDNN_TORCH_SDPA_OUT_LAYOUT", "match_q")
    got, aot, native = _run_routed(
        "sdpa",
        lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
    )
    assert aot > 0, f"SDPA did not route (aot={aot}, native={native})"
    assert got.stride() == q.stride(), (
        f"match_q must return O in Q's packed BSHD strides; got {got.stride()} "
        f"want {q.stride()}"
    )
    _assert_parity(torch, got, want, dtype)


def test_sdpa_out_layout_default_is_contiguous(torch, monkeypatch):
    """Unset (the default) keeps native's contract: a contiguous BHSD output."""
    import torch.nn.functional as F

    dtype = torch.float16
    q = torch.randn(2, 128, 8, 64, device="cuda", dtype=dtype).transpose(1, 2)
    k = torch.randn(2, 128, 8, 64, device="cuda", dtype=dtype).transpose(1, 2)
    v = torch.randn(2, 128, 8, 64, device="cuda", dtype=dtype).transpose(1, 2)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    monkeypatch.delenv("HIPDNN_TORCH_SDPA_OUT_LAYOUT", raising=False)
    got, aot, native = _run_routed(
        "sdpa",
        lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
    )
    assert aot > 0, f"SDPA did not route (aot={aot}, native={native})"
    assert got.is_contiguous(), "default must keep native's contiguous BHSD contract"
    _assert_parity(torch, got, want, dtype)


@pytest.mark.parametrize("dtype_name", ["f16", "bf16"])
def test_layernorm_multi_axis_parity(torch, dtype_name):
    """Multi-axis normalized_shape built at true rank (no 2-D collapse)."""
    import torch.nn.functional as F

    dtype = {"f16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    x = torch.randn(4, 8, 16, 32, device="cuda", dtype=dtype)
    w = torch.randn(16, 32, device="cuda", dtype=dtype)
    b = torch.randn(16, 32, device="cuda", dtype=dtype)
    want = F.layer_norm(x, (16, 32), w, b, 1e-5)
    got, aot, native = _run_routed(
        "layernorm", lambda: F.layer_norm(x, (16, 32), w, b, 1e-5)
    )
    assert aot > 0, f"multi-axis layernorm did not route (aot={aot}, native={native})"
    _assert_parity(torch, got, want, dtype)


@pytest.mark.xfail(
    strict=False,
    reason="hipDNN coverage gap: no loaded engine (AOT/hipblaslt/MIOpen) serves "
    "3-D weighted rms_norm at true rank -- backend returns get_ranked_engine_ids: "
    "No engine configurations available for the graph. Accepted as a current "
    "coverage limit; remove xfail when an engine lands it.",
)
@pytest.mark.parametrize("dtype_name", ["f16", "bf16"])
def test_rmsnorm_parity(torch, dtype_name):
    """RMSNorm at true rank (no flatten), weighted."""
    import torch.nn.functional as F

    dtype = {"f16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    x = torch.randn(4, 8, 256, device="cuda", dtype=dtype)
    w = torch.randn(256, device="cuda", dtype=dtype)
    want = F.rms_norm(x, (256,), w)
    got, aot, native = _run_routed("rmsnorm", lambda: F.rms_norm(x, (256,), w))
    assert aot > 0, f"rmsnorm did not route (aot={aot}, native={native})"
    _assert_parity(torch, got, want, dtype)


@pytest.mark.parametrize("dtype_name", ["f16", "bf16"])
def test_conv2d_parity(torch, dtype_name):
    import torch.nn.functional as F

    dtype = {"f16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    # Scale to an activation-like range (~0.1). Unit-variance randn makes the 3x3x32
    # reduction (288 terms) reach magnitude ~17, where near-zero cancellation outputs
    # carry an absolute f16 accumulation error that dwarfs a fixed atol -- not a
    # correctness signal. 0.1 keeps outputs O(1), matching the sample harnesses.
    x = torch.randn(1, 32, 28, 28, device="cuda", dtype=dtype) * 0.1
    w = torch.randn(64, 32, 3, 3, device="cuda", dtype=dtype) * 0.1
    b = torch.randn(64, device="cuda", dtype=dtype) * 0.1
    want = F.conv2d(x, w, b, stride=1, padding=1)
    got, aot, native = _run_routed(
        "conv2d", lambda: F.conv2d(x, w, b, stride=1, padding=1)
    )
    assert aot > 0, f"conv2d did not route (aot={aot}, native={native})"
    _assert_parity(torch, got, want, dtype)
