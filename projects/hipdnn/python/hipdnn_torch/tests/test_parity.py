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


def test_gelu_erf_falls_back(torch):
    """Exact (erf) GELU has no catalog builder -> must fall back to native, not error,
    and be counted as a native fallback with the documented reason."""
    import torch.nn.functional as F

    x = torch.randn(64, 512, device="cuda", dtype=torch.float16)
    want = F.gelu(x)  # approximate="none" (default)
    got, aot, native = _run_routed("gelu", lambda: F.gelu(x))
    assert (
        aot == 0 and native > 0
    ), f"erf-gelu should fall back (aot={aot}, native={native})"
    _assert_parity(torch, got, want, torch.float16)


@pytest.mark.parametrize("dtype_name", ["f16", "bf16"])
def test_conv2d_parity(torch, dtype_name):
    import torch.nn.functional as F

    dtype = {"f16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    x = torch.randn(1, 32, 28, 28, device="cuda", dtype=dtype)
    w = torch.randn(64, 32, 3, 3, device="cuda", dtype=dtype)
    b = torch.randn(64, device="cuda", dtype=dtype)
    want = F.conv2d(x, w, b, stride=1, padding=1)
    got, aot, native = _run_routed(
        "conv2d", lambda: F.conv2d(x, w, b, stride=1, padding=1)
    )
    assert aot > 0, f"conv2d did not route (aot={aot}, native={native})"
    _assert_parity(torch, got, want, dtype)
