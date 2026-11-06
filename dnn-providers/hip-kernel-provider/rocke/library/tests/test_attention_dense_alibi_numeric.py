# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU numeric lane for ALiBi in the gfx942 / gfx950 dense kernels.

Drives ``run_attention_dense_torch`` of the arch the GPU reports and checks O and
LSE against ``rocke.numeric.sdpa_reference`` with the ALiBi term as an explicit
bias (hipDNN semantics: ``slope * (k - q)``).

    HIP_VISIBLE_DEVICES=0 python -m pytest -m gpu tests/test_attention_dense_alibi_numeric.py
"""

from __future__ import annotations

import numpy as np
import pytest
from rocke.numeric.sdpa_reference import alibi_bias, alibi_slopes, sdpa_reference

torch = pytest.importorskip("torch", reason="ROCm torch required")


def _gpu_arch():
    if not torch.cuda.is_available():
        return None
    arch = torch.cuda.get_device_properties(0).gcnArchName.lower()
    for name in ("gfx942", "gfx950"):
        if name in arch:
            return name
    return None


_ARCH = _gpu_arch()

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(_ARCH is None, reason="needs a gfx942 or gfx950 GPU"),
]

_TORCH_DT = {"fp16": "float16", "bf16": "bfloat16"}


def _kernel_module():
    if _ARCH == "gfx942":
        from kernels.gfx942 import attention_dense as mod

        return mod, mod.Gfx942AttentionDenseSpec
    from kernels.gfx950 import attention_dense as mod

    return mod, mod.Gfx950AttentionDenseSpec


def _tolerance(dtype):
    return 2e-2 if dtype == "fp16" else 4e-2


def _run_case(
    *,
    dtype="fp16",
    head_size=128,
    heads=(8, 8),
    batch=1,
    sq=512,
    skv=512,
    causal=True,
    bottom_right=False,
    window=0,
    right_bound=0,
    external=False,
    slopes=None,
    lse=True,
    seed=0,
    **spec_over,
):
    """Run one case; return (out, lse, ref_out, ref_lse) as float64 numpy, with
    out [B, Sq, Hq, D] and lse [B, Hq, Sq]."""
    mod, spec_cls = _kernel_module()
    hq, hkv = heads
    if right_bound:
        spec_over["right_bound"] = right_bound
    if lse:
        spec_over["emit_lse"] = True
    spec = spec_cls(
        batch=batch,
        seqlen_q=sq,
        seqlen_kv=skv,
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_size=head_size,
        dtype=dtype,
        causal=causal,
        causal_bottom_right=bottom_right,
        sliding_window=window,
        use_alibi=True,
        **spec_over,
    )
    ok, why = mod.supports_attention_dense(spec, arch=_ARCH)
    if not ok:
        pytest.skip(f"{_ARCH} does not support this case: {why}")
    tdt = getattr(torch, _TORCH_DT[dtype])
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(batch, sq, hq, head_size, device="cuda", dtype=tdt, generator=gen)
    k = torch.randn(batch, skv, hkv, head_size, device="cuda", dtype=tdt, generator=gen)
    v = torch.randn(batch, skv, hkv, head_size, device="cuda", dtype=tdt, generator=gen)
    out = torch.empty_like(q)
    lse_t = (
        torch.full((batch, hq, sq, 1), float("nan"), device="cuda", dtype=torch.float32)
        if lse
        else None
    )
    if external:
        slopes_np = np.asarray(slopes, dtype=np.float64)
        slopes_t = torch.tensor(slopes_np, device="cuda", dtype=torch.float32)
    else:
        slopes_np = alibi_slopes(hq)
        slopes_t = None
    scale = head_size**-0.5
    mod.run_attention_dense_torch(
        spec=spec,
        q=q,
        k=k,
        v=v,
        out=out,
        scale=scale,
        arch=_ARCH,
        lse=lse_t,
        alibi_slopes=slopes_t,
    )
    torch.cuda.synchronize()

    def host(t):
        return t.float().cpu().numpy().astype(np.float64)

    ref_o, ref_lse = sdpa_reference(
        host(q).transpose(0, 2, 1, 3),
        host(k).transpose(0, 2, 1, 3),
        host(v).transpose(0, 2, 1, 3),
        scale=scale,
        bias=alibi_bias(np.float32(slopes_np), sq, skv),
        left_bound=window - 1 if window else None,
        right_bound=right_bound if causal else None,
        top_left=not bottom_right,
    )
    lse_np = host(lse_t)[..., 0] if lse else None
    return host(out), lse_np, ref_o.transpose(0, 2, 1, 3), ref_lse


def _check(case, dtype, *, lse=True):
    out, lse_v, ref_o, ref_lse = case
    tol = _tolerance(dtype)
    assert np.abs(out - ref_o).max() < tol
    if lse:
        assert np.abs(lse_v - ref_lse).max() < tol


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("heads", [(8, 8), (12, 4)], ids=["mha", "gqa"])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "full"])
def test_builtin_slopes(dtype, head_size, heads, causal):
    _check(
        _run_case(dtype=dtype, head_size=head_size, heads=heads, causal=causal),
        dtype,
    )


@pytest.mark.parametrize("hq", [6, 40])
def test_non_power_of_two_heads(hq):
    _check(_run_case(heads=(hq, hq // 2), causal=False), "fp16")


def test_external_slopes():
    rng = np.random.default_rng(3)
    slopes = rng.uniform(0.01, 0.5, size=8)
    _check(_run_case(external=True, slopes=slopes), "fp16")


def test_external_hipdnn_slopes_match_builtin():
    builtin = _run_case(heads=(12, 4), causal=False, seed=5)
    caller = _run_case(
        heads=(12, 4), causal=False, external=True, slopes=alibi_slopes(12), seed=5
    )
    assert np.abs(builtin[0] - caller[0]).max() < 2e-3
    assert np.abs(builtin[1] - caller[1]).max() < 1e-4


def test_top_left_cross_length():
    _check(_run_case(sq=512, skv=1024, heads=(8, 2)), "bf16")


def test_bottom_right():
    # No LSE: off the top-left diagonal it is undefined (see _attention_alibi).
    case = _run_case(sq=256, skv=512, bottom_right=True, heads=(12, 4), lse=False)
    _check(case, "bf16", lse=False)


def test_batch():
    _check(_run_case(batch=2, heads=(8, 2), causal=False), "fp16")


def test_sliding_window():
    _check(_run_case(sq=1024, skv=1024, window=128), "fp16")


def test_band_right_bound():
    if _ARCH == "gfx942":
        pytest.skip("gfx942 dense has no right_bound")
    _check(_run_case(sq=512, skv=512, window=128, right_bound=64), "fp16")


def test_kv_segments():
    _check(_run_case(sq=256, skv=2048, lse=False, kv_segments=4), "fp16", lse=False)


def test_long_noncausal_precision():
    # The steepest slope at 4 heads is 2**-2, so the far key's term is about -2048.
    _check(_run_case(sq=256, skv=8192, heads=(4, 4), causal=False), "fp16")


def test_persistent():
    if _ARCH == "gfx950":
        pytest.skip("gfx950 refuses ALiBi on the persistent grid")
    _check(_run_case(persistent=True, heads=(8, 2)), "bf16")
