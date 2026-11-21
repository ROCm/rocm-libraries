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


def _quantize_fp8(t):
    """Per-tensor fp8 e4m3 storage of ``t``: (fp8 tensor, scale, dequantized t)."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = (t.float().abs().max().clamp(min=1e-8) / fp8_max).item()
    t8 = (t.float() / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return t8, scale, (t8.float() * scale).to(t.dtype)


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
    fp8_kv=False,
    sinks=False,
    **spec_over,
):
    """Run one case; return (out, lse, ref_out, ref_lse) as float64 numpy, with
    out [B, Sq, Hq, D] and lse [B, Hq, Sq]. ``fp8_kv`` stores K/V as fp8 with
    per-tensor scales; the reference attends over the same dequantized K/V, so
    only kernel error is measured."""
    mod, spec_cls = _kernel_module()
    hq, hkv = heads
    if fp8_kv:
        spec_over["kv_storage_dtype"] = "fp8e4m3"
    if right_bound:
        spec_over["right_bound"] = right_bound
    if lse:
        spec_over["emit_lse"] = True
    if sinks:
        spec_over["use_sinks"] = True
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
    k_run, v_run, fp8_kwargs = k, v, {}
    if fp8_kv:
        k_run, k_scale, k = _quantize_fp8(k)
        v_run, v_scale, v = _quantize_fp8(v)
        fp8_kwargs = {"k_scale": k_scale, "v_scale": v_scale}
    sink_t = torch.randn(hq, device="cuda", dtype=tdt, generator=gen) if sinks else None
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
        k=k_run,
        v=v_run,
        out=out,
        scale=scale,
        arch=_ARCH,
        lse=lse_t,
        alibi_slopes=slopes_t,
        sinks=sink_t,
        **fp8_kwargs,
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
    if sinks:
        ref_o, ref_lse = _with_sinks(ref_o, ref_lse, host(sink_t))
    lse_np = host(lse_t)[..., 0] if lse else None
    return host(out), lse_np, ref_o.transpose(0, 2, 1, 3), ref_lse


def _with_sinks(ref_o, ref_lse, sinks):
    """Add a per-head sink logit (score units, no positional term: SEM-022) to
    the softmax denominator of ``(O [B, H, Sq, D], LSE [B, H, Sq])``."""
    s = sinks[None, :, None]
    lse = np.logaddexp(ref_lse, s)
    keep = np.where(np.isneginf(ref_lse), 0.0, np.exp(ref_lse - lse))
    return np.nan_to_num(ref_o) * keep[..., None], lse


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


@pytest.mark.parametrize(
    "sq,skv", [(512, 1024), (1024, 512)], ids=["sq_lt_skv", "sq_gt_skv"]
)
def test_top_left_cross_length(sq, skv):
    # sq > skv: the rows past the last key see every key; the anchor is the last one.
    _check(_run_case(sq=sq, skv=skv, heads=(8, 2)), "bf16")


def test_bottom_right():
    # No LSE: off the top-left diagonal it is undefined (see _attention_alibi).
    case = _run_case(sq=256, skv=512, bottom_right=True, heads=(12, 4), lse=False)
    _check(case, "bf16", lse=False)


def test_batch():
    _check(_run_case(batch=2, heads=(8, 2), causal=False), "fp16")


def test_sliding_window():
    _check(_run_case(sq=1024, skv=1024, window=128), "fp16")


@pytest.mark.parametrize("persistent", [False, True], ids=["grid", "persistent"])
@pytest.mark.parametrize("lse", [False, True], ids=["no_lse", "lse"])
@pytest.mark.parametrize("window", [128, 64])
def test_sliding_window_keyless_rows(window, lse, persistent):
    # Sq > Skv + W - 1: rows from Skv + W - 1 on see no key and must be exactly 0
    # (LSE -inf), with the ALiBi term on the rows before them. W 64 puts the first
    # keyless row mid-block.
    sq, skv = 1024, 256
    out, lse_v, ref_o, ref_lse = _run_case(
        sq=sq, skv=skv, window=window, heads=(8, 2), lse=lse, persistent=persistent
    )
    first = skv + window - 1
    assert not np.any(out[:, first:]), f"rows >= {first} see no key but are not 0"
    tol = _tolerance("fp16")
    assert np.abs(out[:, :first] - ref_o[:, :first]).max() < tol
    if lse:
        assert np.isneginf(lse_v[:, :, first:]).all()
        assert np.abs(lse_v[:, :, :first] - ref_lse[:, :, :first]).max() < tol


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
    _check(_run_case(persistent=True, heads=(8, 2)), "bf16")


@pytest.mark.parametrize("design", ["seed_tile", "fma_tile", "seed_split", "fma_split"])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "full"])
@pytest.mark.parametrize("persistent", [False, True], ids=["grid", "persistent"])
def test_designs(design, head_size, causal, persistent):
    # Every design on every body, not only each family's default.
    case = _run_case(
        dtype="bf16",
        head_size=head_size,
        heads=(8, 2),
        causal=causal,
        persistent=persistent,
        alibi_design=design,
    )
    _check(case, "bf16")


@pytest.mark.parametrize("design", ["seed_tile", "fma_tile", "seed_split", "fma_split"])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("persistent", [False, True], ids=["grid", "persistent"])
def test_designs_window_keyless(design, head_size, persistent):
    # A window unfolds the scale on the gfx950 persistent body, so the split designs
    # take other paths there; Sq > Skv + W - 1 adds rows without keys.
    sq, skv, window = 1024, 256, 128
    out, lse_v, ref_o, ref_lse = _run_case(
        dtype="bf16",
        head_size=head_size,
        heads=(8, 2),
        sq=sq,
        skv=skv,
        window=window,
        persistent=persistent,
        alibi_design=design,
    )
    first = skv + window - 1
    assert not np.any(out[:, first:])
    assert np.isneginf(lse_v[:, :, first:]).all()
    tol = _tolerance("bf16")
    assert np.abs(out[:, :first] - ref_o[:, :first]).max() < tol
    assert np.abs(lse_v[:, :, :first] - ref_lse[:, :, :first]).max() < tol


@pytest.mark.parametrize(
    "over",
    [
        {},
        dict(window=128),
        dict(persistent=True),
        dict(sq=1024, skv=512),
        dict(head_size=64),
    ],
    ids=["causal", "window", "persistent", "sq_gt_skv", "d64"],
)
def test_sinks(over):
    # Top-left causal anchors each row at its query, so the sink competes with
    # the exact slope * (k - q) scores (other masks are refused).
    kw = dict(dtype="bf16", heads=(8, 2), sinks=True)
    kw.update(over)
    _check(_run_case(**kw), "bf16")


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "full"])
def test_fp8_kv(dtype, causal):
    # fp8 K/V dequantizes in the loader; the ALiBi bias still rides the QK MFMA.
    if _ARCH != "gfx950":
        pytest.skip("fp8 KV dense is gfx950-only")
    _check(_run_case(dtype=dtype, heads=(12, 4), causal=causal, fp8_kv=True), dtype)
