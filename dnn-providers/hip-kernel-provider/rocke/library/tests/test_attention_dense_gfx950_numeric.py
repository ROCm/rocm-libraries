# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU numeric lane for the gfx950 dense flash-attn kernel.

Drives the PUBLIC entry point ``run_attention_dense_torch`` end-to-end on a real
gfx950 GPU and checks max_abs against an fp32 reference, including sink support.

Every test is marked ``gpu`` and gated with a device skipif, so it is a graceful
skip on a CPU CI box and only executes on a gfx950. Select
it with ``run_all.py --gpu`` (or ``pytest -m gpu``); the default CPU lane excludes
it via ``-m "not gpu"``. Run standalone:

    HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_attention_dense_gfx950_numeric.py
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import replace

import pytest

from kernels.common.attention_dense_spec import DENSE_TILE_GEOMETRIES
from kernels.gfx950.attention_dense import (
    Gfx950AttentionDenseSpec,
    run_attention_dense_torch,
)


torch = pytest.importorskip("torch", reason="ROCm torch required")


def _gpu_ready():
    """True only on a gfx950 box with ROCm torch. Gate on ``gcnArchName`` (the ISA
    target), NOT the marketing name."""
    if not torch.cuda.is_available():
        return False
    arch = torch.cuda.get_device_properties(0).gcnArchName.lower()
    return "gfx950" in arch


requires_gfx950_gpu = pytest.mark.skipif(
    not _gpu_ready(), reason="needs a gfx950 GPU with ROCm torch"
)

_TORCH_DT = {"fp16": "float16", "bf16": "bfloat16"}

# (dtype, head_size, num_query_heads, num_kv_heads, persistent) -- a compact cohort
# spanning both dtypes, D64/D128, GQA + MHA, and both grid variants. Sq is fixed at
# 512 (a 256 multiple so the persistent grid-stride has >1 q-block of work).
_COHORT = [
    ("fp16", 128, 16, 4, False),  # flagship default
    ("fp16", 128, 16, 4, True),  # flagship persistent
    ("bf16", 128, 16, 4, True),  # bf16 D128 persistent
    ("bf16", 128, 16, 4, False),  # bf16 D128 default
    ("fp16", 64, 16, 16, False),  # D64 MHA default
    ("fp16", 128, 16, 16, True),  # D128 MHA persistent (auto -> qb_major)
    ("bf16", 64, 16, 4, True),  # D64 bf16 persistent
]


def _spec(
    dtype,
    d,
    hq,
    hkv,
    persistent,
    *,
    batch=1,
    sq=512,
    use_sinks=False,
    sliding_window=0,
    causal=True,
):
    """The SHIPPED gfx950 dense spec for a cohort row, built through the dispatch
    factory rather than hand-rolled.

    Deriving the spec from the factory means a future gfx950 tuning change is picked
    up here with no edit. The candidate is pinned by ``spec_id``: the cohort
    asserts BOTH bodies at one fixed Sq, and the persistent row runs wide
    DMA on aligned causal D128 without sinks/SWA (``TestWideDmaFeatures``
    covers the other masks).
    """
    # Imported lazily: keeps module import (and hence CPU collection of this
    # gpu-marked file) independent of the dispatch package.
    from dispatch.attention import AttentionRequest, attention_tuning_spec

    wide = d == 128 and causal and not use_sinks and not sliding_window
    if not persistent:
        spec_id = "gfx950_dense_grid"
    elif wide:
        spec_id = "gfx950_dense_persist_widedma"
    else:
        spec_id = "gfx950_dense_persist"
    return attention_tuning_spec(
        AttentionRequest(
            batch=batch,
            nhead_q=hq,
            nhead_k=hkv,
            seqlen_q=sq,
            seqlen_k=sq,
            hdim_q=d,
            hdim_v=d,
            arch="gfx950",
            mask_type=1 if causal else 0,
            dtype=dtype,
            use_sinks=use_sinks,
            sliding_window=sliding_window,
        ),
        spec_id,
    ).kernel_spec


def _launcher_for(spec):
    """The cached ``KernelLauncher`` for ``spec``, or None if none is compiled.

    ``run_attention_dense_torch`` owns ``_DENSE_LAUNCHER_CACHE`` internally and
    exposes no accessor, so this reads the module-level dict through the very
    key function the production path uses -- a test-local reimplementation of
    the key would assert against itself rather than against the shipped one.
    """
    from kernels.common.attention_dense_spec import attention_dense_cache_key
    from kernels.gfx950.attention_dense import _DENSE_LAUNCHER_CACHE

    return _DENSE_LAUNCHER_CACHE.get(attention_dense_cache_key(spec, arch="gfx950"))


def _tolerance(dtype):
    """Numerical tolerance for max_abs error (matches gfx942).

    fp16 has tighter tolerance (2e-2) than bf16 (4e-2) due to better precision
    in the mantissa (10 bits vs 7 bits).
    """
    return 2e-2 if dtype == "fp16" else 4e-2


def _standard_reference(q, k, v, scale, sliding_window=0):
    """Standard causal attention reference using PyTorch SDPA.

    Args:
        q: [B, S, Hq, D] query tensor
        k: [B, S, Hkv, D] key tensor
        v: [B, S, Hkv, D] value tensor
        scale: softmax scale (1/sqrt(D))

    Returns:
        ref: [B, S, Hq, D] reference output in fp32
    """
    import torch
    import torch.nn.functional as F

    Hq = q.shape[2]
    Hkv = k.shape[2]
    rep = Hq // Hkv

    # Expand to [B, Hq, S, D] and expand GQA
    qf = q.transpose(1, 2).float()
    kf = k.transpose(1, 2).float().repeat_interleave(rep, dim=1)
    vf = v.transpose(1, 2).float().repeat_interleave(rep, dim=1)

    # PyTorch SDPA with causal masking
    if sliding_window:
        sq, sk = q.shape[1], k.shape[1]
        qi = torch.arange(sq, device=q.device).view(-1, 1)
        ki = torch.arange(sk, device=q.device).view(1, -1)
        allowed = (ki <= qi) & (ki > qi - sliding_window)
        ref = F.scaled_dot_product_attention(
            qf, kf, vf, attn_mask=allowed, scale=scale
        ).transpose(1, 2)
    else:
        ref = F.scaled_dot_product_attention(
            qf, kf, vf, is_causal=True, scale=scale
        ).transpose(
            1, 2
        )  # -> [B, S, Hq, D]

    return ref


class TestDenseNumeric:
    """Standard (non-sink) dense attention numerical validation."""

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype,d,hq,hkv,persistent", _COHORT)
    def test_dense_numeric_vs_fp32_sdpa(self, dtype, d, hq, hkv, persistent):
        """Verify standard dense kernel against PyTorch SDPA."""
        import torch

        tol = _tolerance(dtype)
        tdt = getattr(torch, _TORCH_DT[dtype])
        B, S = 1, 512
        scale = 1.0 / math.sqrt(d)
        torch.manual_seed(0)

        # run_attention_dense_torch ABI: q/out [B,S,Hq,D], k/v [B,S,Hkv,D], dense.
        q = torch.randn(B, S, hq, d, device="cuda", dtype=tdt)
        k = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
        v = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
        out = torch.empty(B, S, hq, d, device="cuda", dtype=tdt)

        spec = _spec(dtype, d, hq, hkv, persistent, batch=B, sq=S)
        run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
        torch.cuda.synchronize()

        # Reference: PyTorch fp32 SDPA
        ref = _standard_reference(q, k, v, scale)

        max_abs = (ref - out.float()).abs().max().item()
        assert max_abs < tol, (
            f"{dtype} D{d} GQA{hq}/{hkv} "
            f"{'persist' if persistent else 'default'}: max_abs={max_abs:.3e} >= {tol}"
        )

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("persistent", [False, True])
    def test_bf16_d128_non_default_scale(self, persistent):
        """BF16 D128 at softmax scale 0.5, well above the default 1/sqrt(D).

        The score error a lossy scale step introduces grows with ``scale``, so
        the default-scale cohort above cannot see it. Rounding
        ``Q * scale * log2(e)`` back to bf16 before the QK MFMA puts this shape
        past the bf16 tolerance; the kernel must apply the scale in fp32.
        """
        import torch

        dtype, d, hq, hkv = "bf16", 128, 16, 4
        tol = _tolerance(dtype)
        B, S, scale = 1, 512, 0.5
        torch.manual_seed(0)

        q = torch.randn(B, S, hq, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, S, hkv, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, S, hkv, d, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(B, S, hq, d, device="cuda", dtype=torch.bfloat16)

        spec = _spec(dtype, d, hq, hkv, persistent, batch=B, sq=S)
        run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
        torch.cuda.synchronize()

        ref = _standard_reference(q, k, v, scale)
        max_abs = (ref - out.float()).abs().max().item()
        assert max_abs < tol, (
            f"bf16 D128 GQA16/4 scale=0.5 "
            f"{'persist' if persistent else 'default'}: max_abs={max_abs:.3e} >= {tol}"
        )

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("persistent", [False, True])
    @pytest.mark.parametrize("dtype", ["bf16", "fp16"])
    @pytest.mark.parametrize("scale", [2.0**-64, 2.0**4], ids=["2^-64", "2^4"])
    def test_d128_scale_range_edges(self, scale, dtype, persistent):
        """Both bounds of the supported softmax-scale range, on silicon.

        At 2**-64 every scaled score is about 0, so the softmax is close to
        uniform; at 2**4 it is close to one-hot and the ordinary kernel's fma
        residue is at its largest. Both must match the fp32 reference.
        """
        import torch

        d, hq, hkv = 128, 16, 4
        tol = _tolerance(dtype)
        tdt = getattr(torch, _TORCH_DT[dtype])
        B, S = 1, 512
        torch.manual_seed(0)

        q = torch.randn(B, S, hq, d, device="cuda", dtype=tdt)
        k = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
        v = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
        out = torch.empty(B, S, hq, d, device="cuda", dtype=tdt)

        spec = _spec(dtype, d, hq, hkv, persistent, batch=B, sq=S)
        run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
        torch.cuda.synchronize()

        assert torch.isfinite(out).all(), "non-finite output"
        ref = _standard_reference(q, k, v, scale)
        max_abs = (ref - out.float()).abs().max().item()
        assert max_abs < tol, (
            f"{dtype} D128 GQA16/4 scale={scale:g} "
            f"{'persist' if persistent else 'default'}: max_abs={max_abs:.3e} >= {tol}"
        )

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("persistent", [False, True])
    @pytest.mark.parametrize(
        "d,scale", [(64, 1.0 / math.sqrt(64)), (128, 0.5), (128, 1.0), (128, 2.0**4)]
    )
    def test_bf16_sliding_window_no_sinks(self, d, scale, persistent):
        """Causal sliding window without sinks, where whole query rows of the
        first visited KV tile are masked.

        Those rows start the online softmax from the mask sentinel, not from a
        real score or a sink logit. The sentinel must survive the softmax scale
        exactly, or exp2 of a huge rounding residue turns the row into inf/NaN.
        scale * log2(e) <= 1 (D64 at its default scale, D128 at 0.5) and > 1
        (D128 at 1.0, the hipDNN default when no scale is given, and at 2**4,
        the upper bound) take different exact branches in the ordinary kernel;
        both are covered.
        """
        import torch

        dtype, hq, hkv, window = "bf16", 16, 4, 128
        tol = _tolerance(dtype)
        B, S = 1, 512
        torch.manual_seed(0)

        q = torch.randn(B, S, hq, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, S, hkv, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, S, hkv, d, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(B, S, hq, d, device="cuda", dtype=torch.bfloat16)

        spec = _spec(
            dtype, d, hq, hkv, persistent, batch=B, sq=S, sliding_window=window
        )
        run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
        torch.cuda.synchronize()

        assert torch.isfinite(out).all(), "non-finite output"
        ref = _standard_reference(q, k, v, scale, sliding_window=window)
        max_abs = (ref - out.float()).abs().max().item()
        assert max_abs < tol, (
            f"bf16 D{d} SWA{window} scale={scale:g} "
            f"{'persist' if persistent else 'default'}: max_abs={max_abs:.3e} >= {tol}"
        )

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "d,persistent,width",
        [(128, False, 1), (128, True, 2), (64, False, 2), (64, True, 1)],
    )
    def test_bf16_narrow_output_store_is_bit_identical(self, d, persistent, width):
        """bf16 o_store_width 1/2 must write the same bits as the width-4 store
        (fp16 is rejected below 4 because it does not)."""
        import torch

        hq, hkv, B, S = 16, 4, 1, 512
        scale = 1.0 / math.sqrt(d)
        torch.manual_seed(0)
        q = torch.randn(B, S, hq, d, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, S, hkv, d, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, S, hkv, d, device="cuda", dtype=torch.bfloat16)
        base = _spec("bf16", d, hq, hkv, persistent, batch=B, sq=S)

        outs = []
        for spec in (base, dataclasses.replace(base, o_store_width=width)):
            out = torch.empty(B, S, hq, d, device="cuda", dtype=torch.bfloat16)
            run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
            outs.append(out)
        torch.cuda.synchronize()
        assert torch.equal(outs[0], outs[1]), (
            f"bf16 D{d} {'persist' if persistent else 'default'} "
            f"o_store_width={width} diverged from width 4"
        )

    @requires_gfx950_gpu
    @pytest.mark.gpu
    def test_one_binary_serves_every_shape(self):
        """One compiled artifact, two shapes, correct numerics at both.

        The cohort above runs many shapes, but it stopped discriminating the
        moment batch/seqlen_q/seqlen_kv became runtime kernel params: it passes
        identically whether N shapes are served by N binaries or by one. The
        property that actually needs a guard now is *artifact reuse*.

        ``is`` on the launcher covers the whole path rather than just the key
        function. A key regression would land the two shapes in different cache
        slots; a lookup regression would overwrite the one slot with a freshly
        compiled launcher. Both yield a different object, and neither is visible
        to a numeric assertion -- recompiling per shape is *correct*, merely
        wasteful, so what regresses is the AOT instance count and first-call
        latency, which no accuracy check can see.

        Paired with the numeric check at both shapes so the test cannot pass by
        reusing one binary that happens to be wrong for the second shape.
        """
        import torch

        from kernels.common.attention_dense_spec import attention_dense_cache_key
        from kernels.gfx950.attention_dense import _DENSE_LAUNCHER_CACHE

        dtype, d, hq, hkv = "bf16", 128, 16, 4  # flagship default, non-persistent
        tol = _tolerance(dtype)
        tdt = getattr(torch, _TORCH_DT[dtype])
        scale = 1.0 / math.sqrt(d)

        shapes = ((1, 512), (4, 1024))
        specs = [_spec(dtype, d, hq, hkv, False, batch=b, sq=s) for b, s in shapes]

        # Preconditions: genuinely different shapes, on the runtime path, and
        # sharing one key -- otherwise the reuse assertion is vacuous.
        assert specs[0].runtime_shape, "cohort row is not on the runtime-shape path"
        assert (specs[0].batch, specs[0].seqlen_q) != (
            specs[1].batch,
            specs[1].seqlen_q,
        )
        keys = [attention_dense_cache_key(s, arch="gfx950") for s in specs]
        assert keys[0] == keys[1], f"shapes {shapes} did not share a cache key"

        # Own the cache state: evicting first makes the "exactly one new entry"
        # assertion independent of which tests ran before this one.
        _DENSE_LAUNCHER_CACHE.pop(keys[0], None)
        before = set(_DENSE_LAUNCHER_CACHE)

        launchers = []
        for (B, S), spec in zip(shapes, specs):
            torch.manual_seed(0)
            q = torch.randn(B, S, hq, d, device="cuda", dtype=tdt)
            k = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
            v = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
            out = torch.empty(B, S, hq, d, device="cuda", dtype=tdt)

            run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
            torch.cuda.synchronize()
            launchers.append(_launcher_for(spec))

            ref = _standard_reference(q, k, v, scale)
            max_abs = (ref - out.float()).abs().max().item()
            assert max_abs < tol, f"B={B} S={S}: max_abs={max_abs:.3e} >= {tol}"

        assert launchers[0] is not None, (
            "no launcher cached after a successful run; _DENSE_LAUNCHER_CACHE is "
            "no longer keyed by attention_dense_cache_key and this test is blind"
        )
        assert launchers[0] is launchers[1], (
            f"shapes {shapes} share a cache key but were served by DIFFERENT "
            "launcher objects -- the runtime-shape kernel recompiled per shape, "
            "so the AOT instance count still scales with the shape space"
        )
        assert set(_DENSE_LAUNCHER_CACHE) - before == {keys[0]}, (
            "two shapes on the runtime path added more than one cache entry: "
            f"{sorted(set(_DENSE_LAUNCHER_CACHE) - before)}"
        )


# Sink-enabled numerical tests: verify that the sink-enabled kernel matches the
# reference softmax(concat([QK scores, sink]))[..., :-1] @ V formula.
#
# Two critical cases:
# 1. Sink ABOVE first tile's QK maximum -> tests m_init handling
# 2. Sink BELOW first tile's QK maximum -> tests l_init * alpha0 correction
#
# These verify the online softmax rescale logic that IR parity alone cannot validate.
# Cohort format: (dtype, head_size, num_query_heads, num_kv_heads, persistent, sliding_window, causal)
_SINK_COHORT = [
    # Plain causal (no sliding window)
    ("bf16", 128, 32, 8, False, 0, True),  # bf16 D128 GQA default
    ("bf16", 128, 32, 8, True, 0, True),  # bf16 D128 GQA persistent
    ("bf16", 64, 32, 8, False, 0, True),  # bf16 D64 GQA default
    ("bf16", 64, 32, 8, True, 0, True),  # bf16 D64 GQA persistent
    ("fp16", 128, 32, 8, False, 0, True),  # fp16 D128 GQA default
    ("fp16", 128, 32, 8, True, 0, True),  # fp16 D128 GQA persistent
    # SWA (sinks + sliding window)
    ("bf16", 128, 32, 8, False, 128, True),  # bf16 D128 GQA SWA default
    ("bf16", 128, 32, 8, True, 128, True),  # bf16 D128 GQA SWA persistent
    ("fp16", 128, 32, 8, False, 256, True),  # fp16 D128 GQA SWA default
    ("fp16", 128, 32, 8, True, 256, True),  # fp16 D128 GQA SWA persistent
    # Non-causal (full attention + sinks)
    ("bf16", 128, 32, 8, False, 0, False),  # bf16 D128 GQA non-causal default
    ("bf16", 128, 32, 8, True, 0, False),  # bf16 D128 GQA non-causal persistent
]


def _sink_reference(q, k, v, sinks, scale, sliding_window=0, causal=True):
    """Manual attention with sinks: softmax(concat([QK, sink]))[..., :-1] @ V.

    This reference computes the full [B, Hq, S, S] attention matrix at once (NOT
    query-chunked). Unlike the builder script's reference
    (library/builders/gfx950/attention/prefill/attention_dense_prefill.py), which
    chunks over queries to avoid OOM at large seqlens (Sq=8192, Hq=128 -> 34.4 GB),
    this test uses small fixed shapes (S=512, Hq=32 -> 32 MB) that fit comfortably
    in memory. The full-matrix version is simpler and clearer as a test oracle.

    Args:
        q: [B, S, Hq, D] query tensor
        k: [B, S, Hkv, D] key tensor
        v: [B, S, Hkv, D] value tensor
        sinks: [Hq] sink logits (per query head, raw attention scores)
        scale: softmax scale (1/sqrt(D))
        sliding_window: sliding window size (0 = disabled)
        causal: whether to apply causal masking

    Returns:
        ref: [B, S, Hq, D] reference output in fp32
    """
    import torch

    B, S, Hq, D = q.shape
    Hkv = k.shape[2]
    rep = Hq // Hkv
    dev = q.device

    # Expand to [B, Hq, S, D] and expand GQA
    qh = q.transpose(1, 2).float()
    kh = k.transpose(1, 2).repeat_interleave(rep, 1).float()
    vh = v.transpose(1, 2).repeat_interleave(rep, 1).float()

    # Compute QK scores: [B, Hq, S, S]
    attn = torch.einsum("bhqd,bhkd->bhqk", qh, kh) * scale

    # Apply causal and/or sliding window mask
    if causal or sliding_window > 0:
        qi = torch.arange(S, device=dev).view(-1, 1)
        ki = torch.arange(S, device=dev).view(1, -1)
        mask = torch.zeros(S, S, dtype=torch.bool, device=dev)
        # Causal: mask future tokens (only if causal)
        if causal:
            mask |= ki > qi
        # Sliding window: mask tokens beyond window (when enabled)
        if sliding_window > 0:
            mask |= ki <= (qi - sliding_window)
        attn.masked_fill_(mask.view(1, 1, S, S), float("-inf"))

    # Append sink column: sinks are raw attention scores (same domain as attn)
    sink_col = sinks.float().view(1, Hq, 1, 1).expand(B, Hq, S, 1)
    attn = torch.cat([attn, sink_col], dim=-1)  # [B, Hq, S, S+1]

    # Softmax over all scores (including sink)
    attn = torch.softmax(attn, dim=-1)

    # Remove sink from attention weights (sink has no V vector)
    attn = attn[..., :-1]  # [B, Hq, S, S]

    # Weighted sum over values
    ref = torch.einsum("bhqk,bhkd->bhqd", attn, vh).transpose(1, 2)
    return ref


_GQA_PAIR_VARIANTS = [
    ("plain", 0, False, False),
    ("interleave", 0, False, True),
    ("sliding_window", 128, False, False),
    ("sinks", 0, True, False),
    ("sinks_sliding_window", 128, True, False),
]


class TestDenseGqaPairVariants:
    """Numeric coverage for features composed with the optimized work mapping."""

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "_name,sliding_window,use_sinks,interleave", _GQA_PAIR_VARIANTS
    )
    def test_gqa_pair_variant_numeric(
        self, _name, sliding_window, use_sinks, interleave
    ):
        import torch

        B, S, Hq, Hkv, D = 1, 512, 32, 8, 128
        torch.manual_seed(0)
        q = torch.randn(B, S, Hq, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, S, Hkv, D, device="cuda", dtype=torch.float16)
        out = torch.empty_like(q)
        sinks = (
            torch.zeros(Hq, device="cuda", dtype=torch.float16) if use_sinks else None
        )
        scale = 1.0 / math.sqrt(D)

        # NQB=2 and Hkv=8, so the balanced mapping needs exactly 16 CTAs.
        spec = Gfx950AttentionDenseSpec(
            batch=B,
            seqlen_q=S,
            seqlen_kv=S,
            num_query_heads=Hq,
            num_kv_heads=Hkv,
            head_size=D,
            causal=True,
            dtype="fp16",
            block_n=64,
            persistent=True,
            num_persistent=16,
            persist_decode="gqa_pair",
            interleave=interleave,
            sliding_window=sliding_window,
            use_sinks=use_sinks,
            wide_lds_dma=True,
        )
        run_attention_dense_torch(
            spec=spec,
            q=q,
            k=k,
            v=v,
            out=out,
            scale=scale,
            sinks=sinks,
        )
        torch.cuda.synchronize()

        if use_sinks:
            ref = _sink_reference(
                q,
                k,
                v,
                sinks,
                scale,
                sliding_window=sliding_window,
                causal=True,
            )
        else:
            ref = _standard_reference(q, k, v, scale, sliding_window=sliding_window)
        max_abs = (ref - out.float()).abs().max().item()
        assert max_abs < _tolerance("fp16"), f"gqa_pair {_name}: max_abs={max_abs:.3e}"

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("block_m", (256, 128), ids=("bm256", "bm128"))
    def test_wide_dma_mha_numeric(self, block_m):
        """Both wide-DMA line-pass geometries work without GQA reuse."""
        import torch

        B, S, H, D = 1, 512, 16, 128
        torch.manual_seed(0)
        q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        out = torch.empty_like(q)
        scale = 1.0 / math.sqrt(D)
        spec = Gfx950AttentionDenseSpec(
            batch=B,
            seqlen_q=S,
            seqlen_kv=S,
            num_query_heads=H,
            num_kv_heads=H,
            head_size=D,
            causal=True,
            dtype="fp16",
            block_m=block_m,
            block_n=64,
            persistent=True,
            num_persistent=32,
            persist_decode="qb_major",
            wide_lds_dma=True,
        )
        run_attention_dense_torch(spec=spec, q=q, k=k, v=v, out=out, scale=scale)
        torch.cuda.synchronize()
        ref = _standard_reference(q, k, v, scale)
        max_abs = (ref - out.float()).abs().max().item()
        assert max_abs < _tolerance("fp16"), f"wide DMA MHA: max_abs={max_abs:.3e}"


class TestDenseSinksNumeric:
    """Sink-enabled dense attention numerical validation.

    Verifies that the sink-enabled kernel matches the reference
    softmax(concat([QK scores, sink]))[..., :-1] @ V formula.

    Two critical cases:
    1. Sink ABOVE first tile's QK maximum -> tests m_init handling
    2. Sink BELOW first tile's QK maximum -> tests l_init * alpha0 correction

    These verify the online softmax rescale logic that IR parity alone cannot validate.
    """

    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype,d,hq,hkv,persistent,sw,causal", _SINK_COHORT)
    @pytest.mark.parametrize("sink_magnitude", ["above_qk_max", "below_qk_max"])
    def test_sinks_numeric(
        self, dtype, d, hq, hkv, persistent, sw, causal, sink_magnitude
    ):
        """Verify sink-enabled kernel against manual sink reference.

        Tests two critical cases:
        - sink_magnitude='above_qk_max': Sink > first tile's QK max -> tests m_init
        - sink_magnitude='below_qk_max': Sink < first tile's QK max -> tests l_init*alpha0

        These verify the online softmax rescale logic with sinks, which IR parity alone
        cannot validate (m_init and l_init depend on the runtime QK values).

        Note on sliding windows (sw > 0): With the first tile (queries [0:256], keys
        [0:64]), some query rows may see no keys from [0:64] (their sliding window
        has moved past the key range) and produce all -inf scores. However, queries
        earlier in the sequence still see keys and produce finite QK scores. The max
        operation across all query rows naturally picks the finite maximum from the
        queries that DO see keys, ensuring the above/below sink comparison remains
        meaningful. At least one WG (first tile) hits the valid path.
        """
        import torch

        tol = _tolerance(dtype)
        tdt = getattr(torch, _TORCH_DT[dtype])
        B, S = 1, 512
        scale = 1.0 / math.sqrt(d)
        torch.manual_seed(0)

        # Standard Q/K/V tensors
        q = torch.randn(B, S, hq, d, device="cuda", dtype=tdt)
        k = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
        v = torch.randn(B, S, hkv, d, device="cuda", dtype=tdt)
        out = torch.empty(B, S, hq, d, device="cuda", dtype=tdt)

        # Compute first tile's QK maximum to set sink relative to it.
        # We use the FIRST tile (not global max) because sinks are constant scalars
        # available from the start, so the critical test is: "when the kernel processes
        # tile 0, does it correctly initialize m_init and l_init relative to the sink?"
        # Setting sink above/below first_tile_max directly tests this initialization.
        # First tile: queries [0:256], keys [0:64] (block_n=64 default)
        q_tile = q[:, :256].transpose(1, 2).float()  # [B, Hq, 256, D]
        k_tile = (
            k[:, :64].transpose(1, 2).float().repeat_interleave(hq // hkv, 1)
        )  # [B, Hq, 64, D]
        qk_tile = (
            torch.einsum("bhqd,bhkd->bhqk", q_tile, k_tile) * scale
        )  # [B, Hq, 256, 64]
        # Apply causal mask to first tile (only if causal)
        if causal:
            qi = torch.arange(256, device="cuda").view(-1, 1)
            ki = torch.arange(64, device="cuda").view(1, -1)
            mask = ki > qi
            qk_tile.masked_fill_(mask.view(1, 1, 256, 64), float("-inf"))
        first_tile_max = qk_tile.max(dim=-1)[0].max(dim=-1)[0]  # [B, Hq] max per head

        # Create sink values relative to first tile's QK max
        if sink_magnitude == "above_qk_max":
            # Sink ABOVE first tile max -> tests m_init = max(m_tile0, sink)
            sinks = first_tile_max[0] + 1.0  # [Hq], +1 above max
        else:  # below_qk_max
            # Sink BELOW first tile max -> tests l_init * alpha0 correction
            sinks = first_tile_max[0] - 1.0  # [Hq], -1 below max

        sinks = sinks.to(tdt)  # Match input dtype

        # Run kernel with sinks (and optional sliding window / causal)
        spec = _spec(
            dtype,
            d,
            hq,
            hkv,
            persistent,
            batch=B,
            sq=S,
            use_sinks=True,
            sliding_window=sw,
            causal=causal,
        )
        run_attention_dense_torch(
            spec=spec, q=q, k=k, v=v, out=out, scale=scale, sinks=sinks
        )
        torch.cuda.synchronize()

        # Reference: softmax(concat([QK, sink]))[..., :-1] @ V
        ref = _sink_reference(q, k, v, sinks, scale, sliding_window=sw, causal=causal)

        max_abs = (ref - out.float()).abs().max().item()
        # Build descriptive label
        mask_label = []
        if not causal:
            mask_label.append("full")
        else:
            mask_label.append("causal")
        if sw > 0:
            mask_label.append(f"sw{sw}")
        mask_str = "+".join(mask_label)

        assert max_abs < tol, (
            f"{dtype} D{d} GQA{hq}/{hkv} "
            f"{'persist' if persistent else 'default'} {mask_str} sink_{sink_magnitude}: "
            f"max_abs={max_abs:.3e} >= {tol}"
        )


def _bottom_right_reference(q, k, v, sinks, scale, *, top_left):
    """Explicit fp32 causal oracle for unequal query and KV lengths."""
    import torch

    batch, sq, hq, _ = q.shape
    skv = k.shape[1]
    hkv = k.shape[2]
    qh = q.transpose(1, 2).float()
    kh = k.transpose(1, 2).repeat_interleave(hq // hkv, dim=1).float()
    vh = v.transpose(1, 2).repeat_interleave(hq // hkv, dim=1).float()

    scores = torch.einsum("bhqd,bhkd->bhqk", qh, kh) * scale
    qi = torch.arange(sq, device=q.device).view(-1, 1)
    ki = torch.arange(skv, device=q.device).view(1, -1)
    diagonal = 0 if top_left else skv - sq
    allowed = ki <= qi + diagonal
    scores.masked_fill_(~allowed.view(1, 1, sq, skv), float("-inf"))

    if sinks is None:
        probabilities = torch.softmax(scores, dim=-1)
    else:
        sink_col = sinks.float().view(1, hq, 1, 1).expand(batch, hq, sq, 1)
        probabilities = torch.softmax(torch.cat([scores, sink_col], dim=-1), dim=-1)[
            ..., :-1
        ]
    return torch.einsum("bhqk,bhkd->bhqd", probabilities, vh).transpose(1, 2)


# The original nine PR cases plus one aligned and one ragged BM128 case. Each
# pytest.param is a separate GPU test node: on a real gfx950 none can disappear
# behind an aggregate helper that reports success after skipping every case.
_BOTTOM_RIGHT_CASES = [
    pytest.param("default", 197, 400, 1, True, False, id="bm256-ragged-197x400"),
    pytest.param("default", 300, 1234, 1, True, False, id="bm256-ragged-300x1234"),
    pytest.param("default", 512, 4097, 1, True, False, id="bm256-ragged-512x4097"),
    pytest.param("default", 100, 8000, 1, True, False, id="bm256-ragged-100x8000"),
    # The ragged buffer spans the whole tensor: padded rows in batch 0 can read
    # live batch-1 data, so correctness depends on the qtok < Sq store guard.
    pytest.param("default", 300, 1000, 2, True, False, id="bm256-ragged-batch2"),
    pytest.param("default", 256, 512, 1, False, False, id="bm256-aligned-256x512"),
    pytest.param("default", 512, 1024, 1, False, False, id="bm256-aligned-512x1024"),
    pytest.param("default", 512, 1024, 1, False, True, id="bm256-sinks-aligned"),
    pytest.param("default", 300, 1234, 1, True, True, id="bm256-sinks-ragged"),
    pytest.param("bm128", 512, 1024, 1, False, False, id="bm128-aligned-512x1024"),
    pytest.param("bm128", 197, 400, 1, True, False, id="bm128-ragged-197x400"),
]

_BOTTOM_RIGHT_TOL = 2e-2


class TestDenseBottomRightNumeric:
    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "geometry,sq,skv,batch,ragged,use_sinks", _BOTTOM_RIGHT_CASES
    )
    def test_jit_matches_shifted_diagonal_not_top_left(
        self, geometry, sq, skv, batch, ragged, use_sinks
    ):
        import torch

        tile = DENSE_TILE_GEOMETRIES[geometry]
        spec = Gfx950AttentionDenseSpec(
            batch=batch,
            seqlen_q=sq,
            seqlen_kv=skv,
            num_query_heads=4,
            num_kv_heads=1,
            head_size=128,
            causal=True,
            dtype="bf16",
            block_m=int(tile["block_m"]),
            block_n=int(tile["block_n"]),
            ragged=ragged,
            persistent=False,
            use_sinks=use_sinks,
            causal_bottom_right=True,
        )
        assert skv > sq, "every case must exercise a moving diagonal"

        torch.manual_seed(0)
        q = torch.randn(batch, sq, 4, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(batch, skv, 1, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(batch, skv, 1, 128, device="cuda", dtype=torch.bfloat16)
        out = torch.empty_like(q)
        sinks = (
            torch.randn(4, device="cuda", dtype=torch.bfloat16) if use_sinks else None
        )
        scale = 1.0 / math.sqrt(spec.head_size)

        run_attention_dense_torch(
            spec=spec,
            q=q,
            k=k,
            v=v,
            out=out,
            scale=scale,
            sinks=sinks,
        )
        torch.cuda.synchronize()

        shifted = _bottom_right_reference(q, k, v, sinks, scale, top_left=False)
        top_left = _bottom_right_reference(q, k, v, sinks, scale, top_left=True)
        shifted_err = (out.float() - shifted).abs().max().item()
        top_left_err = (out.float() - top_left).abs().max().item()

        label = (
            f"{geometry} B{batch} {sq}x{skv} "
            f"{'ragged' if ragged else 'aligned'}"
            f"{'+sinks' if use_sinks else ''}"
        )
        assert shifted_err < _BOTTOM_RIGHT_TOL, (
            f"{label}: shifted max_abs={shifted_err:.3e} " f">= {_BOTTOM_RIGHT_TOL}"
        )
        assert top_left_err > 1e-3, (
            f"{label}: unexpectedly matches top-left " f"(max_abs={top_left_err:.3e})"
        )


# The wide-DMA candidate admits every mask the persistent body implements, not
# only the causal no-sinks no-SWA shapes the cohort above routes to it.
_WIDE_DMA_MASKS = [
    # (name, causal, sliding_window, use_sinks)
    ("non_causal", False, 0, False),
    ("causal_sinks", True, 0, True),
    ("causal_swa", True, 128, False),
    ("non_causal_sinks", False, 0, True),
    ("causal_swa_sinks", True, 128, True),
]


class TestWideDmaFeatures:
    @requires_gfx950_gpu
    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", ("bf16", "fp16"))
    @pytest.mark.parametrize("block_m", (256, 128))
    @pytest.mark.parametrize("_name,causal,sliding_window,use_sinks", _WIDE_DMA_MASKS)
    def test_wide_dma_numeric(
        self, _name, causal, sliding_window, use_sinks, block_m, dtype
    ):
        import torch

        from dispatch.attention import AttentionRequest, tuning_spec_with_knobs

        B, S, Hq, Hkv, D = 1, 512, 32, 8, 128
        spec = tuning_spec_with_knobs(
            AttentionRequest(
                batch=B,
                nhead_q=Hq,
                nhead_k=Hkv,
                seqlen_q=S,
                seqlen_k=S,
                hdim_q=D,
                hdim_v=D,
                arch="gfx950",
                mask_type=1 if causal else 0,
                dtype=dtype,
                use_sinks=use_sinks,
                sliding_window=sliding_window,
            ),
            "gfx950_dense_persist_widedma",
            {"block_m": block_m},
        ).kernel_spec
        assert spec.wide_lds_dma
        assert spec.block_m == block_m

        tdt = getattr(torch, _TORCH_DT[dtype])
        torch.manual_seed(0)
        q = torch.randn(B, S, Hq, D, device="cuda", dtype=tdt)
        k = torch.randn(B, S, Hkv, D, device="cuda", dtype=tdt)
        v = torch.randn(B, S, Hkv, D, device="cuda", dtype=tdt)
        out = torch.empty_like(q)
        sinks = torch.randn(Hq, device="cuda", dtype=tdt) if use_sinks else None
        scale = 1.0 / math.sqrt(D)
        run_attention_dense_torch(
            spec=spec, q=q, k=k, v=v, out=out, scale=scale, sinks=sinks
        )
        torch.cuda.synchronize()

        # A sink of -inf contributes nothing, so one reference covers both.
        ref_sinks = (
            sinks
            if use_sinks
            else torch.full((Hq,), float("-inf"), device="cuda", dtype=torch.float32)
        )
        ref = _sink_reference(
            q, k, v, ref_sinks, scale, sliding_window=sliding_window, causal=causal
        )
        max_abs = (ref - out.float()).abs().max().item()
        assert max_abs < _tolerance(
            dtype
        ), f"{spec_id} {dtype} {_name}: max_abs={max_abs:.3e}"


def _lse_reference(q, k, v, scale, *, causal, diagonal=0, window=0, sinks=None):
    """Independent FP32 O and natural-log LSE, including truly empty rows.

    Scores use FP32 Q/K with the softmax scale applied in FP32.
    A sink adds denominator mass but contributes no value vector.
    """
    batch, sq, hq, _ = q.shape
    skv, hkv = k.shape[1:3]
    qh = q.transpose(1, 2).float()
    kh = k.transpose(1, 2).float().repeat_interleave(hq // hkv, dim=1)
    vh = v.transpose(1, 2).float().repeat_interleave(hq // hkv, dim=1)
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
    qi = torch.arange(sq, device=q.device).view(-1, 1) + diagonal
    ki = torch.arange(skv, device=q.device).view(1, -1)
    allowed = torch.ones((sq, skv), device=q.device, dtype=torch.bool)
    if causal:
        allowed &= ki <= qi
    if window:
        allowed &= ki > qi - window
    scores = scores.masked_fill(~allowed.view(1, 1, sq, skv), float("-inf"))
    if sinks is not None:
        sink_scores = sinks.float().view(1, hq, 1, 1).expand(batch, hq, sq, 1)
        scores = torch.cat((scores, sink_scores), dim=-1)
    lse = torch.logsumexp(scores, dim=-1)
    has_mass = torch.isfinite(lse)
    safe_lse = torch.where(has_mass, lse, torch.zeros_like(lse))
    probabilities = torch.exp(scores - safe_lse.unsqueeze(-1))
    probabilities = torch.where(has_mass.unsqueeze(-1), probabilities, 0.0)
    out = torch.matmul(probabilities[..., :skv], vh).transpose(1, 2)
    return out, lse.unsqueeze(-1)


@requires_gfx950_gpu
@pytest.mark.gpu
class TestDenseLseNumeric:
    """Direct-runner LSE coverage; no dispatcher or hipDNN feature is implied."""

    @pytest.mark.parametrize("dtype", ["fp16", "bf16"])
    @pytest.mark.parametrize("head_size", [64, 128])
    @pytest.mark.parametrize(
        "heads", [(4, 4), (4, 1), (8, 2)], ids=["mha", "mqa", "gqa"]
    )
    @pytest.mark.parametrize(
        "persistent", [False, True], ids=["ordinary", "persistent"]
    )
    @pytest.mark.parametrize("geometry", ["default", "bm128"])
    def test_output_and_lse(self, dtype, head_size, heads, persistent, geometry):
        hq, hkv = heads
        spec = Gfx950AttentionDenseSpec(
            batch=2,
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=hq,
            num_kv_heads=hkv,
            head_size=head_size,
            dtype=dtype,
            persistent=persistent,
            block_m=int(DENSE_TILE_GEOMETRIES[geometry]["block_m"]),
            emit_lse=True,
        )
        tdt = getattr(torch, _TORCH_DT[dtype])
        scale = 0.5
        torch.manual_seed(11)
        q = torch.randn(2, 512, hq, head_size, device="cuda", dtype=tdt)
        k = torch.randn(2, 512, hkv, head_size, device=q.device, dtype=tdt)
        v = torch.randn_like(k)
        out = torch.empty_like(q)
        lse = torch.full(
            (2, hq, 512, 1),
            float("nan"),
            device=q.device,
            dtype=torch.float32,
        )
        returned = run_attention_dense_torch(
            spec=spec,
            q=q,
            k=k,
            v=v,
            out=out,
            lse=lse,
            scale=scale,
        )
        torch.cuda.synchronize()
        ref, lse_ref = _lse_reference(q, k, v, scale, causal=True)
        assert returned is out
        assert (out.float() - ref).abs().max().item() < _tolerance(dtype)
        assert (lse - lse_ref).abs().max().item() < _tolerance(dtype)
        off_out = torch.empty_like(out)
        run_attention_dense_torch(
            spec=replace(spec, emit_lse=False),
            q=q,
            k=k,
            v=v,
            out=off_out,
            scale=scale,
        )
        torch.cuda.synchronize()
        # Every row has a key. Empty-row corrections are enabled-path-only.
        assert torch.equal(out, off_out)

    @pytest.mark.parametrize("width", [1, 2])
    @pytest.mark.parametrize(
        "persistent", [False, True], ids=["ordinary", "persistent"]
    )
    def test_narrow_o_store(self, width, persistent):
        """BF16 store widths 1/2: keyless rows are zero, keyed rows match LSE-off."""
        spec = Gfx950AttentionDenseSpec(
            batch=1,
            seqlen_q=256,
            seqlen_kv=64,
            num_query_heads=4,
            num_kv_heads=1,
            head_size=128,
            dtype="bf16",
            persistent=persistent,
            sliding_window=64,
            o_store_width=width,
            emit_lse=True,
        )
        scale = 0.5
        torch.manual_seed(17)
        q = torch.randn(1, 256, 4, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 64, 1, 128, device=q.device, dtype=q.dtype)
        v = torch.randn_like(k)
        out = torch.full_like(q, float("nan"))
        lse = torch.full(
            (1, 4, 256, 1), float("nan"), device=q.device, dtype=torch.float32
        )
        run_attention_dense_torch(
            spec=spec, q=q, k=k, v=v, out=out, lse=lse, scale=scale
        )
        off_out = torch.empty_like(q)
        run_attention_dense_torch(
            spec=replace(spec, emit_lse=False),
            q=q,
            k=k,
            v=v,
            out=off_out,
            scale=scale,
        )
        torch.cuda.synchronize()
        ref, lse_ref = _lse_reference(q, k, v, scale, causal=True, window=64)
        # Rows 0..126 see at least one key; rows 127.. see none.
        assert (out[:, :127].float() - ref[:, :127]).abs().max().item() < _tolerance(
            "bf16"
        )
        assert torch.equal(out[:, :127], off_out[:, :127])
        assert torch.equal(out[:, 127:], torch.zeros_like(out[:, 127:]))
        finite = torch.isfinite(lse_ref)
        assert (lse[finite] - lse_ref[finite]).abs().max().item() < _tolerance("bf16")
        assert torch.equal(torch.isneginf(lse), torch.isneginf(lse_ref))

    @pytest.mark.parametrize(
        "persistent", [False, True], ids=["ordinary", "persistent"]
    )
    @pytest.mark.parametrize("with_sink", [False, True], ids=["no-sink", "sink"])
    def test_empty_window_rows(self, persistent, with_sink):
        spec = Gfx950AttentionDenseSpec(
            batch=1,
            seqlen_q=256,
            seqlen_kv=64,
            num_query_heads=3,
            num_kv_heads=1,
            head_size=128,
            dtype="bf16",
            persistent=persistent,
            sliding_window=64,
            use_sinks=with_sink,
            emit_lse=True,
        )
        q = torch.zeros(1, 256, 3, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.zeros(1, 64, 1, 128, device=q.device, dtype=q.dtype)
        v = torch.ones_like(k)
        out = torch.full_like(q, float("nan"))
        lse = torch.full(
            (1, 3, 256, 1),
            float("nan"),
            device=q.device,
            dtype=torch.float32,
        )
        sinks = (
            torch.tensor([-2.0, 0.0, 2.0], device=q.device, dtype=q.dtype)
            if with_sink
            else None
        )
        run_attention_dense_torch(
            spec=spec,
            q=q,
            k=k,
            v=v,
            out=out,
            lse=lse,
            scale=0.5,
            sinks=sinks,
        )
        torch.cuda.synchronize()
        rows = torch.arange(256, device=q.device)
        counts = (
            (
                torch.minimum(rows + 1, torch.full_like(rows, 64))
                - (rows - 63).clamp_min(0)
            )
            .clamp_min(0)
            .float()
        )
        assert counts[63].item() == 64
        assert counts[126].item() == 1
        assert counts[127].item() == 0
        mass = counts.view(1, 1, 256, 1).expand(1, 3, 256, 1)
        if with_sink:
            mass = mass + sinks.float().exp().view(1, 3, 1, 1)
        analytic_lse = mass.log()
        ref, lse_ref = _lse_reference(
            q,
            k,
            v,
            0.5,
            causal=True,
            window=64,
            sinks=sinks,
        )
        finite = torch.isfinite(analytic_lse)
        assert (lse[finite] - analytic_lse[finite]).abs().max().item() < 1e-5
        assert (lse_ref[finite] - analytic_lse[finite]).abs().max().item() < 1e-5
        assert torch.equal(torch.isneginf(lse), torch.isneginf(analytic_lse))
        assert (out.float() - ref).abs().max().item() < _tolerance("bf16")
        assert torch.equal(out[:, 127:], torch.zeros_like(out[:, 127:]))
        if with_sink:
            expected = sinks.float().view(1, 3, 1, 1).expand(1, 3, 129, 1)
            assert (lse[:, :, 127:] - expected).abs().max().item() < 1e-5
        else:
            assert torch.isneginf(lse[:, :, 127:]).all().item()

    @pytest.mark.parametrize("dtype", ["fp16", "bf16"])
    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                {"ragged": True, "seqlen_q": 273, "seqlen_kv": 273}, id="ragged"
            ),
            pytest.param(
                {"ragged": True, "seqlen_q": 273, "seqlen_kv": 273, "persistent": True},
                id="ragged-persistent",
            ),
            pytest.param({"causal": False}, id="noncausal"),
            pytest.param(
                {"causal": False, "persistent": True}, id="noncausal-persistent"
            ),
            pytest.param({"sliding_window": 128}, id="window"),
            pytest.param(
                {"sliding_window": 128, "persistent": True}, id="window-persistent"
            ),
            pytest.param(
                {
                    "causal_bottom_right": True,
                    "ragged": True,
                    "seqlen_q": 197,
                    "seqlen_kv": 400,
                },
                id="shifted-ragged",
            ),
            pytest.param(
                {"causal_bottom_right": True, "seqlen_q": 256},
                id="shifted-aligned",
            ),
            pytest.param({"use_sinks": True}, id="sinks"),
            pytest.param(
                {"use_sinks": True, "persistent": True}, id="sinks-persistent"
            ),
            pytest.param(
                {"persistent": True, "persist_decode": "qb_major", "num_persistent": 4},
                id="qb-major",
            ),
            pytest.param(
                {
                    "persistent": True,
                    "persist_decode": "hkv_major",
                    "num_persistent": 4,
                },
                id="hkv-major",
            ),
            pytest.param(
                {"persistent": True, "persist_decode": "gqa_pair", "num_persistent": 4},
                id="gqa-pair",
            ),
            pytest.param(
                {
                    "persistent": True,
                    "persist_decode": "gqa_pair_2phase",
                    "num_persistent": 8,
                },
                id="gqa-pair-2phase",
            ),
            pytest.param(
                {
                    "persistent": True,
                    "persist_decode": "qb_major",
                    "num_persistent": 4,
                    "interleave": True,
                },
                id="interleaved",
            ),
            pytest.param(
                {
                    "persistent": True,
                    "persist_decode": "qb_major",
                    "num_persistent": 4,
                    "wide_lds_dma": True,
                },
                id="wide-dma",
            ),
            pytest.param(
                {
                    "batch": 1,
                    "paged": True,
                    "block_size": 16,
                    "num_kv_blocks": 32,
                    "sliding_window": 128,
                },
                id="reverse-paged",
            ),
        ],
    )
    def test_legal_modes_and_tail_guards(self, dtype, case):
        # One launch/check body, with separate pytest nodes for each legal mode.
        options = dict(
            batch=2,
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=4,
            num_kv_heads=1,
            head_size=128,
            dtype=dtype,
            emit_lse=True,
        )
        options.update(case)
        spec = Gfx950AttentionDenseSpec(**options)
        batch, sq, skv = spec.batch, spec.seqlen_q, spec.seqlen_kv
        tdt = getattr(torch, _TORCH_DT[dtype])
        torch.manual_seed(19)
        q = torch.randn(batch, sq, 4, 128, device="cuda", dtype=tdt)
        k = torch.randn(batch, skv, 1, 128, device=q.device, dtype=tdt)
        v = torch.randn_like(k)
        out_count = q.numel()
        out_storage = torch.full(
            (out_count + 16,),
            123.0,
            device=q.device,
            dtype=tdt,
        )
        out = out_storage[:out_count].view_as(q)
        count = batch * 4 * sq
        storage = torch.full(
            (count + 16,),
            12345.0,
            device=q.device,
            dtype=torch.float32,
        )
        lse = storage[:count].view(batch, 4, sq, 1)
        sinks = (
            torch.linspace(-1, 1, 4, device=q.device).to(tdt)
            if spec.use_sinks
            else None
        )
        launch_k, launch_v = k, v
        scale = 0.5
        metadata = {}
        if spec.paged:
            page_ids = torch.arange(31, -1, -1, device=q.device, dtype=torch.int32)
            launch_k = torch.empty(32, 16, 1, 128, device=q.device, dtype=tdt)
            launch_v = torch.empty_like(launch_k)
            launch_k[page_ids.long()] = k[0].view(32, 16, 1, 128)
            launch_v[page_ids.long()] = v[0].view(32, 16, 1, 128)
            metadata = {
                "block_tables": page_ids.view(1, 32),
                "kv_lens": torch.tensor([skv], device=q.device, dtype=torch.int32),
            }
        returned = run_attention_dense_torch(
            spec=spec,
            q=q,
            k=launch_k,
            v=launch_v,
            out=out,
            lse=lse,
            scale=scale,
            sinks=sinks,
            **metadata,
        )
        torch.cuda.synchronize()
        ref, lse_ref = _lse_reference(
            q,
            k,
            v,
            scale,
            causal=spec.causal,
            diagonal=skv - sq if spec.causal_bottom_right else 0,
            window=spec.sliding_window,
            sinks=sinks,
        )
        assert returned is out
        assert (out.float() - ref).abs().max().item() < _tolerance(dtype)
        assert (lse - lse_ref).abs().max().item() < _tolerance(dtype)
        assert torch.equal(storage[count:], torch.full_like(storage[count:], 12345.0))
        assert torch.equal(
            out_storage[out_count:],
            torch.full_like(out_storage[out_count:], 123.0),
        )
        off_out = torch.empty_like(out)
        run_attention_dense_torch(
            spec=replace(spec, emit_lse=False),
            q=q,
            k=launch_k,
            v=launch_v,
            out=off_out,
            scale=scale,
            sinks=sinks,
            **metadata,
        )
        torch.cuda.synchronize()
        assert torch.equal(out, off_out)

    @pytest.mark.parametrize("dtype", ["fp16", "bf16"])
    @pytest.mark.parametrize(
        "zero_q", [False, True], ids=["different-lengths", "zero-q"]
    )
    def test_packed_dense_lse(self, dtype, zero_q):
        q_offsets = [0, 0, 512] if zero_q else [0, 256, 768]
        k_offsets = [0, 128, 384]
        spec = Gfx950AttentionDenseSpec(
            batch=2,
            seqlen_q=512,
            seqlen_kv=256,
            num_query_heads=4,
            num_kv_heads=1,
            head_size=128,
            dtype=dtype,
            varlen=True,
            emit_lse=True,
        )
        torch.manual_seed(23)
        tdt = getattr(torch, _TORCH_DT[dtype])
        q = torch.randn(q_offsets[-1], 4, 128, device="cuda", dtype=tdt)
        k = torch.randn(k_offsets[-1], 1, 128, device=q.device, dtype=tdt)
        v = torch.randn_like(k)
        out = torch.full_like(q, float("nan"))
        count = q_offsets[-1] * 4
        storage = torch.full(
            (count + 16,),
            12345.0,
            device=q.device,
            dtype=torch.float32,
        )
        lse = storage[:count].view(q_offsets[-1], 4, 1)
        scale = 0.5
        cu_q = torch.tensor(q_offsets, device=q.device, dtype=torch.int32)
        cu_k = torch.tensor(k_offsets, device=q.device, dtype=torch.int32)
        returned = run_attention_dense_torch(
            spec=spec,
            q=q,
            k=k,
            v=v,
            out=out,
            lse=lse,
            scale=scale,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_k,
        )
        torch.cuda.synchronize()
        expected_out = torch.empty_like(out, dtype=torch.float32)
        expected_lse = torch.empty_like(lse)
        for seq in range(2):
            q0, q1 = q_offsets[seq : seq + 2]
            k0, k1 = k_offsets[seq : seq + 2]
            if q0 == q1:
                continue
            ref, lse_ref = _lse_reference(
                q[q0:q1].unsqueeze(0),
                k[k0:k1].unsqueeze(0),
                v[k0:k1].unsqueeze(0),
                scale,
                causal=True,
            )
            expected_out[q0:q1] = ref[0]
            expected_lse[q0:q1] = lse_ref[0].transpose(0, 1)
        assert returned is out
        assert (out.float() - expected_out).abs().max().item() < _tolerance(dtype)
        assert (lse - expected_lse).abs().max().item() < _tolerance(dtype)
        assert torch.equal(storage[count:], torch.full_like(storage[count:], 12345.0))
        # Packed varlen is the one mode where LSE changes the O store guard
        # (stores only rows that exist), so pin the bits against LSE off.
        off_out = torch.full_like(q, float("nan"))
        run_attention_dense_torch(
            spec=replace(spec, emit_lse=False),
            q=q,
            k=k,
            v=v,
            out=off_out,
            scale=scale,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_k,
        )
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), off_out.view(torch.int16))

    @pytest.mark.parametrize(
        "failure",
        [
            "q_tail",
            "kv_tail",
            "zero_kv",
            "decreasing",
            "first_offset",
            "terminal_offset",
            "q_above_max",
            "kv_above_max",
            "q_offset_dtype",
            "kv_offset_dtype",
            "q_offset_count",
            "kv_offset_count",
            "offset_layout",
            "offset_device",
            "missing_offsets",
            "kv_extent",
            "out_extent",
            "packed_shape",
            "packed_layout",
        ],
    )
    def test_packed_lse_rejections(self, failure):
        q_offsets, k_offsets = [0, 256, 768], [0, 128, 384]
        if failure == "q_tail":
            q_offsets = [0, 257, 768]
        elif failure == "kv_tail":
            k_offsets = [0, 129, 384]
        elif failure == "zero_kv":
            # The other KV length stays aligned and within the maximum.
            k_offsets = [0, 0, 256]
        elif failure == "decreasing":
            q_offsets = [0, 512, 256]
        elif failure == "first_offset":
            q_offsets = [1, 256, 768]
        elif failure == "terminal_offset":
            q_offsets = [0, 256, 512]
        elif failure == "q_above_max":
            q_offsets = [0, 0, 768]
        elif failure == "kv_above_max":
            k_offsets = [0, 64, 384]
        spec = Gfx950AttentionDenseSpec(
            batch=2,
            seqlen_q=512,
            seqlen_kv=256,
            num_query_heads=4,
            num_kv_heads=1,
            head_size=128,
            dtype="bf16",
            varlen=True,
            emit_lse=True,
        )
        tq = 256 if failure == "decreasing" else 768
        tk = 256 if failure == "zero_kv" else 384
        q = torch.zeros(tq, 4, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.zeros(tk, 1, 128, device=q.device, dtype=q.dtype)
        v = torch.zeros_like(k)
        out = torch.empty_like(q)
        lse = torch.empty(tq, 4, 1, device=q.device, dtype=torch.float32)
        cu_q = torch.tensor(q_offsets, device=q.device, dtype=torch.int32)
        cu_k = torch.tensor(k_offsets, device=q.device, dtype=torch.int32)
        if failure == "q_offset_dtype":
            cu_q = cu_q.long()
        elif failure == "kv_offset_dtype":
            cu_k = cu_k.long()
        elif failure == "q_offset_count":
            cu_q = cu_q[:-1]
        elif failure == "kv_offset_count":
            cu_k = cu_k[:-1]
        elif failure == "offset_layout":
            cu_q = torch.tensor(
                [0, 0, 256, 0, 768, 0],
                device=q.device,
                dtype=torch.int32,
            )[::2]
        elif failure == "offset_device":
            cu_k = cu_k.cpu()
        elif failure == "missing_offsets":
            cu_q = None
        elif failure == "kv_extent":
            v = v[:-1]
        elif failure == "out_extent":
            out = out[:-1]
        elif failure == "packed_shape":
            k = k.view(tk, 2, 64)
        elif failure == "packed_layout":
            k = torch.zeros(tk, 1, 256, device=q.device, dtype=q.dtype)[..., ::2]
        with pytest.raises(ValueError):
            run_attention_dense_torch(
                spec=spec,
                q=q,
                k=k,
                v=v,
                out=out,
                lse=lse,
                scale=0.5,
                cu_seqlens_q=cu_q,
                cu_seqlens_kv=cu_k,
            )

    @pytest.mark.parametrize(
        "failure",
        [
            "missing",
            "unexpected",
            "dtype",
            "shape",
            "layout",
            "device",
            "overlap_q",
            "overlap_k",
            "overlap_v",
            "overlap_out",
            "overlap_sink",
            "overlap_metadata",
            "companion",
        ],
    )
    def test_lse_buffer_errors(self, failure):
        spec = Gfx950AttentionDenseSpec(
            batch=2,
            seqlen_q=256,
            seqlen_kv=256,
            num_query_heads=4,
            num_kv_heads=1,
            head_size=128,
            dtype="bf16",
            emit_lse=True,
        )
        q = torch.zeros(2, 256, 4, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.zeros(2, 256, 1, 128, device=q.device, dtype=q.dtype)
        v = torch.zeros_like(k)
        out = torch.empty_like(q)
        lse = torch.empty(2, 4, 256, 1, device=q.device, dtype=torch.float32)
        sinks, metadata = None, {}
        if failure == "missing":
            lse = None
        elif failure == "unexpected":
            spec = replace(spec, emit_lse=False)
        elif failure == "dtype":
            lse = lse.bfloat16()
        elif failure == "shape":
            lse = lse[..., 0]
        elif failure == "layout":
            lse = torch.empty(
                2,
                4,
                256,
                2,
                device=q.device,
                dtype=torch.float32,
            )[..., :1]
        elif failure == "device":
            lse = lse.cpu()
        elif failure in {"overlap_q", "overlap_k", "overlap_v", "overlap_out"}:
            tensor = {
                "overlap_q": q,
                "overlap_k": k,
                "overlap_v": v,
                "overlap_out": out,
            }[failure]
            # Reinterpret the real input/output allocation, not a copied tensor.
            lse = tensor.view(torch.float32).view(-1)[:2048].view(2, 4, 256, 1)
        elif failure == "overlap_sink":
            spec = replace(spec, use_sinks=True)
            sinks = lse.view(torch.bfloat16).view(-1)[:4]
        elif failure == "overlap_metadata":
            spec = replace(spec, varlen=True)
            q, k, v, out = (tensor.flatten(0, 1) for tensor in (q, k, v, out))
            lse = lse.view(512, 4, 1)
            cu_q = lse.view(torch.int32).view(-1)[:3]
            cu_q.copy_(torch.tensor([0, 256, 512], device=q.device, dtype=torch.int32))
            metadata = {
                "cu_seqlens_q": cu_q,
                "cu_seqlens_kv": torch.tensor(
                    [0, 256, 512],
                    device=q.device,
                    dtype=torch.int32,
                ),
            }
        elif failure == "companion":
            k = k.cpu()
        with pytest.raises(ValueError):
            run_attention_dense_torch(
                spec=spec,
                q=q,
                k=k,
                v=v,
                out=out,
                lse=lse,
                scale=0.5,
                sinks=sinks,
                **metadata,
            )

    @pytest.mark.parametrize(
        "spec_id",
        ["gfx950_dense_grid", "gfx950_dense_persist", "gfx950_dense_persist_widedma"],
    )
    def test_dispatched_lse(self, spec_id):
        """Request -> dispatch -> bind -> launch fills a caller-owned LSE buffer."""
        from dispatch.attention import (
            AttentionRequest,
            attention_execution_candidates,
            dispatch_attention,
        )

        candidate = next(
            c for c in attention_execution_candidates() if c.spec_id == spec_id
        )
        result = dispatch_attention(
            AttentionRequest(
                batch=2,
                nhead_q=8,
                nhead_k=2,
                seqlen_q=512,
                seqlen_k=512,
                hdim_q=128,
                hdim_v=128,
                arch="gfx950",
                mask_type=1,
                dtype="bf16",
                algorithm=candidate.algorithm,
                spec_id=spec_id,
                emit_lse=True,
            )
        )
        scale = 0.5
        torch.manual_seed(23)
        q = torch.randn(2, 512, 8, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(2, 512, 2, 128, device=q.device, dtype=q.dtype)
        v = torch.randn_like(k)
        out = torch.empty_like(q)
        lse = torch.full(
            (2, 8, 512, 1), float("nan"), device=q.device, dtype=torch.float32
        )
        tensors = {"q": q, "k": k, "v": v, "out": out}
        with pytest.raises(ValueError):
            result.bind_torch(tensors, scale=scale).launch()
        result.bind_torch({**tensors, "lse": lse}, scale=scale).launch()
        torch.cuda.synchronize()
        ref, lse_ref = _lse_reference(q, k, v, scale, causal=True)
        assert (out.float() - ref).abs().max().item() < _tolerance("bf16")
        assert (lse - lse_ref).abs().max().item() < _tolerance("bf16")

    def test_lse_runtime_shape_reuse(self):
        from kernels.common.attention_dense_spec import attention_dense_cache_key
        from kernels.gfx950.attention_dense import _DENSE_LAUNCHER_CACHE

        specs = [
            Gfx950AttentionDenseSpec(
                batch=batch,
                seqlen_q=sq,
                seqlen_kv=512,
                num_query_heads=4,
                num_kv_heads=1,
                head_size=128,
                dtype="bf16",
                emit_lse=True,
            )
            for batch, sq in [(1, 512), (2, 1024)]
        ]
        enabled_keys = [
            attention_dense_cache_key(spec, arch="gfx950") for spec in specs
        ]
        off_specs = [replace(spec, emit_lse=False) for spec in specs]
        disabled_keys = [
            attention_dense_cache_key(spec, arch="gfx950") for spec in off_specs
        ]
        # Own every potentially affected slot, even if a cache-key regression
        # unexpectedly splits shape identity. Always restore prior entries.
        owned_keys = set(enabled_keys + disabled_keys)
        old_entries = {key: _DENSE_LAUNCHER_CACHE.pop(key, None) for key in owned_keys}
        launchers = []
        scale = 0.5
        try:
            for index, spec in enumerate(specs):
                torch.manual_seed(29 + index)
                batch, sq = spec.batch, spec.seqlen_q
                q = torch.randn(batch, sq, 4, 128, device="cuda", dtype=torch.bfloat16)
                k = torch.randn(batch, 512, 1, 128, device=q.device, dtype=q.dtype)
                v = torch.randn_like(k)
                out = torch.empty_like(q)
                lse = torch.full(
                    (batch, 4, sq, 1),
                    float("nan"),
                    device=q.device,
                    dtype=torch.float32,
                )
                run_attention_dense_torch(
                    spec=spec,
                    q=q,
                    k=k,
                    v=v,
                    out=out,
                    lse=lse,
                    scale=scale,
                )
                torch.cuda.synchronize()
                ref, lse_ref = _lse_reference(q, k, v, scale, causal=True)
                assert (out.float() - ref).abs().max().item() < _tolerance("bf16")
                assert (lse - lse_ref).abs().max().item() < _tolerance("bf16")
                launchers.append(_launcher_for(spec))
            assert launchers[0] is not None
            assert launchers[0] is launchers[1]
            off_out = torch.empty_like(out)
            run_attention_dense_torch(
                spec=off_specs[-1],
                q=q,
                k=k,
                v=v,
                out=off_out,
                scale=scale,
            )
            torch.cuda.synchronize()
            assert torch.equal(out, off_out)
            assert _launcher_for(off_specs[-1]) is not launchers[0]
        finally:
            for key, entry in old_entries.items():
                _DENSE_LAUNCHER_CACHE.pop(key, None)
                if entry is not None:
                    _DENSE_LAUNCHER_CACHE[key] = entry


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
