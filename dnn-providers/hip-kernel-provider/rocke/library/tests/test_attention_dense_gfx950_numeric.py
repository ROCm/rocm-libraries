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

import math

import pytest

from kernels.gfx950.attention_dense import (
    run_attention_dense_torch,
)


def _gpu_ready():
    """True only on a gfx950 box with ROCm torch. Gate on ``gcnArchName`` (the ISA
    target), NOT the marketing name."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    if not torch.cuda.is_available():
        return False
    arch = torch.cuda.get_device_properties(0).gcnArchName.lower()
    return "gfx950" in arch


requires_gfx950_gpu = pytest.mark.skipif(
    not _gpu_ready(), reason="needs a gfx950 (MI355X) GPU with ROCm torch"
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
    up here with no edit. Only ``dense_persistent`` is pinned rather than left on
    "auto": the cohort asserts BOTH grid variants at one fixed Sq.
    """
    # Imported lazily: keeps module import (and hence CPU collection of this
    # gpu-marked file) independent of the dispatch package.
    from dispatch.attention import AttentionRequest
    from dispatch.attention.gfx950 import dense_spec_for_request

    return dense_spec_for_request(
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
            algorithm="attention_dense",
            dense_persistent="on" if persistent else "off",
            use_sinks=use_sinks,
            sliding_window=sliding_window,
        )
    )


def _tolerance(dtype):
    """Numerical tolerance for max_abs error (matches gfx942).

    fp16 has tighter tolerance (2e-2) than bf16 (4e-2) due to better precision
    in the mantissa (10 bits vs 7 bits).
    """
    return 2e-2 if dtype == "fp16" else 4e-2


def _standard_reference(q, k, v, scale):
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
