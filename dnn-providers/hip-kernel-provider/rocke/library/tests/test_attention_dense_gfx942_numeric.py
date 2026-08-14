# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU numeric lane for the gfx942 dense flash-attn kernel.

Drives the PUBLIC entry point ``run_attention_dense_torch`` end-to-end on a real
gfx942 GPU and checks max_abs against an fp32 ``scaled_dot_product_attention``
oracle, for both the default and the P4 persistent grid. This is the committed,
CI-collectable form of the acceptance criterion's "functional GPU-numeric bench,
both variants, vs torch fp32 SDPA" -- previously only an out-of-tree verifier.

Specs come from the gfx942 DISPATCH factory, never hand-rolled (see :func:`_spec`),
so every row compiles the binary that actually ships rather than one that differs
from it by an untracked tuning default.

Every test is marked ``gpu`` and gated with a device skipif, so it is a graceful
skip on a CPU CI box and only executes on a gfx942 (MI300X) ROCm runner. Select
it with ``run_all.py --gpu`` (or ``pytest -m gpu``); the default CPU lane excludes
it via ``-m "not gpu"``. Run standalone:

    HIP_VISIBLE_DEVICES=0 python -m pytest tests/test_attention_dense_gfx942_numeric.py
"""

from __future__ import annotations

import math

import pytest

from kernels.gfx942.attention_dense import run_attention_dense_torch


def _gpu_ready():
    """True only on a gfx942 box with ROCm torch. Gate on ``gcnArchName`` (the ISA
    target), NOT the marketing name: the whole MI300 family is gfx942, but the
    marketing string varies (``MI300X``/``MI300A``/``MI308X``) and a substring check
    for ``"mi300"`` silently MISSES ``MI308X`` -- the exact skip that hid this lane on
    the first run. The arch string is stable across the family."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    if not torch.cuda.is_available():
        return False
    arch = torch.cuda.get_device_properties(0).gcnArchName.lower()
    return "gfx942" in arch


requires_gfx942_gpu = pytest.mark.skipif(
    not _gpu_ready(), reason="needs a gfx942 (MI300X) GPU with ROCm torch"
)

_TORCH_DT = {"fp16": "float16", "bf16": "bfloat16"}

# (dtype, head_size, num_query_heads, num_kv_heads, persistent) -- a compact cohort
# spanning both dtypes, D64/D128, GQA + MHA, and both grid variants. Sq is fixed at
# 512 (a 256 multiple so the persistent grid-stride has >1 q-block of work).
_COHORT = [
    ("fp16", 128, 16, 4, False),  # flagship default
    ("fp16", 128, 16, 4, True),  # flagship persistent
    ("bf16", 128, 16, 4, True),  # bf16 D128 persistent (the VGPR-starved config)
    ("fp16", 64, 16, 16, False),  # D64 MHA default
    ("bf16", 64, 16, 4, True),  # D64 bf16 persistent (the wpe=4 config)
]


def _spec(dtype, d, hq, hkv, persistent, *, batch=1, sq=512):
    """The SHIPPED gfx942 dense spec for a cohort row, built through the dispatch
    factory (``dispatch.attention.gfx942._dense_spec``) rather than hand-rolled.

    Hand-rolling the spec silently pins every tuned lever to the shared (gfx950)
    dataclass default, so the lane would assert on configs that do not ship. The
    concrete one this cohort hit: ``waves_per_eu``. Dispatch resolves it from the
    kernel's own policy (``_tuned_waves_per_eu``), which returns 4 for bf16/D64 --
    the row ``_COHORT`` above labels "the wpe=4 config" -- while the dataclass default
    is 2. That is not a cosmetic difference: ``waves_per_eu`` is emitted as the
    ``amdgpu-waves-per-eu`` attribute, changes register allocation, and is tagged into
    ``gfx942_kernel_name`` as ``wpe{N}``, so wpe2 and wpe4 are DIFFERENT binaries.
    ``num_persistent`` was likewise hard-coded to 304 beside a dispatch constant that
    already resolves to 304 -- left at the request default here so ``_dense_spec``
    substitutes the gfx942 CU count itself and the two cannot drift apart.

    Deriving the spec from the factory (the pattern
    ``test_attention_dense_gfx942_golden.py::mk_dispatch`` uses for its D64 cases)
    also means a future gfx942 tuning change is picked up here with no edit.

    Only ``dense_persistent`` is pinned rather than left on "auto": the cohort asserts
    BOTH grid variants at one fixed Sq, where "auto" would pick a single one. Every
    other lever -- block_n, the D64 K row-group pad, persist_decode, ragged -- is
    whatever the shipped path folds in.
    """
    # Imported lazily, mirroring the golden sibling: keeps module import (and hence
    # CPU collection of this gpu-marked file) independent of the dispatch package.
    from dispatch.attention import AttentionRequest
    from dispatch.attention.gfx942 import _dense_spec

    return _dense_spec(
        AttentionRequest(
            batch=batch,
            nhead_q=hq,
            nhead_k=hkv,
            seqlen_q=sq,
            seqlen_k=sq,
            hdim_q=d,
            hdim_v=d,
            arch="gfx942",
            mask_type=1,  # causal
            dtype=dtype,
            algorithm="attention_dense",
            dense_persistent="on" if persistent else "off",
        )
    )


@requires_gfx942_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("dtype,d,hq,hkv,persistent", _COHORT)
def test_dense_numeric_vs_fp32_sdpa(dtype, d, hq, hkv, persistent):
    import torch
    import torch.nn.functional as F

    tol = 2e-2 if dtype == "fp16" else 4e-2
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

    # fp32 SDPA oracle in [B,H,S,D] layout. GQA is expanded HERE, by repeating each
    # kv head to its query heads, rather than via the ``enable_gqa=`` kwarg: that
    # kwarg is a recent addition to ``scaled_dot_product_attention``, and on an older
    # ROCm torch passing it raises TypeError -- which ERRORS the whole gpu cohort
    # instead of leaving it to the device gate. ``repeat_interleave`` along the head
    # axis is exactly what ``enable_gqa`` does internally, and it is the mapping the
    # kernel itself uses (hkv = hq // gqa), so the asserted reference is unchanged.
    # rep == 1 (the MHA row) makes it a plain copy, matching the old
    # ``enable_gqa=(hkv != hq)`` no-op.
    rep = hq // hkv
    qf = q.transpose(1, 2).float()
    kf = k.transpose(1, 2).float().repeat_interleave(rep, dim=1)
    vf = v.transpose(1, 2).float().repeat_interleave(rep, dim=1)
    ref = F.scaled_dot_product_attention(
        qf, kf, vf, is_causal=True, scale=scale
    ).transpose(
        1, 2
    )  # -> [B,S,Hq,D]

    max_abs = (ref - out.float()).abs().max().item()
    assert max_abs < tol, (
        f"{dtype} D{d} GQA{hq}/{hkv} "
        f"{'persist' if persistent else 'default'}: max_abs={max_abs:.3e} >= {tol}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
