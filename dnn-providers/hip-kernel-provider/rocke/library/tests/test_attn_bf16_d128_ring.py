# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression tests for the gfx942 D128 sliced-K ring exclusion.

The sliced-K ring (32-wide K slices -> k_groups = HD/32) is correctness-verified
only for D64 (k_groups=2). The D128 ring (k_groups=4) is numerically WRONG at
realistic input magnitude -- max_abs ~0.5-1.3 vs the fp32 oracle for BOTH bf16
and fp16, GQA and MHA, even at S512 when the ring is forced -- while the non-ring
D128 flash path (nw2/nw4, T=64) passes at max_abs ~0.0156 (bf16) / ~0.002 (fp16).

#9198 briefly enabled the D128 bf16 ring on the strength of a "max_abs 0.00049"
check, but that check used a uniform_(-0.1, 0.1) oracle whose near-uniform softmax
never exercises the online-softmax running-max rescale across the four K slices,
so the defect was masked. It reproduces immediately with randn (unit-variance)
inputs (study s34_gfx942_bf16_d128_prefill_ring_correctness_regression). The fix
excludes ALL D128 (both dtypes) from the ring until the k_groups=4 sliced-K
accumulation is fixed in attention_tiled_2d.py.

Two layers:
  * Pure-Python spec assertions (always run): D128 ring OFF, geometry sane,
    D64 ring unchanged.
  * On-GPU numeric guardrail (skipped without a gfx942 device + torch): launch
    the production kernel at randn magnitude and assert max_abs vs fp32 SDPA.
    A spec-only test cannot catch a numerics regression in the emitted kernel --
    that is exactly what let #9198 through.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from kernels import UnifiedAttentionProblem
from kernels.common import attention_unified as au


@pytest.fixture
def gfx942(monkeypatch):
    # Set the memoized arch global directly (same pattern as
    # test_gfx1250_attention.py) rather than monkeypatching the resolver: the
    # resolver caches into _RESOLVED_ATTENTION_ARCH, so replacing only the
    # function leaves the cached global stale and leaks into sibling tests.
    old_arch = au._RESOLVED_ATTENTION_ARCH
    au._RESOLVED_ATTENTION_ARCH = "gfx942"
    # Neutralize any inherited env overrides so we test the default policy.
    for var in (
        "HIPDNN_GFX942_K_SLICED_RING",
        "HIPDNN_GFX942_BF16_WIDE",
        "HIPDNN_GFX942_D128_SMALLTILE_DK",
        "HIPDNN_GFX942_FLASH_MLIM",
        "HIPDNN_GFX942_FLASH_WIDE",
        "HIPDNN_GFX942_K_LDSSEQ",
    ):
        monkeypatch.delenv(var, raising=False)
    try:
        yield
    finally:
        au._RESOLVED_ATTENTION_ARCH = old_arch
        au._2D_LAUNCH_META.clear()


def _problem(dtype, sq=4096, hq=32, hk=8, d=128, bs=64):
    return UnifiedAttentionProblem(
        total_q=sq,
        num_seqs=1,
        num_query_heads=hq,
        num_kv_heads=hk,
        head_size=d,
        block_size=bs,
        max_seqlen_q=sq,
        max_seqlen_k=sq,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Pure-Python spec assertions (no GPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("hq,hk", [(32, 8), (16, 16), (64, 8), (128, 8), (64, 4)])
def test_d128_excluded_from_ring(gfx942, dtype, hq, hk):
    # The k_groups=4 D128 ring is numerically wrong; it must stay OFF for both
    # dtypes and every geometry.
    p = _problem(dtype, hq=hq, hk=hk)
    assert not au._enable_gfx942_flash_k_sliced_ring(
        p
    ), "D128 ring is numerically wrong (k_groups=4) and must be excluded"
    spec = au._tiled_spec_from_problem(p)
    assert not spec.use_k_sliced_ring
    assert spec.tile_size == 64


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_d128_ring_force_on_still_off(gfx942, monkeypatch, dtype):
    # Even the explicit force-on env must NOT re-enable the D128 ring: the
    # head_size==128 guard short-circuits before the env is consulted.
    monkeypatch.setenv("HIPDNN_GFX942_K_SLICED_RING", "1")
    p = _problem(dtype)
    assert not au._enable_gfx942_flash_k_sliced_ring(p)


@pytest.mark.parametrize("hq,hk", [(32, 8), (16, 16), (64, 8), (128, 8), (64, 4)])
def test_d128_launch_meta_matches_nonring_spec(gfx942, hq, hk):
    # The launch grid/block must match the (non-ring) geometry the launcher
    # actually builds -- a mismatch silently corrupts output.
    au._2D_LAUNCH_META.clear()
    p = _problem("bf16", hq=hq, hk=hk)
    spec = au._tiled_spec_from_problem(p)
    meta = au._get_2d_launch_meta(p, au._tiled_cache_key(p))
    assert meta.block[0] == 64 * spec.num_warps


def test_d64_ring_unchanged(gfx942):
    # D64 (k_groups=2) is correctness-verified and keeps the ring.
    for dtype in ("bf16", "fp16"):
        s = au._tiled_spec_from_problem(_problem(dtype, d=64, bs=16))
        assert s.use_k_sliced_ring, f"D64 {dtype} must keep the ring"
        assert s.num_warps == 4


# ---------------------------------------------------------------------------
# On-GPU numeric guardrail -- the check #9198 lacked
# ---------------------------------------------------------------------------


def _load_parity_harness():
    import sys

    here = Path(__file__).resolve()
    # rocke/library/tests -> rocke/library/builders/gfx942/attention/prefill
    harness = (
        here.parents[1]
        / "builders/gfx942/attention/prefill/parity_unified_attention.py"
    )
    if not harness.exists():
        return None
    spec = importlib.util.spec_from_file_location("parity_unified_attention", harness)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses in the harness (e.g. Shape) can resolve
    # cls.__module__ during field introspection.
    sys.modules["parity_unified_attention"] = mod
    spec.loader.exec_module(mod)
    return mod


def _gpu_ready():
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    if not torch.cuda.is_available():
        return False
    name = torch.cuda.get_device_name(0).lower()
    return "mi300" in name or "gfx942" in name


requires_gfx942_gpu = pytest.mark.skipif(
    not _gpu_ready(), reason="needs a gfx942 (MI300X) GPU with ROCm torch"
)


@requires_gfx942_gpu
@pytest.mark.parametrize("dtype,tol", [("bf16", 5e-2), ("fp16", 5e-2)])
@pytest.mark.parametrize("hq,hk", [(32, 8), (16, 16)])
@pytest.mark.parametrize("sq", [512, 1024, 4096])
def test_d128_numeric_vs_fp32_oracle_at_magnitude(dtype, tol, hq, hk, sq):
    """Launch the production D128 kernel at realistic (unit-variance) magnitude
    and assert max_abs vs the fp32 paged-attn oracle.

    This is the guardrail #9198 lacked: with the default uniform_(-0.1, 0.1)
    inputs the ring bug is invisible (softmax ~uniform); randn inputs expose it
    immediately. If a future change re-enables the broken D128 ring, this fails.
    """
    import torch

    H = _load_parity_harness()
    if H is None:
        pytest.skip("parity harness not present in this checkout")

    # Force the arch resolver so the production selector picks the gfx942 path.
    old_arch = au._RESOLVED_ATTENTION_ARCH
    au._RESOLVED_ATTENTION_ARCH = "gfx942"
    for var in (
        "HIPDNN_GFX942_K_SLICED_RING",
        "HIPDNN_GFX942_FLASH_WIDE",
        "HIPDNN_GFX942_FLASH_MLIM",
    ):
        os.environ.pop(var, None)
    try:
        from builders.common.attention_spec_builder import _tiled_spec_from_problem

        s = H.Shape(
            f"d128_{hq}_{hk}_S{sq}", dtype, sq, sq, 128, hq, hk, 1, True, "perf"
        )

        # randn (unit-variance) inputs -- NOT the uniform_(-0.1,0.1) default that
        # masked the ring bug.
        torch.manual_seed(0)
        td = torch.float16 if dtype == "fp16" else torch.bfloat16
        BS = H.BLOCK_SIZE
        max_blk = (sq + BS - 1) // BS
        query = torch.randn(sq, hq, 128, device="cuda").to(td)
        key_cache = torch.randn(max_blk, BS, hk, 128, device="cuda").to(td)
        value_cache = torch.randn(max_blk, BS, hk, 128, device="cuda").to(td)
        cu_q = torch.tensor([0, sq], dtype=torch.int32, device="cuda")
        kv_lens = torch.tensor([sq], dtype=torch.int32, device="cuda")
        block_tables = torch.arange(max_blk, dtype=torch.int32, device="cuda").view(
            1, max_blk
        )
        scale = 128**-0.5
        data = {
            "query": query,
            "key_cache": key_cache,
            "value_cache": value_cache,
            "cu_q": cu_q,
            "kv_lens": kv_lens,
            "query_lens": [sq],
            "kv_lens_list": [sq],
            "block_tables": block_tables,
            "scale": scale,
            "max_query_len": sq,
            "max_kv_len": sq,
        }
        ref = H.ref_paged_attn(
            query, key_cache, value_cache, [sq], [sq], block_tables, scale
        )

        p = _problem(dtype, sq=sq, hq=hq, hk=hk)
        spec = _tiled_spec_from_problem(p)
        assert not spec.use_k_sliced_ring, "production D128 spec must not use the ring"

        from rocke import compile_kernel
        from kernels import build_unified_attention_2d_tiled
        from kernels.common.attention_unified import _attn_signature
        from rocke.runtime import KernelLauncher

        kernel = build_unified_attention_2d_tiled(spec, arch="gfx942")
        artifact = compile_kernel(kernel, arch="gfx942", capture_ir_text=False)
        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=_attn_signature(
                dtype, include_bt_stride=True, include_qq_bias_stride=True
            ),
            cache_key=("d128guard", spec.kernel_name(), dtype, sq, hq, hk),
        )
        out, _ = H._run_rocke(s, data, launcher, spec, warmup=3, attempts=3)
        m = H.compare(ref, out)
        assert m["max_abs"] <= tol, (
            f"D128 {dtype} {hq}/{hk} S{sq}: max_abs {m['max_abs']:.4f} > {tol} "
            f"(ring regression?)"
        )
    finally:
        au._RESOLVED_ATTENTION_ARCH = old_arch
