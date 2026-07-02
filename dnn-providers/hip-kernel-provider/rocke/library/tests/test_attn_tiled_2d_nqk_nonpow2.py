# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for the padded-block-q / non-pow2 NQK support in tiled-2D attention.

These tests are pure Python (no GPU / compilation required) and cover:

  - ``supports_tiled_2d`` now accepts NQK values that do not divide BLOCK_M
    (previously rejected with a divisibility error).
  - ``spec.block_q`` (= BLOCK_M // NQK, floor) and
    ``VALID_ROWS = block_q * NQK`` are correct for non-pow2 ratios:
    VALID_ROWS < BLOCK_M so the extra rows are masked.
  - Power-of-2 / exact-divisor NQK baseline: VALID_ROWS == BLOCK_M
    (no masking needed, regression guard).
  - Both gfx942 and gfx950 spec classes agree on these invariants.
"""

from __future__ import annotations

import importlib

import pytest

# ---------------------------------------------------------------------------
# Per-arch config: smallest (num_warps, block_m_per_warp) that passes
# supports_tiled_2d for D128/BS64 bf16 without exceeding the LDS budget.
# gfx942 narrow path: nw=2, mw=16 -> BLOCK_M=32
# gfx950 narrow path: nw=8, mw=16 -> BLOCK_M=128 (larger LDS budget)
# ---------------------------------------------------------------------------

_ARCH_CFG = {
    "gfx942": dict(num_warps=2, block_m_per_warp=16),  # BLOCK_M=32
    "gfx950": dict(num_warps=8, block_m_per_warp=16),  # BLOCK_M=128
}

ARCHES = list(_ARCH_CFG.keys())


def _import(arch: str):
    mod = importlib.import_module(f"kernels.{arch}.attention_tiled_2d")
    return mod.supports_tiled_2d, mod.UnifiedAttention2DTiledSpec


def _supports(arch: str, *, nqk: int, **kw) -> tuple:
    fn, _ = _import(arch)
    cfg = _ARCH_CFG[arch]
    defaults = dict(
        head_size=128,
        block_size=64,
        dtype="bf16",
        use_alibi=False,
        use_qq_bias=False,
        use_fp8=False,
        q_dtype="bf16",
        tile_size=64,
        arch=arch,
    )
    defaults.update(cfg)
    defaults.update(kw)
    return fn(num_queries_per_kv=nqk, **defaults)


def _spec(arch: str, *, hq: int, hkv: int, **kw):
    _, SpecCls = _import(arch)
    cfg = _ARCH_CFG[arch]
    defaults = dict(
        head_size=128,
        block_size=64,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_seqs=1,
        tile_size=64,
    )
    defaults.update(cfg)
    defaults.update(kw)
    return SpecCls(num_query_heads=hq, num_kv_heads=hkv, **defaults)


# ---------------------------------------------------------------------------
# supports_tiled_2d acceptance tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("nqk", [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15])
def test_supports_nonpow2_nqk(arch, nqk):
    """supports_tiled_2d must accept NQK values that don't divide BLOCK_M."""
    ok, reason = _supports(arch, nqk=nqk)
    assert ok, f"arch={arch} NQK={nqk} rejected: {reason}"


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("nqk", [1, 2, 4, 8, 16])
def test_supports_pow2_nqk(arch, nqk):
    """Power-of-2 NQK values must still be accepted (regression guard)."""
    ok, reason = _supports(arch, nqk=nqk)
    assert ok, f"arch={arch} NQK={nqk} rejected: {reason}"


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("nqk", [0, 17, 32])
def test_rejects_out_of_range_nqk(arch, nqk):
    """NQK outside [1, 16] must still be rejected."""
    ok, _ = _supports(arch, nqk=nqk)
    assert not ok, f"arch={arch} NQK={nqk} should be rejected but was accepted"


# ---------------------------------------------------------------------------
# VALID_ROWS = block_q * NQK invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize(
    "nqk",
    [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
    ids=lambda n: f"nqk{n}",
)
def test_valid_rows_less_than_block_m_for_nonpow2(arch, nqk):
    """For NQK that doesn't divide BLOCK_M, VALID_ROWS must be < BLOCK_M.

    This is the key correctness invariant: the extra rows padded into the CTA
    by the padded-block-q fix must be masked out before writing output.
    """
    hkv = 8
    hq = nqk * hkv
    spec = _spec(arch, hq=hq, hkv=hkv)
    block_m = spec.block_m
    block_q = spec.block_q
    valid_rows = block_q * nqk

    if block_m % nqk == 0:
        pytest.skip(
            f"NQK={nqk} happens to divide BLOCK_M={block_m} for this arch config"
        )

    assert valid_rows < block_m, (
        f"arch={arch} NQK={nqk} BLOCK_M={block_m} BLOCK_Q={block_q} "
        f"VALID_ROWS={valid_rows} should be < BLOCK_M"
    )


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize(
    "nqk",
    [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15],
    ids=lambda n: f"nqk{n}",
)
def test_block_q_is_floor_div(arch, nqk):
    """block_q == block_m // nqk (floor division)."""
    hkv = 8
    hq = nqk * hkv
    spec = _spec(arch, hq=hq, hkv=hkv)
    block_m = spec.block_m
    expected = block_m // nqk
    assert spec.block_q == expected, (
        f"arch={arch} NQK={nqk} BLOCK_M={block_m} "
        f"block_q={spec.block_q} != floor({block_m}/{nqk})={expected}"
    )


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize(
    "hq,hkv",
    [(8, 8), (16, 8), (32, 8), (64, 8)],
    ids=lambda p: f"nqk{p[0]//p[1]}" if isinstance(p, tuple) else str(p),
)
def test_valid_rows_equals_block_m_for_divisor_nqk(arch, hq, hkv):
    """When NQK divides BLOCK_M, VALID_ROWS == BLOCK_M (no masking needed)."""
    nqk = hq // hkv
    spec = _spec(arch, hq=hq, hkv=hkv)
    block_m = spec.block_m
    if block_m % nqk != 0:
        pytest.skip(f"NQK={nqk} doesn't divide BLOCK_M={block_m} for this arch config")
    valid_rows = spec.block_q * nqk
    assert valid_rows == block_m, (
        f"arch={arch} NQK={nqk} BLOCK_M={block_m} BLOCK_Q={spec.block_q} "
        f"VALID_ROWS={valid_rows} should equal BLOCK_M"
    )


# ---------------------------------------------------------------------------
# Concrete repro cases from gqa_nqk_nonpow2_repro_fast.jsonl
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arch,hq,hkv,nw,mw",
    [
        # gfx942 narrow path (num_warps=2 passes LDS check for D128/BS64)
        ("gfx942", 40, 8, 2, 16),  # NQK=5, BLOCK_M=32
        ("gfx942", 28, 4, 2, 16),  # NQK=7, BLOCK_M=32
        ("gfx942", 24, 8, 2, 16),  # NQK=3, BLOCK_M=32
        # gfx950 (larger LDS budget allows BLOCK_M=128)
        ("gfx950", 40, 8, 8, 16),  # NQK=5, BLOCK_M=128
        ("gfx950", 28, 4, 8, 16),  # NQK=7, BLOCK_M=128
        ("gfx950", 24, 8, 8, 16),  # NQK=3, BLOCK_M=128
    ],
    ids=lambda x: str(x),
)
def test_repro_cases_supported_and_valid_rows(arch, hq, hkv, nw, mw):
    """The exact shapes from the non-pow2 repro are accepted and have correct VALID_ROWS."""
    fn, SpecCls = _import(arch)
    nqk = hq // hkv

    ok, reason = fn(
        head_size=128,
        block_size=64,
        dtype="bf16",
        num_queries_per_kv=nqk,
        use_alibi=False,
        use_qq_bias=False,
        use_fp8=False,
        q_dtype="bf16",
        num_warps=nw,
        block_m_per_warp=mw,
        tile_size=64,
        arch=arch,
    )
    assert ok, f"arch={arch} NQK={nqk} nw={nw} mw={mw} rejected: {reason}"

    spec = SpecCls(
        head_size=128,
        block_size=64,
        dtype="bf16",
        num_query_heads=hq,
        num_kv_heads=hkv,
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_seqs=1,
        tile_size=64,
        num_warps=nw,
        block_m_per_warp=mw,
    )
    block_m = spec.block_m
    block_q = spec.block_q
    valid_rows = block_q * nqk

    assert (
        block_q == block_m // nqk
    ), f"arch={arch} NQK={nqk}: block_q={block_q} != floor({block_m}/{nqk})"
    assert (
        valid_rows <= block_m
    ), f"arch={arch} NQK={nqk}: VALID_ROWS={valid_rows} > BLOCK_M={block_m}"
    if block_m % nqk != 0:
        assert (
            valid_rows < block_m
        ), f"arch={arch} NQK={nqk}: VALID_ROWS={valid_rows} should be < BLOCK_M={block_m}"
