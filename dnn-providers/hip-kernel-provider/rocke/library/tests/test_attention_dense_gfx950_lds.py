# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LDS-budget gate for the gfx950 dense attention kernel.

``supports_attention_dense`` rejects specs whose static LDS exceeds the gfx950
capacity. These tests pin the closed-form byte count against the allocation the
builder really emits -- for every tile the catalog ships and for non-default
``lds_*_pad`` values -- check that ``gfx950_dense_tile_legality`` agrees with the
gate, and compare the Python model with the C++ ``Gfx950AttentionDenseGeometry.hpp``
that restates it.

Pure text lowering -- no GPU and no comgr required.
"""

import re
from pathlib import Path

import pytest

from kernels.gfx950 import attention_dense as dense
from kernels.gfx950.attention_dense import AttentionDenseSpec
from rocke.core.arch import ArchTarget

from test_attention_dense_d64_lds import _lds_pool_bytes

# tests -> library -> rocke -> hip-kernel-provider
_GEOMETRY_HEADER = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "engines"
    / "kernel_ingestor_engine"
    / "packs"
    / "Gfx950AttentionDenseGeometry.hpp"
)


def _spec(head_size, block_m, block_n, **kw):
    defaults = dict(
        batch=1,
        seqlen_q=512,
        seqlen_kv=512,
        num_query_heads=64,
        num_kv_heads=8,
        head_size=head_size,
        causal=True,
        dtype="bf16",
        block_m=block_m,
        block_n=block_n,
        persistent=False,
    )
    defaults.update(kw)
    return AttentionDenseSpec(**defaults)


def _formula(spec):
    return dense.gfx950_dense_static_lds_bytes(
        head_size=spec.head_size,
        block_n=spec.block_n,
        k_row_pad=spec.lds_k_row_pad,
        v_row_pad=spec.lds_v_row_pad,
        k_group_pad=spec.lds_k_group_pad,
        wide_lds_dma=spec.wide_lds_dma,
    )


_WITHIN_BUDGET = [
    (128, 128, 64, {}),
    (128, 128, 64, {"lds_v_row_pad": 64}),
    (128, 128, 64, {"lds_k_row_pad": 24}),
    (64, 128, 128, {}),
    (64, 128, 128, {"lds_k_group_pad": 40}),
    # wide_lds_dma=True uses a different layout (K/V slab lines of 64*8 elems)
    # and should produce a SMALLER footprint than the non-wide D128/BN64 path.
    # Non-wide: 2*64*(128+8 + 128+32)*2 = 75,776 B; wide: 68,096 B.
    # persistent=True required by __post_init__ when wide_lds_dma=True.
    (128, 128, 64, {"wide_lds_dma": True, "persistent": True}),
]

# Over-budget layouts that the default-pad formula used to accept.
_OVER_BUDGET = [
    (64, 256, 256, {"lds_k_group_pad": 72}),
    (128, 256, 128, {"lds_v_row_pad": 64}),
    (128, 256, 128, {"lds_k_row_pad": 64}),
]


@pytest.mark.parametrize("head_size,block_m,block_n,pads", _WITHIN_BUDGET)
def test_formula_matches_real_lds_pool(head_size, block_m, block_n, pads):
    spec = _spec(head_size, block_m, block_n, **pads)
    ok, reason = dense.supports_attention_dense(spec)
    assert ok, reason
    assert _formula(spec) == _lds_pool_bytes(spec)


_LEGAL_TILES = [
    (head_size, block_m, block_n)
    for head_size in (64, 128)
    for block_m, block_n in dense.gfx950_dense_tile_legality(head_size=head_size)
]


@pytest.mark.parametrize("head_size,block_m,block_n", _LEGAL_TILES)
def test_every_legal_tile_fits_and_matches_real_lds_pool(head_size, block_m, block_n):
    """Includes the tiles the LDS rule turns on: D128/BN128 (151,552 B) and
    D64/BM256/BN256 (135,168 B)."""
    spec = _spec(head_size, block_m, block_n)
    ok, reason = dense.supports_attention_dense(spec)
    assert ok, reason
    pool = _lds_pool_bytes(spec)
    assert pool == _formula(spec)
    assert pool <= dense.GFX950_LDS_CAPACITY_BYTES


@pytest.mark.parametrize("head_size,block_m,block_n,pads", _OVER_BUDGET)
def test_over_budget_pads_are_rejected(head_size, block_m, block_n, pads):
    spec = _spec(head_size, block_m, block_n, **pads)
    assert _formula(spec) > dense.GFX950_LDS_CAPACITY_BYTES
    ok, reason = dense.supports_attention_dense(spec)
    assert not ok
    assert "LDS" in reason


@pytest.mark.parametrize("head_size", [64, 128])
def test_gate_accepts_exactly_the_legal_tiles(head_size):
    """The gate (spec validation + shared preflight) and gfx950_dense_tile_legality
    state the structural tile rules separately; they must pick the same tiles out of
    every block_m/block_n the spec could name. The LDS term they share is pinned
    against the real pool above."""
    legal = set(dense.gfx950_dense_tile_legality(head_size=head_size))
    # lcm of every block_n below, so no tile is refused for its extents alone.
    extent = 256 * 3 * 5 * 7
    accepted = {
        (block_m, block_n)
        for block_m in (128, 256)
        for block_n in range(32, 257, 32)
        if dense.supports_attention_dense(
            _spec(head_size, block_m, block_n, seqlen_q=extent, seqlen_kv=extent)
        )[0]
    }
    assert accepted == legal
    assert accepted, "no legal tile"


def _header_constants(text):
    """Every ``constexpr int64_t NAME = <integer>;`` in the C++ geometry header."""
    return {
        name: int(value)
        for name, value in re.findall(
            r"constexpr\s+int64_t\s+(\w+)\s*=\s*(\d+)\s*;", text
        )
    }


def _cpp_supported_tiles(c, head_size):
    """``isSupportedGfx950AttentionDenseTile`` evaluated over the constants parsed
    from the header, for every ``(block_m, block_n)`` up to the largest query tile."""

    def lds_bytes(block_n):
        if head_size == 128:
            return (
                c["BUFFERS"]
                * block_n
                * ((head_size + c["K_PAD"]) + (head_size + c["V_PAD"]))
                * c["ELEMENT_BYTES"]
            )
        k_bytes = (
            c["BUFFERS"]
            * (block_n // c["ROWS_PER_GROUP"])
            * (c["ROWS_PER_GROUP"] * head_size + c["K_PAD"])
        )
        v_bytes = c["BUFFERS"] * block_n * head_size
        return (k_bytes + v_bytes) * c["ELEMENT_BYTES"]

    tiles = set()
    quantum = c["GFX950_ATTENTION_DENSE_BLOCK_N_QUANTUM"]
    for block_m in (
        c["GFX950_ATTENTION_DENSE_BLOCK_M_SMALL"],
        c["GFX950_ATTENTION_DENSE_BLOCK_M_LARGE"],
    ):
        num_waves = block_m // c["GFX950_ROWS_PER_WAVE"]
        for block_n in range(quantum, block_m + 1, quantum):
            if block_m % block_n or block_n % num_waves:
                continue
            if lds_bytes(block_n) <= c["GFX950_LDS_CAPACITY_BYTES"]:
                tiles.add((block_m, block_n))
    return tiles


@pytest.fixture(scope="module")
def cpp_constants():
    if not _GEOMETRY_HEADER.is_file():
        pytest.skip("C++ sources are not part of this tree")
    return _header_constants(_GEOMETRY_HEADER.read_text(encoding="utf-8"))


def test_cpp_lds_capacity_is_the_gfx950_arch_capacity(cpp_constants):
    capacity = ArchTarget.from_gfx("gfx950").lds_capacity_bytes
    assert dense.GFX950_LDS_CAPACITY_BYTES == capacity
    assert cpp_constants["GFX950_LDS_CAPACITY_BYTES"] == capacity


def test_cpp_lds_pads_match_the_python_layout_defaults(cpp_constants):
    """The header restates one K pad (per row at D128, per row-group at D64), one V
    pad, the buffer count and the element size at the layout every shipped variant is
    built with."""
    defaults = _spec(128, 128, 64)
    assert cpp_constants["K_PAD"] == defaults.lds_k_row_pad
    assert cpp_constants["K_PAD"] == defaults.lds_k_group_pad
    assert cpp_constants["V_PAD"] == defaults.lds_v_row_pad
    assert cpp_constants["ELEMENT_BYTES"] == 2  # bf16/fp16
    k_row = 128 + cpp_constants["K_PAD"]
    v_row = 128 + cpp_constants["V_PAD"]
    assert dense.gfx950_dense_static_lds_bytes(head_size=128, block_n=1) == (
        cpp_constants["BUFFERS"] * (k_row + v_row) * cpp_constants["ELEMENT_BYTES"]
    )


def test_cpp_block_m_tiles_are_the_python_block_m_tiles(cpp_constants):
    assert {
        cpp_constants["GFX950_ATTENTION_DENSE_BLOCK_M_SMALL"],
        cpp_constants["GFX950_ATTENTION_DENSE_BLOCK_M_LARGE"],
    } == {int(g["block_m"]) for g in dense.DENSE_TILE_GEOMETRIES.values()}


@pytest.mark.parametrize("head_size", [64, 128])
def test_cpp_legal_tiles_are_the_python_legal_tiles(cpp_constants, head_size):
    assert _cpp_supported_tiles(cpp_constants, head_size) == set(
        dense.gfx950_dense_tile_legality(head_size=head_size)
    )
