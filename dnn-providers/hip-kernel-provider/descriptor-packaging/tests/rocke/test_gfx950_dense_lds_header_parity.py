# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The C++ ``Gfx950AttentionDenseGeometry.hpp`` restates the Python LDS model.

The header's constants and its ``isSupportedGfx950AttentionDenseTile`` rule are
compared here with ``kernels.gfx950.attention_dense``, so a change to one side
without the other fails. The header is a source file of this tree, so the check
runs in the source-tree lane and fails when the header is absent.
"""

import re
from pathlib import Path

import pytest
from kernels.gfx950 import attention_dense as dense
from kernels.gfx950.attention_dense import AttentionDenseSpec
from rocke.core.arch import ArchTarget

pytestmark = pytest.mark.quick

# tests/rocke -> tests -> descriptor-packaging -> hip-kernel-provider
_GEOMETRY_HEADER = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "engines"
    / "kernel_ingestor_engine"
    / "packs"
    / "Gfx950AttentionDenseGeometry.hpp"
)


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
    assert _GEOMETRY_HEADER.is_file(), f"geometry header missing: {_GEOMETRY_HEADER}"
    return _header_constants(_GEOMETRY_HEADER.read_text(encoding="utf-8"))


def test_cpp_lds_capacity_is_the_gfx950_arch_capacity(cpp_constants):
    capacity = ArchTarget.from_gfx("gfx950").lds_capacity_bytes
    assert dense.GFX950_LDS_CAPACITY_BYTES == capacity
    assert cpp_constants["GFX950_LDS_CAPACITY_BYTES"] == capacity


def test_cpp_lds_pads_match_the_python_layout_defaults(cpp_constants):
    """The header restates one K pad (per row at D128, per row-group at D64), one V
    pad, the buffer count and the element size at the layout every shipped variant is
    built with."""
    defaults = AttentionDenseSpec(
        batch=1,
        seqlen_q=512,
        seqlen_kv=512,
        num_query_heads=64,
        num_kv_heads=8,
        head_size=128,
        causal=True,
        dtype="bf16",
        block_m=128,
        block_n=64,
        persistent=False,
    )
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
