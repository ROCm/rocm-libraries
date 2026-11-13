# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run the emitted group search on headers, without allocating the matrices."""

import numpy as np
import pytest

from gpu_test_helpers import (
    GFX_TARGET,
    assemble_kernel,
    generate_kernel_asm,
    generate_load_params,
    hip,
    init_rocisa,
    run_on_gpu,
)
from rocisa.code import Label
from rocisa.container import ContinuousRegister
from rocisa.enum import RegisterType
from rocisa.register import RegisterPool
from tensilelite.KernelWriterAssembly import KernelWriterAssembly

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu,
    pytest.mark.skipif(
        hip is None or GFX_TARGET not in ("gfx942", "gfx950"),
        reason="requires HIP and a wave-search target (gfx942 or gfx950)",
    ),
]

TILE = 16


def search_kernel(chunks):
    init_rocisa()
    writer = object.__new__(KernelWriterAssembly)
    writer.sgprs = {"WorkGroup0": 24, "GSU": 23}
    # Keep v0 (the thread ID) out of the search's temporary register block.
    writer.vgprPool = RegisterPool(
        4, RegisterType.Vgpr, defaultPreventOverflow=False, printRP=False
    )
    kernel = {
        "MacroTile0": TILE,
        "MacroTile1": TILE,
        "GlobalSplitU": 1,
        "ProblemType": {"UseScaleAlphaVec": 0},
    }
    # s[8:15] are the search temporaries; s[16:19] receive M, N, batch, K.
    search = writer.groupedGemmWaveSearch(
        kernel, 20, ContinuousRegister(8, 8), 16, 21, 22, Label("FOUND", ""), chunks
    )
    prologue = generate_load_params(
        [
            (8, 2, 0, "headers"),
            (6, 2, 8, "workgroup IDs to look up"),
            (4, 2, 16, "results"),
            (20, 1, 24, "group count"),
            (23, 1, 28, "GSU"),
        ]
    )
    inner = (
        str(prologue)
        + """
  s_mov_b32 s25, s2
  s_lshl_b32 s26, s2, 2
  s_load_dword s24, s[6:7], s26
  s_mov_b32 s22, 16
  s_mov_b32 s14, 0
  s_waitcnt lgkmcnt(0)
"""
        + str(search)
        + """
label_FOUND:
  s_mov_b64 exec, 1
  s_lshl_b32 s25, s25, 5
  v_mov_b32 v0, s25
  v_mov_b32 v4, s16
  v_mov_b32 v5, s17
  v_mov_b32 v6, s18
  v_mov_b32 v7, s19
  v_mov_b32 v8, s12
  v_mov_b32 v9, s15
  v_mov_b32 v10, s21
  v_mov_b32 v11, 0
  global_store_dwordx4 v0, v[4:7], s[4:5]
  global_store_dwordx4 v0, v[8:11], s[4:5] offset:16
"""
    )
    args = (
        ("headers", 8, "global_buffer", "u32"),
        ("queries", 8, "global_buffer", "u32"),
        ("output", 8, "global_buffer", "u32"),
        ("count", 4, "by_value", "u32"),
        ("gsu", 4, "by_value", "u32"),
    )
    return generate_kernel_asm(inner, writer, args, num_threads=64)


def search_cases():
    # Cover row/chunk cutovers, a partial final chunk, and repeated memory rounds.
    rng = np.random.default_rng(14788)
    for count in (2, 16, 17, 64, 65, 384, 385, 1000):
        headers = rng.integers(0, 129, (count, 4), dtype=np.uint32)
        headers[:, 2] %= 4
        yield headers, 2
    # The public user-argument layout stores M and N as uint32_t. These inputs
    # bracket the point where adding TILE - 1 before division would wrap.
    for axis in (0, 1):
        for dimension in (0, 1, 0xFFFFFFF0, 0xFFFFFFF1, 0xFFFFFFFF):
            # Leave a group after the large one: losing its tile count must
            # also fail the group selection, not just the returned prefix.
            headers = np.tile(np.array([1, 1, 1, 0], dtype=np.uint32), (66, 1))
            headers[64, axis] = dimension
            yield headers, 1


def queries_and_expected(headers, gsu):
    wide = headers.astype(np.uint64)
    tiles = (wide[:, 0] + TILE - 1) // TILE * ((wide[:, 1] + TILE - 1) // TILE) * wide[:, 2] * gsu
    prefixes = np.cumsum(tiles)
    # Both sides of every nonempty group's boundary, and IDs beyond all tiles.
    queries = sorted(
        {0, int(prefixes[-1]), int(prefixes[-1]) + 1}
        | {int(p) for p in prefixes}
        | {int(p) - 1 for p in prefixes if p}
    )
    indices = np.minimum(np.searchsorted(prefixes, queries, side="right"), len(headers) - 1)
    expected = np.column_stack(
        (headers[indices], tiles[indices], prefixes[indices], indices + 1, np.zeros(len(indices)))
    ).astype(np.uint32)
    return np.array(queries, dtype=np.uint32), expected


@pytest.mark.parametrize("chunks", [1, 6])
def test_search_matches_tile_intervals(chunks, tmp_path):
    asm = search_kernel(chunks)
    co = str(tmp_path / "group_search.co")
    assemble_kernel(asm, co)
    for headers, gsu in search_cases():
        queries, expected = queries_and_expected(headers, gsu)
        raw = run_on_gpu(
            co,
            expected.nbytes,
            inputs=(headers, queries),
            scalars=(len(headers), gsu),
            num_threads=64,
            grid=len(queries),
        )
        actual = np.frombuffer(raw, dtype=np.uint32).reshape(expected.shape)
        np.testing.assert_array_equal(
            actual, expected, err_msg=f"chunks={chunks}, max M/N={headers[:, :2].max(axis=0)}"
        )
