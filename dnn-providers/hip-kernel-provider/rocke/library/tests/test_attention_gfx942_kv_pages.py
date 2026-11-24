# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 tiled 2D K/V copies with i64 addressing share one buffer resource per
page of a tile instead of building one per copy.

The structural check runs on the CPU. The numeric checks need a gfx942 GPU;
they launch on ``rocke.runtime`` device arrays and compare with an FP64 numpy
paged reference. Two overstate ``num_kv_blocks`` so a small cache takes the i64
path, which the launch guard allows; one reads a cache whose pages straddle
2^31 bytes.
"""

import re
from dataclasses import replace

import numpy as np
import pytest
from kernels.common.attention_unified import (
    UnifiedAttentionProblem,
    _tiled_spec_from_problem,
)
from kernels.gfx942.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
)
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.runtime.device_array import DeviceArray
from rocke.runtime.hip_module import get_device_total_memory
from rocke.tests.gpu import current_device_arch_in

from ._attention_paged_reference import (
    decode_half,
    encode_half,
    paged_attention_reference,
)
from ._attention_tiled_launch import launch_unified

# The selector's spec for bf16 D128 decode with 256 sequences of 4096 tokens,
# GQA 32/8 and block size 16: each 64-token tile spans four pages.
_DECODE = UnifiedAttention2DTiledSpec(
    head_size=128,
    block_size=16,
    num_query_heads=32,
    num_kv_heads=8,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    num_seqs=256,
    waves_per_eu=2,
    use_register_pv=True,
    tile_size=64,
)


def _counts(spec):
    ir = _lower_kernel_to_llvm_python(
        build_unified_attention_2d_tiled(spec, arch="gfx942"),
        arch="gfx942",
        llvm_flavor="llvm22",
    )
    return dict(
        resources=len(
            re.findall(r"call ptr addrspace\(8\) @llvm\.amdgcn\.make\.buffer\.rsrc", ir)
        ),
        copies=ir.count("call void @llvm.amdgcn.raw.ptr.buffer.load.lds"),
        page_loads=len(
            re.findall(
                r"getelementptr inbounds i32, ptr addrspace\(1\) %block_tables_ptr", ir
            )
        ),
    )


@pytest.mark.parametrize(
    "spec",
    [
        _DECODE,
        # Four warps and two 32-token pages per tile.
        replace(_DECODE, block_size=32, num_warps=4),
        # One 64-token page per tile.
        replace(_DECODE, block_size=64),
    ],
    ids=["nw1_4pages", "nw4_2pages", "nw1_1page"],
)
def test_i64_copies_share_one_resource_per_page(spec):
    """Counts only: the numeric tests check the addressing. Builds the spec
    directly, not through the selector, so it runs without a cache of any
    size; the numeric tests reach the decode spec through the selector."""
    i32 = _counts(spec)
    i64 = _counts(replace(spec, use_i64_kv_addr=True))
    pages = spec.n_blocks_per_tile
    # Each lane copies one dword (two halves) per copy.
    copies_per_tile = spec.tile_size * spec.head_size // (64 * spec.num_warps * 2)
    tile_loads = i64["copies"] // copies_per_tile
    assert tile_loads > 0 and i64["copies"] % copies_per_tile == 0
    assert i64["copies"] == i32["copies"]
    # One page-id load and one resource per page of each tile load, where a
    # per-copy resource would add one of each per copy.
    assert i64["page_loads"] == pages * tile_loads
    assert i64["resources"] - i32["resources"] == pages * tile_loads


@pytest.fixture(scope="module")
def gfx942():
    current_device_arch_in(("gfx942",), "the gfx942 i64 K/V page test")


_HEAD_SIZE, _BLOCK_SIZE = 128, 16
_SCALE = _HEAD_SIZE**-0.5


def _bf16(rng, *shape):
    """Random normal values as bf16 bits."""
    return encode_half(rng.standard_normal(shape, dtype=np.float32), "bf16")


def _run_decode(q, k, v, block_table, lengths, num_kv_blocks):
    """Launch bf16 decode through the selector and return the output; the spec
    must take the per-page i64 K/V path. ``k`` and ``v`` are bf16 bit caches,
    ``block_table`` a host table or a DeviceArray view."""
    batch, num_query_heads, _ = q.shape
    problem = UnifiedAttentionProblem(
        total_q=batch,
        num_seqs=batch,
        num_query_heads=num_query_heads,
        num_kv_heads=k.shape[2],
        head_size=_HEAD_SIZE,
        block_size=_BLOCK_SIZE,
        max_seqlen_q=1,
        max_seqlen_k=max(lengths),
        dtype="bf16",
        num_kv_blocks=num_kv_blocks,
    )
    spec = _tiled_spec_from_problem(problem, "gfx942")
    assert spec.use_i64_kv_addr and spec.n_blocks_per_tile > 1
    return launch_unified(
        problem,
        DeviceArray(q, "bf16"),
        DeviceArray(k, "bf16"),
        DeviceArray(v, "bf16"),
        block_table,
        _SCALE,
        q_lens=[1] * batch,
        kv_lens=lengths,
        backend="tiled",
    )


def _assert_matches_reference(out, q, k, v, block_table, lengths):
    """Each sequence against the FP64 paged reference. Only the pages the table
    names are decoded, so a cache of several GiB costs no more than its use."""
    pages = np.unique(block_table)
    decoded_k, decoded_v = (decode_half(cache[pages], "bf16") for cache in (k, v))
    compact_table = np.searchsorted(pages, block_table)
    for b, length in enumerate(lengths):
        ref = paged_attention_reference(
            decode_half(q[b : b + 1], "bf16"),
            decoded_k,
            decoded_v,
            length,
            compact_table[b],
            _SCALE,
        )
        # On long sequences the output's spread is about 0.05 (max |o| about
        # 0.2), so a looser absolute bound could hide one misaddressed page.
        # rtol covers the bf16 rounding of the output, which grows with |o| on
        # short sequences (half an ulp is at most |o| / 256).
        np.testing.assert_allclose(out[b], ref[0], atol=2e-3, rtol=2**-8)


@pytest.mark.parametrize("num_query_heads,num_kv_heads", [(32, 8), (64, 4)])
def test_i64_decode_matches_reference(gfx942, num_query_heads, num_kv_heads):
    batch, seqlen = 8, 1024
    pages = seqlen // _BLOCK_SIZE
    num_blocks = batch * pages + 16
    rng = np.random.default_rng(0)
    k = _bf16(rng, num_blocks, _BLOCK_SIZE, num_kv_heads, _HEAD_SIZE)
    v = _bf16(rng, num_blocks, _BLOCK_SIZE, num_kv_heads, _HEAD_SIZE)
    q = _bf16(rng, batch, num_query_heads, _HEAD_SIZE)
    block_table = (
        rng.permutation(num_blocks)[: batch * pages]
        .reshape(batch, pages)
        .astype(np.int32)
    )
    lengths = [seqlen] * batch
    # num_kv_blocks is overstated so this small cache takes the i64 path.
    out = _run_decode(q, k, v, block_table, lengths, 1 << 20)
    _assert_matches_reference(out, q, k, v, block_table, lengths)


def test_i64_ragged_lengths_and_block_table_tail(gfx942):
    """Ragged per-sequence lengths, with a last tile that runs past the end of the
    block table. The table is a view into a buffer whose tail holds an
    unmapped page id, so reading it unmasked would fault."""
    lengths = [1000, 999, 977, 961, 640, 513, 17, 1000]
    batch, num_query_heads, num_kv_heads = len(lengths), 32, 8
    width = -(-max(lengths) // _BLOCK_SIZE)
    # The last 64-token tile covers pages 60..63; page 63 is past each row.
    assert width % 4 != 0
    num_blocks = batch * width + 16
    rng = np.random.default_rng(1)
    k = _bf16(rng, num_blocks, _BLOCK_SIZE, num_kv_heads, _HEAD_SIZE)
    v = _bf16(rng, num_blocks, _BLOCK_SIZE, num_kv_heads, _HEAD_SIZE)
    q = _bf16(rng, batch, num_query_heads, _HEAD_SIZE)
    storage = np.full(batch * width + 64, 0x7FFFFFFF, np.int32)
    storage[: batch * width] = rng.permutation(num_blocks)[: batch * width]
    block_table = storage[: batch * width].reshape(batch, width)
    table_view = DeviceArray(storage).view((batch, width), (width, 1))
    out = _run_decode(q, k, v, table_view, lengths, 1 << 20)
    _assert_matches_reference(out, q, k, v, block_table, lengths)


# Its caches take most of a small device's memory, so it runs with no other
# test process on the device (rocke.tests.ctest_workers).
@pytest.mark.rocke_run_alone
def test_i64_decode_reads_past_2gib(gfx942):
    """A cache just over 2 GiB per tensor whose pages straddle the 2^31-byte
    offset, so the i64 page bases are exercised with real addresses."""
    batch, seqlen, num_query_heads, num_kv_heads = 4, 1024, 32, 8
    block_bytes = _BLOCK_SIZE * num_kv_heads * _HEAD_SIZE * 2
    first_past = 2**31 // block_bytes
    num_blocks = first_past + 128
    pages = seqlen // _BLOCK_SIZE
    total = get_device_total_memory()
    if total is None or total < 2 * num_blocks * block_bytes + (1 << 30):
        pytest.skip("needs a device with about 5 GiB of memory")
    rng = np.random.default_rng(2)
    shape = (num_blocks, _BLOCK_SIZE, num_kv_heads, _HEAD_SIZE)
    k, v = np.zeros(shape, np.uint16), np.zeros(shape, np.uint16)
    # Only the pages the sequences use are filled: half below 2^31 bytes, half
    # above.
    used = np.arange(num_blocks - batch * pages, num_blocks)
    assert used[0] < first_past <= used[-1]
    k[used] = _bf16(rng, len(used), *shape[1:])
    v[used] = _bf16(rng, len(used), *shape[1:])
    q = _bf16(rng, batch, num_query_heads, _HEAD_SIZE)
    block_table = used[rng.permutation(batch * pages)].reshape(batch, pages)
    block_table = block_table.astype(np.int32)
    lengths = [seqlen] * batch
    out = _run_decode(q, k, v, block_table, lengths, num_blocks)
    _assert_matches_reference(out, q, k, v, block_table, lengths)
