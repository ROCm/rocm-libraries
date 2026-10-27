# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 tiled 2D K/V copies with i64 addressing share one buffer resource per
page of a tile instead of building one per copy.

The structural check runs on the CPU. The numeric checks need a gfx942 GPU.
Two overstate ``num_kv_blocks`` so a small cache takes the i64 path, which the
launch guard allows; one reads a cache whose pages straddle 2^31 bytes.
"""

import re
from dataclasses import replace

import pytest

from kernels.gfx942.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
)
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

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
def gfx942_torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    from rocke.runtime.hip_module import get_device_arch

    if get_device_arch() != "gfx942":
        pytest.skip("requires a gfx942 GPU")
    return torch


_HEAD_SIZE, _BLOCK_SIZE = 128, 16


def _run_decode(torch, q, k, v, block_table, lengths, num_kv_blocks):
    """Launch bf16 decode through the selector and return the output; the spec
    must take the per-page i64 K/V path."""
    from kernels.common.attention_unified import (
        UnifiedAttentionProblem,
        _tiled_spec_from_problem,
        run_unified_attention_torch,
    )

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
    out = torch.empty_like(q)
    run_unified_attention_torch(
        problem=problem,
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=torch.arange(batch + 1, dtype=torch.int32, device="cuda"),
        seqused_k=torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        softmax_scale=_HEAD_SIZE**-0.5,
        block_table=block_table,
        softcap=0.0,
        backend="tiled",
    )
    return out


def _assert_matches_reference(torch, out, q, k, v, block_table, lengths):
    num_kv_heads = k.shape[2]
    group = q.shape[1] // num_kv_heads
    for b, length in enumerate(lengths):
        pages = block_table[b, : -(-length // _BLOCK_SIZE)].long()
        keys = k[pages].reshape(-1, num_kv_heads, _HEAD_SIZE)[:length].float()
        values = v[pages].reshape(-1, num_kv_heads, _HEAD_SIZE)[:length].float()
        qb = q[b].view(num_kv_heads, group, _HEAD_SIZE).float()
        scores = torch.einsum("hgd,thd->hgt", qb, keys) * _HEAD_SIZE**-0.5
        ref = torch.einsum("hgt,thd->hgd", scores.softmax(-1), values).reshape(
            -1, _HEAD_SIZE
        )
        # On long sequences the output's spread is about 0.05 (max |o| about
        # 0.2), so a looser absolute bound could hide one misaddressed page.
        # rtol covers the bf16 rounding of the output, which grows with |o| on
        # short sequences (half an ulp is at most |o| / 256).
        torch.testing.assert_close(out[b].float(), ref, atol=2e-3, rtol=2**-8)


@pytest.mark.parametrize("num_query_heads,num_kv_heads", [(32, 8), (64, 4)])
def test_i64_decode_matches_reference(gfx942_torch, num_query_heads, num_kv_heads):
    torch = gfx942_torch
    batch, seqlen = 8, 1024
    pages = seqlen // _BLOCK_SIZE
    num_blocks = batch * pages + 16
    torch.manual_seed(0)
    k = torch.randn(
        num_blocks,
        _BLOCK_SIZE,
        num_kv_heads,
        _HEAD_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v = torch.randn_like(k)
    q = torch.randn(
        batch, num_query_heads, _HEAD_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    block_table = (
        torch.randperm(num_blocks, device="cuda")[: batch * pages]
        .view(batch, pages)
        .to(torch.int32)
    )
    lengths = [seqlen] * batch
    # num_kv_blocks is overstated so this small cache takes the i64 path.
    out = _run_decode(torch, q, k, v, block_table, lengths, 1 << 20)
    _assert_matches_reference(torch, out, q, k, v, block_table, lengths)


def test_i64_ragged_lengths_and_block_table_tail(gfx942_torch):
    """Ragged per-sequence lengths, with a last tile that runs past the end of the
    block table. The table is a view into a buffer whose tail holds an
    unmapped page id, so reading it unmasked would fault."""
    torch = gfx942_torch
    lengths = [1000, 999, 977, 961, 640, 513, 17, 1000]
    batch, num_query_heads, num_kv_heads = len(lengths), 32, 8
    width = -(-max(lengths) // _BLOCK_SIZE)
    # The last 64-token tile covers pages 60..63; page 63 is past each row.
    assert width % 4 != 0
    num_blocks = batch * width + 16
    torch.manual_seed(1)
    k = torch.randn(
        num_blocks,
        _BLOCK_SIZE,
        num_kv_heads,
        _HEAD_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v = torch.randn_like(k)
    q = torch.randn(
        batch, num_query_heads, _HEAD_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    storage = torch.full(
        (batch * width + 64,), 0x7FFFFFFF, dtype=torch.int32, device="cuda"
    )
    storage[: batch * width] = torch.randperm(num_blocks, device="cuda")[
        : batch * width
    ].to(torch.int32)
    block_table = storage[: batch * width].view(batch, width)
    out = _run_decode(torch, q, k, v, block_table, lengths, 1 << 20)
    _assert_matches_reference(torch, out, q, k, v, block_table, lengths)


def test_i64_decode_reads_past_2gib(gfx942_torch):
    """A cache just over 2 GiB per tensor whose pages straddle the 2^31-byte
    offset, so the i64 page bases are exercised with real addresses."""
    torch = gfx942_torch
    batch, seqlen, num_query_heads, num_kv_heads = 4, 1024, 32, 8
    block_bytes = _BLOCK_SIZE * num_kv_heads * _HEAD_SIZE * 2
    first_past = 2**31 // block_bytes
    num_blocks = first_past + 128
    pages = seqlen // _BLOCK_SIZE
    free, _ = torch.cuda.mem_get_info()
    if free < 2 * num_blocks * block_bytes + (1 << 30):
        pytest.skip("needs about 5 GiB of free device memory")
    torch.manual_seed(2)
    k = torch.empty(
        num_blocks,
        _BLOCK_SIZE,
        num_kv_heads,
        _HEAD_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v = torch.empty_like(k)
    # Only the pages the sequences use are filled: half below 2^31 bytes, half
    # above.
    used = torch.arange(num_blocks - batch * pages, num_blocks, device="cuda")
    assert int(used[0]) < first_past <= int(used[-1])
    k[used] = torch.randn_like(k[used])
    v[used] = torch.randn_like(v[used])
    q = torch.randn(
        batch, num_query_heads, _HEAD_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    block_table = (
        used[torch.randperm(batch * pages, device="cuda")]
        .view(batch, pages)
        .to(torch.int32)
    )
    lengths = [seqlen] * batch
    out = _run_decode(torch, q, k, v, block_table, lengths, num_blocks)
    _assert_matches_reference(torch, out, q, k, v, block_table, lengths)
