# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 tiled 2D K/V copies with i64 addressing share one buffer resource per
page of a tile instead of building one per copy.

The structural check runs on the CPU. The numeric checks need a gfx942 GPU and no
torch; the reference is float64 NumPy.
Two overstate ``num_kv_blocks`` so a small cache takes the i64 path, which the
launch guard allows; one reads a cache whose pages straddle 2^31 bytes.
"""

import re
from dataclasses import replace

import numpy as np
import pytest
from kernels.gfx942.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
)
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.runtime.hip_module import get_device_arch

from . import _attention_numpy as an
from ._attention_device_array import DeviceArray

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


_ARCH = get_device_arch()


@pytest.fixture(scope="module")
def rt():
    return an.runtime()


_HEAD_SIZE, _BLOCK_SIZE = 128, 16


@pytest.fixture(autouse=True)
def _eager_launches(monkeypatch):
    """Graph replay of short launches needs torch; run them eagerly."""
    monkeypatch.setenv("HIPDNN_GFX942_2D_GRAPH", "0")
    monkeypatch.setenv("HIPDNN_GFX942_3D_GRAPH", "0")


def _run_decode(rt, q, k, v, block_table, lengths, num_kv_blocks):
    """Launch bf16 decode through the selector and return the output as fp32; the
    spec must take the per-page i64 K/V path. ``q`` is a fp32 array, ``k``/``v``
    and ``block_table`` are device tensors."""
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
    out = an.empty_device(rt, q.shape, "bf16")
    run_unified_attention_torch(
        problem=problem,
        q=an.to_device(rt, q, "bf16"),
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=DeviceArray(rt, np.arange(batch + 1, dtype=np.int32)),
        seqused_k=DeviceArray(rt, np.array(lengths, dtype=np.int32)),
        softmax_scale=_HEAD_SIZE**-0.5,
        block_table=block_table,
        softcap=0.0,
        backend="tiled",
    )
    return an.read(out, "bf16")


def _assert_matches_reference(out, q, k_host, v_host, table, lengths):
    """``k_host``/``v_host`` hold the pages ``table`` (host int32) indexes."""
    ref = an.paged_attention_reference(
        q, k_host, v_host, table, [1] * len(lengths), lengths, scale=_HEAD_SIZE**-0.5
    )
    # On long sequences the output's spread is about 0.05 (max |o| about
    # 0.2), so a looser absolute bound could hide one misaddressed page.
    # rtol covers the bf16 rounding of the output, which grows with |o| on
    # short sequences (half an ulp is at most |o| / 256).
    assert np.isfinite(out).all(), "non-finite output"
    excess = np.abs(out - ref) - (2e-3 + 2**-8 * np.abs(ref))
    worst = np.unravel_index(int(np.argmax(excess)), excess.shape)
    assert excess[worst] <= 0, (
        f"output differs from the reference at {worst}: "
        f"got {out[worst]:.6g}, want {ref[worst]:.6g}"
    )


@pytest.mark.parametrize("num_query_heads,num_kv_heads", [(32, 8), (64, 4)])
@pytest.mark.gpu
@pytest.mark.skipif(_ARCH != "gfx942", reason=f"requires a gfx942 GPU, found {_ARCH}")
def test_i64_decode_matches_reference(rt, num_query_heads, num_kv_heads):
    batch, seqlen = 8, 1024
    pages = seqlen // _BLOCK_SIZE
    num_blocks = batch * pages + 16
    rng = np.random.default_rng(0)
    k = an.randn(rng, (num_blocks, _BLOCK_SIZE, num_kv_heads, _HEAD_SIZE), "bf16")
    v = an.randn(rng, k.shape, "bf16")
    q = an.randn(rng, (batch, num_query_heads, _HEAD_SIZE), "bf16")
    table = rng.permutation(num_blocks)[: batch * pages].reshape(batch, pages)
    table = table.astype(np.int32)
    lengths = [seqlen] * batch
    # num_kv_blocks is overstated so this small cache takes the i64 path.
    out = _run_decode(
        rt,
        q,
        an.to_device(rt, k, "bf16"),
        an.to_device(rt, v, "bf16"),
        DeviceArray(rt, table),
        lengths,
        1 << 20,
    )
    _assert_matches_reference(out, q, k, v, table, lengths)


@pytest.mark.gpu
@pytest.mark.skipif(_ARCH != "gfx942", reason=f"requires a gfx942 GPU, found {_ARCH}")
def test_i64_ragged_lengths_and_block_table_tail(rt):
    """Ragged per-sequence lengths, with a last tile that runs past the end of the
    block table. The table is a view of the first rows of a buffer whose tail holds
    an unmapped page id, so reading it unmasked would fault."""
    lengths = [1000, 999, 977, 961, 640, 513, 17, 1000]
    batch, num_query_heads, num_kv_heads = len(lengths), 32, 8
    width = -(-max(lengths) // _BLOCK_SIZE)
    # The last 64-token tile covers pages 60..63; page 63 is past each row.
    assert width % 4 != 0
    num_blocks = batch * width + 16
    rng = np.random.default_rng(1)
    k = an.randn(rng, (num_blocks, _BLOCK_SIZE, num_kv_heads, _HEAD_SIZE), "bf16")
    v = an.randn(rng, k.shape, "bf16")
    q = an.randn(rng, (batch, num_query_heads, _HEAD_SIZE), "bf16")
    storage = np.full((batch + 64 // width + 1, width), 0x7FFFFFFF, dtype=np.int32)
    storage[:batch] = rng.permutation(num_blocks)[: batch * width].reshape(batch, width)
    out = _run_decode(
        rt,
        q,
        an.to_device(rt, k, "bf16"),
        an.to_device(rt, v, "bf16"),
        DeviceArray(rt, storage, shape=(batch, width)),
        lengths,
        1 << 20,
    )
    _assert_matches_reference(out, q, k, v, storage[:batch], lengths)


@pytest.mark.gpu
@pytest.mark.skipif(_ARCH != "gfx942", reason=f"requires a gfx942 GPU, found {_ARCH}")
def test_i64_decode_reads_past_2gib(rt):
    """A cache just over 2 GiB per tensor whose pages straddle the 2^31-byte
    offset, so the i64 page bases are exercised with real addresses."""
    batch, seqlen, num_query_heads, num_kv_heads = 4, 1024, 32, 8
    block_bytes = _BLOCK_SIZE * num_kv_heads * _HEAD_SIZE * 2
    first_past = 2**31 // block_bytes
    num_blocks = first_past + 128
    pages = seqlen // _BLOCK_SIZE
    rng = np.random.default_rng(2)
    # Only the pages the sequences use are filled: half below 2^31 bytes, half
    # above.
    used = np.arange(num_blocks - batch * pages, num_blocks)
    assert int(used[0]) < first_past <= int(used[-1])
    shape = (num_blocks, _BLOCK_SIZE, num_kv_heads, _HEAD_SIZE)
    try:
        k = an.LargeDeviceTensor(rt, shape, "bf16")
        v = an.LargeDeviceTensor(rt, shape, "bf16")
    except RuntimeError as err:
        if "hipMalloc" in str(err):
            pytest.skip(f"needs about 5 GiB of free device memory: {err}")
        raise
    k_host = an.randn(rng, (len(used),) + shape[1:], "bf16")
    v_host = an.randn(rng, k_host.shape, "bf16")
    for i, page in enumerate(used):
        k.write_rows(int(page), k_host[i : i + 1])
        v.write_rows(int(page), v_host[i : i + 1])
    q = an.randn(rng, (batch, num_query_heads, _HEAD_SIZE), "bf16")
    order = rng.permutation(batch * pages)
    block_table = used[order].reshape(batch, pages).astype(np.int32)
    lengths = [seqlen] * batch
    out = _run_decode(rt, q, k, v, DeviceArray(rt, block_table), lengths, num_blocks)
    compact = order.reshape(batch, pages).astype(np.int32)
    _assert_matches_reference(out, q, k_host, v_host, compact, lengths)
