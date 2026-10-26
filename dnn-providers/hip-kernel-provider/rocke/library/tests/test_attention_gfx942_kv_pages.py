# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 tiled 2D K/V copies with i64 addressing share one buffer resource per
page of a tile instead of building one per copy.

The structural check runs on the CPU. The numeric check needs a gfx942 GPU; it
overstates ``num_kv_blocks`` so a small cache takes the i64 path, which the
launch guard allows.
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


def test_i64_copies_share_one_resource_per_page():
    i32 = _counts(_DECODE)
    i64 = _counts(replace(_DECODE, use_i64_kv_addr=True))
    pages = _DECODE.n_blocks_per_tile
    # Each lane copies one dword (two halves) per copy.
    copies_per_tile = (
        _DECODE.tile_size * _DECODE.head_size // (64 * _DECODE.num_warps * 2)
    )
    tile_loads = i64["copies"] // copies_per_tile
    assert pages == 4 and tile_loads > 0
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


@pytest.mark.parametrize("num_query_heads,num_kv_heads", [(32, 8), (64, 4)])
def test_i64_decode_matches_reference(gfx942_torch, num_query_heads, num_kv_heads):
    torch = gfx942_torch
    from kernels.common.attention_unified import (
        UnifiedAttentionProblem,
        _tiled_spec_from_problem,
        run_unified_attention_torch,
    )

    batch, head_size, block_size, seqlen = 8, 128, 16, 1024
    pages = seqlen // block_size
    num_blocks = batch * pages + 16
    torch.manual_seed(0)
    k = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v = torch.randn_like(k)
    q = torch.randn(
        batch, num_query_heads, head_size, dtype=torch.bfloat16, device="cuda"
    )
    block_table = (
        torch.randperm(num_blocks, device="cuda")[: batch * pages]
        .view(batch, pages)
        .to(torch.int32)
    )
    problem = UnifiedAttentionProblem(
        total_q=batch,
        num_seqs=batch,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        max_seqlen_q=1,
        max_seqlen_k=seqlen,
        dtype="bf16",
        num_kv_blocks=1 << 20,
    )
    spec = _tiled_spec_from_problem(problem, "gfx942")
    assert spec.use_i64_kv_addr and spec.n_blocks_per_tile > 1
    out = torch.empty_like(q)
    scale = head_size**-0.5
    run_unified_attention_torch(
        problem=problem,
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=torch.arange(batch + 1, dtype=torch.int32, device="cuda"),
        seqused_k=torch.full((batch,), seqlen, dtype=torch.int32, device="cuda"),
        softmax_scale=scale,
        block_table=block_table,
        softcap=0.0,
        backend="tiled",
    )
    group = num_query_heads // num_kv_heads
    for b in range(batch):
        keys = k[block_table[b].long()].reshape(seqlen, num_kv_heads, head_size).float()
        values = (
            v[block_table[b].long()].reshape(seqlen, num_kv_heads, head_size).float()
        )
        qb = q[b].view(num_kv_heads, group, head_size).float()
        scores = torch.einsum("hgd,thd->hgt", qb, keys) * scale
        ref = torch.einsum("hgt,thd->hgd", scores.softmax(-1), values).reshape(
            -1, head_size
        )
        # The output's spread is about 0.05 (max |o| about 0.2), so a looser
        # bound could hide one misaddressed page.
        torch.testing.assert_close(out[b].float(), ref, atol=2e-3, rtol=0)
