# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-GPU check: gfx942 cfvst prefill reads a paged KV cache past 2^31 bytes.

The default gfx942 prefill V loader for bf16 D64 and fp16 D64/D128 (conflict-
free V store, cfvst) used to form an i32 byte offset from the cache base, so a
cache past 2^31 bytes wrapped (wrong data or a GPU fault), and the launch guard
then refused it. Under ``use_i64_kv_addr`` it now loads from the tile's page
pointers with i32 offsets within the page.

Each case launches through ``run_unified_attention_torch`` on a 2.125 GiB (per
K and V) ``rocke.runtime`` device cache with every block-table entry on the top
blocks, all past 2^31 bytes, and checks max_abs against an FP64 numpy paged
reference. A wrapped offset reads other blocks' random data and fails the check. The routing anchor runs
without a GPU and pins that each shipped shape compiles an i64 cfvst kernel with
no launch limit, so the numeric check cannot silently test another kernel.
"""

from __future__ import annotations

from unittest import mock

import builders.common.attention_spec_builder as _asb
import numpy as np
import pytest
from kernels import UnifiedAttentionProblem
from kernels.common import attention_unified as au
from rocke.runtime.hip_module import get_device_free_memory
from rocke.tests.gpu import current_device_arch_in

from ._attention_paged_reference import round_to_half
from ._attention_tiled_launch import launch_unified
from ._attention_unified_support import chunked_paged_reference, sparse_paged_cache

_SEQ = 4096
_NUM_SEQS = 2
_NUM_QH = 32
_NUM_KV = 8
_CACHE_BYTES = 2 * 2**30 + 2**27  # 2.125 GiB per K and V
# K and V caches, the output and headroom.
_GPU_BYTES_NEEDED = 6 << 30

# The shipped cfvst prefill shapes: (dtype, head_size, block_size).
_SHIPPED = (("bf16", 64, 32), ("fp16", 64, 32), ("fp16", 128, 32))


_CASES = [pytest.param(*shape, id=f"{shape[0]}_d{shape[1]}") for shape in _SHIPPED]


def _block_bytes(head_size: int, block_size: int) -> int:
    # bf16 and fp16 are both 2 bytes.
    return block_size * _NUM_KV * head_size * 2


def _problem(dtype: str, head_size: int, block_size: int, num_kv_blocks: int = 0):
    return UnifiedAttentionProblem(
        total_q=_SEQ * _NUM_SEQS,
        num_seqs=_NUM_SEQS,
        num_query_heads=_NUM_QH,
        num_kv_heads=_NUM_KV,
        head_size=head_size,
        block_size=block_size,
        max_seqlen_q=_SEQ,
        max_seqlen_k=_SEQ,
        dtype=dtype,
        num_kv_blocks=num_kv_blocks,
    )


@pytest.fixture
def gfx942_with_room():
    current_device_arch_in(("gfx942",), "the gfx942 cfvst past-2 GiB check")
    free = get_device_free_memory()
    if free is None or free < _GPU_BYTES_NEEDED:
        pytest.skip(f"needs {_GPU_BYTES_NEEDED >> 30} GiB of free device memory")


def _clean_launch_caches():
    """Patch every 2D launch cache empty so these launches start cold and leave
    nothing behind."""
    return [
        mock.patch.dict(cache, clear=True)
        for cache in (
            au._ATTN_TILED_CACHE,
            au._2D_LAUNCHERS,
            au._2D_LAUNCH_META,
            au._2D_KV_ADDR_LIMITS,
            au._2D_GRAPHS,
            au._2D_GRAPH_REFS,
        )
    ]


@pytest.mark.parametrize("dtype,head_size,block_size", _SHIPPED)
def test_route_compiles_i64_cfvst_without_limit(dtype, head_size, block_size):
    nblk = _CACHE_BYTES // _block_bytes(head_size, block_size)
    p = _problem(dtype, head_size, block_size, nblk)
    patches = [mock.patch.object(au, "_resolve_attention_arch", return_value="gfx942")]
    patches += _clean_launch_caches()
    for patch in patches:
        patch.start()
    try:
        spec = _asb._tiled_spec_from_problem(p, "gfx942")
        meta = au._get_2d_launch_meta(
            p, au._tiled_cache_key(p, "gfx942"), arch="gfx942"
        )
    finally:
        for patch in reversed(patches):
            patch.stop()
    assert spec.use_conflict_free_v_store
    assert spec.use_i64_kv_addr
    assert meta.kv_addr_limit is None


@pytest.mark.parametrize("dtype,head_size,block_size", _CASES)
def test_cfvst_prefill_on_top_blocks_past_2gib(
    gfx942_with_room, dtype, head_size, block_size
):
    rng = np.random.default_rng(0)
    block_bytes = _block_bytes(head_size, block_size)
    num_blocks = _CACHE_BYTES // block_bytes
    blocks_per_seq = _SEQ // block_size
    # The topmost blocks in a fixed shuffle: the table includes the last block
    # and nothing below 2^31 bytes.
    used = _NUM_SEQS * blocks_per_seq
    top = np.arange(num_blocks - used, num_blocks, dtype=np.int32)
    top = top[rng.permutation(used)]
    assert int(top.min()) * block_bytes >= 2**31
    block_tables = top.reshape(_NUM_SEQS, blocks_per_seq)
    # Random data everywhere, so a wrapped offset reads values that differ;
    # the used top blocks hold standard-normal values the reference decodes.
    shape = (num_blocks, block_size, _NUM_KV, head_size)
    used_ids = np.arange(num_blocks - used, num_blocks)
    used_values = [
        round_to_half(rng.standard_normal((used, *shape[1:]), np.float32), dtype)
        for _ in ("k", "v")
    ]
    caches = [
        sparse_paged_cache(rng, shape, used_ids, values, dtype)
        for values in used_values
    ]
    query = round_to_half(
        rng.standard_normal((_SEQ * _NUM_SEQS, _NUM_QH, head_size), np.float32), dtype
    )
    scale = head_size**-0.5

    patches = [mock.patch.object(au, "_resolve_attention_arch", return_value="gfx942")]
    patches += _clean_launch_caches()
    for patch in patches:
        patch.start()
    try:
        # num_kv_blocks stays 0: the launcher fills it from k.shape[0], as in
        # production, which makes the cfvst spec i64.
        out = launch_unified(
            _problem(dtype, head_size, block_size),
            query,
            caches[0],
            caches[1],
            block_tables,
            scale,
            q_lens=[_SEQ] * _NUM_SEQS,
            kv_lens=[_SEQ] * _NUM_SEQS,
            backend="tiled",
        )
    finally:
        for patch in reversed(patches):
            patch.stop()

    # The reference reads only the used blocks, renumbered from 0.
    local_tables = block_tables - (num_blocks - used)
    ref = np.concatenate(
        [
            chunked_paged_reference(
                query[seq * _SEQ : (seq + 1) * _SEQ],
                *used_values,
                _SEQ,
                local_tables[seq],
                scale,
            )
            for seq in range(_NUM_SEQS)
        ]
    )
    max_abs = float(np.abs(ref - out).max())
    # The repo-wide attention gates (ALGORITHM.md section 10): 4e-2 bf16, 2e-2 fp16.
    tol = 4e-2 if dtype == "bf16" else 2e-2
    assert max_abs <= tol, f"cfvst past 2 GiB: max_abs {max_abs:.4f} > {tol}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
