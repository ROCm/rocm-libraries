# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Paged KV caches of exactly 2^31 bytes must read their last 64 KiB.

The tiled paged-attention kernels build the K/V buffer resources with
``num_records = 0x7FFF0000`` (2^31 - 64 KiB), so on the i32 addressing path a
load at a byte offset in ``[0x7FFF0000, 0x80000000)`` returns zero. The i64
path is selected by ``_enable_i64_kv_addr`` (and its runtime copy in
``run_unified_attention_torch``) only when the cache exceeds ``0x8000_0000``
bytes. A cache in ``(0x7FFF0000, 0x80000000]`` therefore takes the i32 path
and silently reads zero K and V for its last 64 KiB; those keys still count in
the softmax denominator.

The numeric check is exact and needs no reference: with Q = 0 every legal key
has probability 1/N, and with V = 1 every output element is exactly 1. Keys
that read zero V instead give ``(N - m) / N`` for the affected sequence --
``4064/4096 = 0.9921875`` at 8 KV heads (m = 32 keys in the last 64 KiB) and
``8128/8192`` at 4 KV heads (m = 64) -- which a 0.05 tolerance does not catch.
One extra cache block pushes the same data onto the i64 path, which is
correct; that case is the control.
"""

from __future__ import annotations

import math

import pytest

from kernels.common import attention_unified as au

_RSRC_BYTES = 0x7FFF0000  # num_records of the paged K/V buffer resources
_BLOCK_SIZE = 16
_HEAD_SIZE = 128
_BATCH = 256

# (num_query_heads, num_kv_heads, seqlen_k): BF16 D128 decode, page size 16.
# With batch 256 each K (and V) cache is exactly 2^31 bytes:
# 65536 blocks x 32 KiB at 8 KV heads, 131072 blocks x 16 KiB at 4 KV heads.
_SHAPES = [(32, 8, 4096), (64, 8, 4096), (64, 4, 8192)]


def _block_stride(num_kv_heads: int) -> int:
    return _BLOCK_SIZE * num_kv_heads * _HEAD_SIZE * 2  # bf16


def _problem(hq: int, hkv: int, sk: int, num_kv_blocks: int = 0):
    return au.UnifiedAttentionProblem(
        total_q=_BATCH,
        num_seqs=_BATCH,
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_size=_HEAD_SIZE,
        block_size=_BLOCK_SIZE,
        max_seqlen_q=1,
        max_seqlen_k=sk,
        dtype="bf16",
        num_kv_blocks=num_kv_blocks,
    )


def _shape_id(shape) -> str:
    hq, hkv, sk = shape
    return f"h{hq}kv{hkv}_sk{sk}"


# ---------------------------------------------------------------------------
# Address-width gate (no GPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", _SHAPES, ids=_shape_id)
def test_gate_selects_i64_when_cache_exceeds_rsrc(shape):
    """Any cache larger than the resource size must take the i64 path."""
    hq, hkv, sk = shape
    stride = _block_stride(hkv)
    blocks = _BATCH * sk // _BLOCK_SIZE
    assert blocks * stride == 1 << 31
    # Smallest block count whose cache exceeds the resource size.
    first_over = _RSRC_BYTES // stride + 1
    for count in (first_over, blocks):
        cache_bytes = count * stride
        assert cache_bytes > _RSRC_BYTES
        assert au._enable_i64_kv_addr(_problem(hq, hkv, sk, count)), (
            f"{count} blocks = {cache_bytes:#x} bytes > rsrc {_RSRC_BYTES:#x} "
            "but the i32 path was selected; its last "
            f"{cache_bytes - _RSRC_BYTES} bytes read as zero"
        )
    # Caches that fit inside the resource keep the fast i32 path.
    assert not au._enable_i64_kv_addr(_problem(hq, hkv, sk, first_over - 1))


# ---------------------------------------------------------------------------
# On-GPU reproduction (gfx942, 2D tiled kernel)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gfx942_torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    from rocke.runtime.hip_module import get_device_arch

    if get_device_arch() != "gfx942":
        pytest.skip("reproduction targets the gfx942 tiled 2D kernel")
    return torch


@pytest.mark.gpu
@pytest.mark.parametrize(
    "extra_blocks",
    [0, 1],
    ids=["cache_2gib_i32", "cache_2gib_plus_block_i64"],
)
@pytest.mark.parametrize("shape", _SHAPES, ids=_shape_id)
def test_zero_q_unit_v_is_exactly_one(gfx942_torch, shape, extra_blocks):
    """Zero Q, constant V = 1, identity block table: every output must be 1.

    ``extra_blocks=0``: K/V caches of exactly 2^31 bytes (fails while the gate
    is ``> 0x8000_0000``). ``extra_blocks=1``: one unused trailing block moves
    the cache onto the i64 path; the data read is identical (control).
    """
    torch = gfx942_torch
    hq, hkv, sk = shape
    blocks_per_seq = sk // _BLOCK_SIZE
    used_blocks = _BATCH * blocks_per_seq
    num_blocks = used_blocks + extra_blocks
    cache_bytes = num_blocks * _block_stride(hkv)

    free_bytes, _total = torch.cuda.mem_get_info()
    if free_bytes < 2 * cache_bytes + (256 << 20):
        pytest.skip(f"needs {2 * cache_bytes >> 20} MiB free for the K/V caches")

    dev = "cuda"
    # Cache layout [num_blocks, block_size, num_kv_heads, head_size].
    cache_shape = (num_blocks, _BLOCK_SIZE, hkv, _HEAD_SIZE)
    k = torch.zeros(cache_shape, dtype=torch.bfloat16, device=dev)
    v = torch.ones(cache_shape, dtype=torch.bfloat16, device=dev)
    # Identity table: sequence b owns blocks [b * blocks_per_seq, ...), so the
    # last sequence's last blocks occupy the last bytes of the used cache.
    block_table = torch.arange(used_blocks, dtype=torch.int32, device=dev).view(
        _BATCH, blocks_per_seq
    )
    q = torch.zeros((_BATCH, hq, _HEAD_SIZE), dtype=torch.bfloat16, device=dev)
    out = torch.full_like(q, float("nan"))
    cu_seqlens_q = torch.arange(_BATCH + 1, dtype=torch.int32, device=dev)
    seqused_k = torch.full((_BATCH,), sk, dtype=torch.int32, device=dev)

    # num_kv_blocks=0: the runtime fills it from k.shape[0], as in production.
    au.run_unified_attention_torch(
        problem=_problem(hq, hkv, sk),
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        softmax_scale=1.0 / math.sqrt(_HEAD_SIZE),
        block_table=block_table,
        softcap=0.0,
        backend="tiled",
    )
    torch.cuda.synchronize()

    result = out.float().cpu()
    wrong = (result != 1.0).flatten(1).any(dim=1).nonzero().flatten().tolist()
    if wrong:
        used_bytes = used_blocks * _block_stride(hkv)
        missing_keys = max(0, used_bytes - _RSRC_BYTES) // (hkv * _HEAD_SIZE * 2)
        values = sorted(set(result[wrong].flatten().tolist()))
        pytest.fail(
            f"cache {cache_bytes:#x} bytes "
            f"(use_i64_kv_addr={cache_bytes > _RSRC_BYTES} expected): "
            f"sequences {wrong} returned {values} instead of 1.0; "
            f"zero-V reads predict {(sk - missing_keys) / sk} "
            f"({missing_keys} keys past the {_RSRC_BYTES:#x}-byte resource)"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
