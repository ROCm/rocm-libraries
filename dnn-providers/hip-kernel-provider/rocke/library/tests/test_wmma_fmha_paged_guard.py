# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Paged RDNA attention contains out-of-range ``block_table`` values on device.

The table is caller data. A page id outside ``[0, num_blocks)`` must not move a
K / V buffer base outside the caches: its keys read as zero K / V rows (a score
of 0 and a zero value row, like every other dead row). Each case lays the K and
V caches inside larger allocations whose surroundings are NaN, points some table
entries at pages before the cache (negative ids), just past its end and far
outside it, and compares O of the RDNA prefill (``wmma_fmha_fwd_v2``) and the
split-KV decode (``wmma_fmha_decode``, one split) with a float64 reference that
zeroes the K / V rows of those pages. A kernel that follows the id reads a NaN
canary (or leaves the allocation) and the output is NaN or the launch faults;
table entries the sequences never reach (padding) hold garbage and are not read.
Pages smaller than a key tile (one lookup per loaded row) and at least one tile
(one lookup per tile) are both covered.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from rocke.core.arch import generic_arch_from_target_id
from rocke.runtime.hip_module import get_device_arch

_ARCH = generic_arch_from_target_id(get_device_arch() or "")
_NEEDS_RDNA = pytest.mark.skipif(
    _ARCH not in ("gfx11-generic", "gfx12-generic"),
    reason="needs an RDNA3/3.5 (gfx11) or RDNA4 (gfx12) GPU",
)
_LOG2E = 1.4426950408889634
_I32_MAX, _I32_MIN = 0x7FFFFFFF, -0x80000000
_GUARD_PAGES = 8  # NaN canary pages before and after each cache
_HQ, _HK, _D = 8, 2, 64
_LENS_Q = (5, 3)
_LENS_K = (100, 70)

# name -> bad ids, assigned in turn to the table positions below. The "near" ones
# land in the canary around the caches; the "far" ones are outside any mapping the
# test allocated (a following kernel faults).
_BAD = {
    "negative_one": (-1,),
    "just_past_end": (None,),  # == num_blocks
    "past_end_plus": (None, "nb+3"),
    "negative_and_past": (-1, None, -_GUARD_PAGES),
    "far": (_I32_MIN, _I32_MAX, 1 << 20, -(1 << 20)),
}


def _host_bytes(array):
    return (ctypes.c_ubyte * array.nbytes).from_buffer(array.reshape(-1).view(np.uint8))


class _Dev:
    def __init__(self, rt):
        self.rt, self.ptrs = rt, []

    def put(self, array):
        array = np.ascontiguousarray(array)
        ptr = self.rt.alloc(max(array.nbytes, 4))
        if array.nbytes:
            self.rt.memcpy_h2d(ptr, _host_bytes(array), array.nbytes)
        self.ptrs.append(ptr)
        return ptr

    def put_guarded(self, array, guard_bytes):
        """Upload ``array`` between NaN (0xFF byte) canaries; return its pointer."""
        array = np.ascontiguousarray(array)
        total = guard_bytes + array.nbytes + guard_bytes
        ptr = self.rt.alloc(total)
        self.ptrs.append(ptr)
        self.rt.memset(ptr, 0xFF, total)
        self.rt.memcpy_h2d(ptr + guard_bytes, _host_bytes(array), array.nbytes)
        return ptr + guard_bytes

    def get(self, ptr, like):
        out = np.empty_like(like)
        self.rt.memcpy_d2h(_host_bytes(out), ptr, out.nbytes)
        return out

    def free(self):
        for ptr in self.ptrs:
            self.rt.free(ptr)


def _make_case(npdt, ps, bad, seed):
    """Logical K / V / Q, the page-permuted caches and the table, plus the
    reference (K / V rows on bad pages zeroed) of every sequence."""
    rng = np.random.default_rng(seed)
    B = len(_LENS_K)
    pages = [-(-n // ps) for n in _LENS_K]
    nb = sum(pages) + 3
    order = rng.permutation(nb)
    k_cache = np.zeros((nb, ps, _HK, _D), npdt)
    v_cache = np.zeros((nb, ps, _HK, _D), npdt)
    table = np.full((B, max(pages) + 2), _I32_MIN, np.int32)  # padding: never read
    ks, vs, qs = [], [], []
    for n in _LENS_K:
        ks.append((rng.standard_normal((n, _HK, _D)) * 0.5).astype(npdt))
        vs.append((rng.standard_normal((n, _HK, _D)) * 0.5).astype(npdt))
    for n in _LENS_Q:
        qs.append((rng.standard_normal((n, _HQ, _D)) * 0.5).astype(npdt))
    nxt = 0
    for bi, n in enumerate(_LENS_K):
        for pg in range(pages[bi]):
            phys = int(order[nxt])
            nxt += 1
            table[bi, pg] = phys
            t0, t1 = pg * ps, min(n, (pg + 1) * ps)
            k_cache[phys, : t1 - t0] = ks[bi][t0:t1]
            v_cache[phys, : t1 - t0] = vs[bi][t0:t1]
    # Table positions that become bad: the first and last page of sequence 0 and
    # the last page of sequence 1 (all inside the visible keys), then more.
    spots = [(0, 0), (0, pages[0] - 1), (1, pages[1] - 1), (1, 0)]
    kz = [x.astype(np.float64) for x in ks]
    vz = [x.astype(np.float64) for x in vs]
    for i, value in enumerate(bad):
        bi, pg = spots[i % len(spots)]
        if value is None:  # one past the last page
            value = nb
        elif value == "nb+3":
            value = nb + 3
        table[bi, pg] = value
        t0, t1 = pg * ps, min(_LENS_K[bi], (pg + 1) * ps)
        kz[bi][t0:t1] = 0.0
        vz[bi][t0:t1] = 0.0
    return dict(
        nb=nb,
        k_cache=k_cache,
        v_cache=v_cache,
        table=table,
        qs=qs,
        kz=kz,
        vz=vz,
    )


def _reference(qs, kz, vz):
    scale = _D**-0.5
    outs = []
    for q, k, v in zip(qs, kz, vz):
        o = np.zeros((q.shape[0], _HQ, _D))
        for h in range(_HQ):
            s = q[:, h].astype(np.float64) @ k[:, h // (_HQ // _HK)].T * scale
            p = np.exp(s - s.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            o[:, h] = p @ v[:, h // (_HQ // _HK)]
        outs.append(o)
    return np.concatenate(outs)


def _run(kind, dtype, ps, bad_key):
    ml_dtypes = pytest.importorskip("ml_dtypes")
    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.packing import pack_args

    npdt = ml_dtypes.bfloat16 if dtype == "bf16" else np.float16
    case = _make_case(npdt, ps, _BAD[bad_key], seed=sum(map(ord, bad_key + kind)))
    ref = _reference(case["qs"], case["kz"], case["vz"])
    B = len(_LENS_K)
    total_q = sum(_LENS_Q)
    max_q, max_k = max(_LENS_Q), max(_LENS_K)
    cu_q = np.concatenate([[0], np.cumsum(_LENS_Q)]).astype(np.int32)
    q_all = np.concatenate(case["qs"])
    o_like = np.zeros((total_q, _HQ, _D), npdt)
    lse_like = np.zeros((_HQ, total_q), np.float32)
    page_bytes = ps * _HK * _D * 2
    guard = _GUARD_PAGES * page_bytes

    rt = Runtime()
    dev = _Dev(rt)
    module = None
    try:
        values = dict(
            scale_log2=_D**-0.5 * _LOG2E,
            seqlen_q=max_q,
            seqlen_k=max_k,
            num_query_heads=_HQ,
            num_kv_heads=_HK,
            num_v_heads=_HK,
            head_dim_q=_D,
            head_dim_v=_D,
            bottom_right=0,
            window_left=-1,
            window_right=-1,
            write_lse=1,
            cu_seqlens_q=dev.put(cu_q),
            seqused_k=dev.put(np.array(_LENS_K, np.int32)),
            block_table=dev.put(case["table"]),
            block_table_stride=case["table"].shape[1],
            stride_k_block=ps * _HK * _D,
            stride_v_block=ps * _HK * _D,
            num_blocks=case["nb"],
            stride_q_batch=0,
            stride_q_token=_HQ * _D,
            stride_q_head=_D,
            stride_o_batch=0,
            stride_o_token=_HQ * _D,
            stride_o_head=_D,
            stride_lse_batch=0,
            stride_lse_token=1,
            stride_lse_head=total_q,
            stride_k_batch=0,
            stride_k_token=_HK * _D,
            stride_k_head=_D,
            stride_v_batch=0,
            stride_v_token=_HK * _D,
            stride_v_head=_D,
            Q=dev.put(q_all),
            K=dev.put_guarded(case["k_cache"], guard),
            V=dev.put_guarded(case["v_cache"], guard),
        )
        o_ptr = values["O"] = dev.put(np.full_like(o_like, np.nan))
        values["LSE"] = dev.put(lse_like)
        if kind == "prefill":
            from kernels.gfx1151.wmma_fmha_fwd_v2 import (
                WmmaFmhaFwdV2Spec,
                build_wmma_fmha_fwd_v2,
                wmma_fmha_fwd_v2_grid,
                wmma_fmha_fwd_v2_signature,
            )

            spec = WmmaFmhaFwdV2Spec(
                dtype=dtype,
                head_dim=_D,
                head_dim_v=_D,
                mask="none",
                layout="paged",
                page_block_size=ps,
            )
            values["batch"] = B
            artifact = compile_kernel(
                build_wmma_fmha_fwd_v2(spec, arch=_ARCH), arch=_ARCH, backend="python"
            )
            grid, block = wmma_fmha_fwd_v2_grid(
                spec, batch=B, seqlen_q=max_q, num_query_heads=_HQ
            )
            signature = wmma_fmha_fwd_v2_signature(spec)
        else:
            from kernels.gfx1151 import wmma_fmha_decode as dec

            spec = dec.WmmaFmhaDecodeSpec(
                head_size=dec.wmma_fmha_decode_bucket(head_dim_q=_D, head_dim_v=_D),
                dtype=dtype,
                layout="paged",
                page_block_size=ps,
            )
            rspec = dec.resolve_wmma_fmha_decode_spec(spec, arch=_ARCH)
            heads = dict(num_query_heads=_HQ, num_kv_heads=_HK, num_v_heads=_HK)
            groups = dec.wmma_fmha_decode_pack_groups(**heads)
            row_tiles = dec.wmma_fmha_decode_row_tiles(seqlen_q=max_q, **heads)
            values.update(num_splits=1, WS_O=0, WS_LSE=0, WS_CNT=0)
            artifact = compile_kernel(
                dec.build_wmma_fmha_decode(spec, arch=_ARCH),
                arch=_ARCH,
                backend="python",
            )
            grid = dec.wmma_fmha_decode_grid(
                rspec, batch=B, pack_groups=groups, row_tiles=row_tiles, num_splits=1
            )
            block = (rspec.block_size, 1, 1)
            signature = dec.wmma_fmha_decode_signature(spec)
        module = rt.load_module(artifact.hsaco)
        rt.launch(
            module.get_function(artifact.kernel_name),
            grid,
            block,
            pack_args(signature, values),
        )
        rt.sync()
        got = dev.get(o_ptr, o_like).astype(np.float64)
    finally:
        rt.sync()
        if module is not None:
            module.unload()
        dev.free()
    assert np.isfinite(got).all(), "a bad page id was followed into the NaN canary"
    tol = 4e-2 if dtype == "bf16" else 2e-2
    np.testing.assert_allclose(got, ref, rtol=0, atol=tol)


@pytest.mark.gpu
@_NEEDS_RDNA
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("ps", [8, 64], ids=["rows_ps8", "tiles_ps64"])
@pytest.mark.parametrize("bad", sorted(_BAD))
@pytest.mark.parametrize("kind", ["prefill", "decode"])
def test_bad_block_table_entries_read_as_zero_pages(kind, bad, ps, dtype):
    _run(kind, dtype, ps, bad)
