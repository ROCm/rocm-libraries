# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Torch custom-op wrapper for the gfx1151 paged split-K decode kernel.

Registers ``rocke_gfx1151::paged_decode_splitk`` as a graph-capture-safe
torch custom op over vLLM's paged KV cache::

    rocke_gfx1151::paged_decode_splitk(
        q, k_cache, v_cache, block_table, seq_lens, out, scale, num_splits)

    q           : [B, Hq, D]                       f16   (decode: one token/request)
    k_cache     : [num_blocks, Hk, D/x, BS, x]     f16   x = 8
    v_cache     : [num_blocks, Hk, D, BS]          f16
    block_table : [B, max_blocks]                  int32
    seq_lens    : [B]                              int32
    out         : [B, Hq, D]                       f16   caller-allocated, in-place
    scale       : float  (1/sqrt(D)); converted to log2 space here
    num_splits  : int    (0 -> :func:`choose_num_splits`)

Optional tuning knobs, each selecting a different compiled kernel (0 / default
means "the shipped choice"); see ``PagedDecodeCfg`` for what each trades::

    d_lanes        : int  lanes per QK dot (0 -> ROCKE_PAGED_DECODE_DLANES or 16)
    block_n        : int  tokens per V vector load (0 -> derived from d_lanes)
    unroll         : int  kv blocks per loop iteration (1)
    heads_per_cta  : int  Q heads per CTA (0 -> the whole GQA ratio)
    waves_per_eu   : int  occupancy hint for the segment kernel (0 -> none)
    reduce_unroll  : int  1 -> straight-line reduce over the splits (0)

Two launches on the caller's stream: the segment kernel writes unnormalised
``(m, l, acc)`` per split into an f32 workspace, the reduce kernel merges them
into ``out``.

The out-parameter ABI is required for HIP graph replay stability, and the
workspace is cached rather than allocated per call for the same reason -- a
replayed graph reuses whatever pointers capture recorded.
"""

from __future__ import annotations

import ctypes
import math
import os
import struct
from typing import Any

_NAMESPACE = "rocke_gfx1151"
_OP_NAME = "paged_decode_splitk"
_FULL_NAME = f"{_NAMESPACE}::{_OP_NAME}"
_SCHEMA = (
    f"{_OP_NAME}(Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_table, "
    "Tensor seq_lens, Tensor(a!) out, float scale, int num_splits, "
    "int d_lanes=0, int block_n=0, int unroll=1, int heads_per_cta=0, "
    "int waves_per_eu=0, int reduce_unroll=0) -> ()"
)
_ARCH = "gfx1151"

# Lanes cooperating on one QK dot product. Process-wide rather than a call
# argument on purpose: it selects a different compiled kernel, so it belongs to
# the cache key and not to the op's ABI. See PagedDecodeCfg.d_lanes for the
# swizzle-vs-VGPR tradeoff this picks a point on.
D_LANES = int(os.environ.get("ROCKE_PAGED_DECODE_DLANES", "16"))

# (head_size, num_q_heads, num_kv_heads, dtype, kv_block_size, num_splits,
#  d_lanes, block_n, unroll, heads_per_cta, waves_per_eu, reduce_unroll)
_KERNEL_CACHE: dict[tuple, Any] = {}

# (device_index, batch, num_kv_heads, num_splits, gqa_fuse, head_size)
_WS_CACHE: dict[tuple, Any] = {}

_TORCH_LIBS: list[Any] = []
_REGISTERED = False

_TORCH_TO_ROCKE_DTYPE = {"torch.float16": "f16", "torch.bfloat16": "bf16"}

# ``Runtime.launch`` retains each packed kernarg buffer on the stream's pending
# list, because the HIP ``extra`` path requires that buffer to outlive the
# kernel. With ``record_event=False`` -- which is what graph capture needs,
# since a per-launch event would become a graph node -- only a stream drain can
# release those refs, so an undrained decode loop grows the list forever.
#
# A drain per forward is not an option: at 36 layers that would serialise decode
# outright. Drain on a coarse launch count instead. Under graph replay none of
# this Python runs at all, so the counter only advances in eager mode and during
# the finite set of captures.
_DRAIN_EVERY = 4096
_since_drain = 0


def _get_launcher(
    head_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    dtype: str,
    kv_block_size: int,
    num_splits: int,
    d_lanes: int,
    block_n: int = 0,
    unroll: int = 1,
    heads_per_cta: int = 0,
    waves_per_eu: int = 0,
    reduce_unroll: int = 0,
):
    """Compile+cache the (segment, reduce) pair for one config."""
    key = (
        head_size,
        num_q_heads,
        num_kv_heads,
        dtype,
        kv_block_size,
        num_splits,
        d_lanes,
        block_n,
        unroll,
        heads_per_cta,
        waves_per_eu,
        reduce_unroll,
    )
    hit = _KERNEL_CACHE.get(key)
    if hit is not None:
        return hit

    from kernels.gfx1151.paged_decode_splitk import (
        PagedDecodeCfg,
        build_paged_decode_splitk_reduce,
        build_paged_decode_splitk_segment,
        is_valid_spec,
    )
    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime

    # block_n is the V vector-load width and must stay inside one key
    # partition, so unless chosen explicitly it follows d_lanes.
    keys_per_sub = kv_block_size // (32 // d_lanes)
    cfg = PagedDecodeCfg(
        head_size=head_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
        kv_block_size=kv_block_size,
        num_splits=num_splits,
        d_lanes=d_lanes,
        block_n=block_n or min(8, keys_per_sub),
        unroll=unroll,
        heads_per_cta=heads_per_cta,
        waves_per_eu=waves_per_eu,
        reduce_unroll=reduce_unroll,
    )
    ok, why = is_valid_spec(cfg, _ARCH)
    if not ok:
        raise RuntimeError(f"invalid paged_decode_splitk config: {why}")

    seg_art = compile_kernel(
        build_paged_decode_splitk_segment(cfg, arch=_ARCH), arch=_ARCH
    )
    red_art = compile_kernel(
        build_paged_decode_splitk_reduce(cfg, arch=_ARCH), arch=_ARCH
    )

    rt = Runtime()
    seg_mod = rt.load_module(seg_art.hsaco)
    red_mod = rt.load_module(red_art.hsaco)
    entry = (
        cfg,
        rt,
        seg_mod.get_function(seg_art.kernel_name),
        red_mod.get_function(red_art.kernel_name),
        seg_mod,
        red_mod,
    )
    _KERNEL_CACHE[key] = entry
    return entry


def _get_workspace(cfg, batch: int, device):
    """Cached f32 workspace. Reused across calls so graph replay stays valid."""
    import torch

    key = (
        device.index if device.index is not None else 0,
        batch,
        cfg.num_kv_heads,
        cfg.num_splits,
        cfg.gqa_fuse,
        cfg.head_size,
    )
    hit = _WS_CACHE.get(key)
    if hit is not None:
        return hit

    from kernels.gfx1151.paged_decode_splitk import paged_decode_workspace_shapes

    ml_shape, acc_shape = paged_decode_workspace_shapes(cfg, batch)
    ws_m = torch.empty(ml_shape, dtype=torch.float32, device=device)
    ws_l = torch.empty(ml_shape, dtype=torch.float32, device=device)
    ws_acc = torch.empty(acc_shape, dtype=torch.float32, device=device)
    entry = (ws_m, ws_l, ws_acc)
    _WS_CACHE[key] = entry
    return entry


def _launch(
    q,
    k_cache,
    v_cache,
    block_table,
    seq_lens,
    out,
    scale,
    num_splits,
    d_lanes=0,
    block_n=0,
    unroll=1,
    heads_per_cta=0,
    waves_per_eu=0,
    reduce_unroll=0,
) -> None:
    from kernels.gfx1151.paged_decode_splitk import (
        WAVE,
        choose_num_splits,
        paged_decode_reduce_grid,
        paged_decode_segment_grid,
    )

    if q.dim() != 3:
        raise RuntimeError(f"q must be [B, Hq, D], got {tuple(q.shape)}")
    B, Hq, D = q.shape
    if k_cache.dim() != 5 or v_cache.dim() != 4:
        raise RuntimeError(
            "expected k_cache [blocks, Hk, D/x, BS, x] and v_cache "
            f"[blocks, Hk, D, BS], got {tuple(k_cache.shape)} / "
            f"{tuple(v_cache.shape)}"
        )
    Hk, kv_block_size = v_cache.shape[1], v_cache.shape[3]
    if v_cache.shape[2] != D:
        raise RuntimeError(f"v_cache head_size {v_cache.shape[2]} != q head_size {D}")
    if q.stride(-1) != 1 or out.stride(-1) != 1:
        raise RuntimeError("q and out must be contiguous along head_size")
    for name, t in (("k_cache", k_cache), ("v_cache", v_cache)):
        if not t.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")
        # The kernel indexes the cache with i32 element offsets.
        if t.numel() * t.element_size() >= 2**31:
            raise RuntimeError(f"{name} exceeds the 2 GiB i32-offset cap")

    dtype = _TORCH_TO_ROCKE_DTYPE.get(str(q.dtype))
    if dtype is None:
        raise RuntimeError(f"unsupported dtype {q.dtype}")

    if num_splits <= 0:
        num_splits = choose_num_splits(B, Hk)

    cfg, rt, seg_fn, red_fn, _seg_mod, _red_mod = _get_launcher(
        D,
        Hq,
        Hk,
        dtype,
        kv_block_size,
        num_splits,
        d_lanes or D_LANES,
        block_n,
        unroll,
        heads_per_cta,
        waves_per_eu,
        reduce_unroll,
    )
    ws_m, ws_l, ws_acc = _get_workspace(cfg, B, q.device)

    scale_log2 = float(scale * math.log2(math.e))
    block = (WAVE, 1, 1)

    # ABI: 8 pointers, 1 float, 4 int32. Every field is already at its natural
    # alignment (8 ptrs = 64 B, then f32 at 64, i32s at 68/72/76/80), so no pad
    # byte is needed here -- unlike the swapqk paged tail, which appends a
    # pointer after an odd number of i32s.
    seg_args = struct.pack(
        "<QQQQQQQQfiiii",
        q.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        block_table.data_ptr(),
        seq_lens.data_ptr(),
        ws_m.data_ptr(),
        ws_l.data_ptr(),
        ws_acc.data_ptr(),
        scale_log2,
        q.stride(0),
        q.stride(1),
        block_table.stride(0),
        block_table.numel(),
    )
    red_args = struct.pack(
        "<QQQQii",
        ws_m.data_ptr(),
        ws_l.data_ptr(),
        ws_acc.data_ptr(),
        out.data_ptr(),
        out.stride(0),
        out.stride(1),
    )

    from rocke.runtime.torch_interop import resolve_stream

    stream = resolve_stream(0, device=q.device)

    rt.launch(
        seg_fn,
        paged_decode_segment_grid(cfg, B),
        block,
        _buf(seg_args),
        stream=stream,
    )
    rt.launch(
        red_fn,
        paged_decode_reduce_grid(cfg, B),
        block,
        _buf(red_args),
        stream=stream,
    )
    # ws_m/ws_l/ws_acc stay alive in _WS_CACHE, so the async launches above
    # cannot outlive their buffers.

    global _since_drain
    _since_drain += 2
    if _since_drain >= _DRAIN_EVERY:
        # hipStreamSynchronize is illegal mid-capture, and a capturing stream
        # has no pending host-side refs to reclaim anyway.
        import torch

        if not torch.cuda.is_current_stream_capturing():
            _since_drain = 0
            rt.wait_stream(stream)


def _buf(packed: bytes):
    return (ctypes.c_uint8 * len(packed)).from_buffer(bytearray(packed))


def paged_decode_splitk_impl(
    q,
    k_cache,
    v_cache,
    block_table,
    seq_lens,
    out,
    scale: float,
    num_splits: int,
    d_lanes: int = 0,
    block_n: int = 0,
    unroll: int = 1,
    heads_per_cta: int = 0,
    waves_per_eu: int = 0,
    reduce_unroll: int = 0,
) -> None:
    """GPU implementation of rocke_gfx1151::paged_decode_splitk."""
    _launch(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        out,
        scale,
        num_splits,
        d_lanes,
        block_n,
        unroll,
        heads_per_cta,
        waves_per_eu,
        reduce_unroll,
    )


def paged_decode_splitk_meta(
    q,
    k_cache,
    v_cache,
    block_table,
    seq_lens,
    out,
    scale: float,
    num_splits: int,
    d_lanes: int = 0,
    block_n: int = 0,
    unroll: int = 1,
    heads_per_cta: int = 0,
    waves_per_eu: int = 0,
    reduce_unroll: int = 0,
) -> None:
    """Shape-only meta implementation (no GPU, used by torch.compile tracing)."""
    return None


def register_torch_custom_ops(namespace: str = _NAMESPACE) -> bool:
    """Register paged_decode_splitk as a torch custom op. Idempotent."""
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        import torch
    except Exception:
        return False

    def_lib = torch.library.Library(namespace, "DEF")
    impl_lib = torch.library.Library(namespace, "IMPL")
    meta_lib = torch.library.Library(namespace, "IMPL")
    try:
        def_lib.define(_SCHEMA)
        impl_lib.impl(_OP_NAME, paged_decode_splitk_impl, "CompositeExplicitAutograd")
        meta_lib.impl(_OP_NAME, paged_decode_splitk_meta, "Meta")
    except RuntimeError as exc:
        if "Only a single TORCH_LIBRARY" not in str(exc) and "already" not in str(exc):
            raise

    _TORCH_LIBS.extend([def_lib, impl_lib, meta_lib])
    _REGISTERED = True
    return True
