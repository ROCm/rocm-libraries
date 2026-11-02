# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""KV-segment split for the dense prefill bodies (gfx942 and gfx950).

With ``kv_segments = S > 1`` each (query block, head, batch) work item is served
by ``S`` CTAs, each over one contiguous chunk of the work item's KV tiles. A
segment writes its unnormalized FP32 partial output plus its running max (log2
domain) and denominator; the split-KV combine kernel of the unified tiled path
merges the segments into the final output. The workspace layout is that combine
kernel's: ``segm_output [B*Sq, Hq, S, D]`` and ``segm_max`` / ``segm_expsum``
``[B*Sq, Hq, S]``, token-major like the dense ``[B, Sq, Hq, D]`` output.

A segment where a row sees no key keeps the body's finite mask sentinel as its
max, so the combine weights it by ``exp2(sentinel - max) = 0``. Segment 0 always
holds key 0, so every row has at least one real segment.
"""

from __future__ import annotations

from rocke.core.ir import F32, PtrType

from kernels.common.attention_dense_spec import INT32_LIMIT

SEGMENT_PARAMS = ("segm_output_ptr", "segm_max_ptr", "segm_expsum_ptr")
MAX_KV_SEGMENTS = 64


def segment_extent(batch, seqlen_q, num_query_heads, head_size, segments) -> int:
    """Element count of ``segm_output``: the largest index the segment store and
    the combine kernel compute in signed 32-bit arithmetic."""
    return (
        int(batch)
        * int(seqlen_q)
        * int(num_query_heads)
        * int(segments)
        * int(head_size)
    )


def max_kv_segments_for_extent(
    batch, seqlen_q, num_query_heads, head_size, max_segments=MAX_KV_SEGMENTS
) -> int:
    """Largest power of two ``S <= max_segments`` whose workspace extent stays below
    the 32-bit addressing limit (at least 1)."""
    s = 1
    while s * 2 <= max_segments and (
        segment_extent(batch, seqlen_q, num_query_heads, head_size, s * 2) < INT32_LIMIT
    ):
        s *= 2
    return s


def check_kv_segments(spec) -> tuple[bool, str]:
    """Shared admission for ``spec.kv_segments``: the default grid only, without a
    sliding window, sinks or LSE output (each would need its own segment rule)."""
    s = spec.kv_segments
    if type(s) is not int or not 1 <= s <= MAX_KV_SEGMENTS:
        return False, f"kv_segments must be an int in [1, {MAX_KV_SEGMENTS}], got {s!r}"
    if s > 1 and (
        spec.persistent
        or spec.sliding_window
        or spec.use_sinks
        or spec.emit_lse
        or spec.ragged
        or spec.varlen
        or spec.paged
    ):
        return False, (
            "kv_segments > 1 runs on the default grid only, without sliding window, "
            "sinks, LSE output, ragged, varlen or paged"
        )
    extent = segment_extent(
        spec.batch, spec.seqlen_q, spec.num_query_heads, spec.head_size, s
    )
    if s > 1 and extent >= INT32_LIMIT:
        return False, (
            f"kv_segments={s} workspace extent is {extent} elements, at or past the "
            f"32-bit addressing limit ({INT32_LIMIT})"
        )
    return True, ""


def declare_segment_params(b):
    """The three workspace pointers, appended after every other kernel param."""
    return tuple(
        b.param(
            name,
            PtrType(F32, "global"),
            noalias=True,
            writeonly=True,
            align=16 if name == "segm_output_ptr" else 4,
        )
        for name in SEGMENT_PARAMS
    )


def segment_tile_range(b, seg, lo, hi, segments):
    """``[seg_lo, seg_hi)``: segment ``seg`` of ``segments`` contiguous chunks of
    ``ceil((hi - lo) / segments)`` tiles over ``[lo, hi)``, clamped to ``hi``."""
    chunk = b.div(
        b.add(b.sub(hi, lo), b.const_i32(segments - 1)), b.const_i32(segments)
    )
    seg_lo = b.add(lo, b.mul(seg, chunk))
    seg_lo = b.select(b.cmp_lt(seg_lo, hi), seg_lo, hi)
    seg_hi = b.add(seg_lo, chunk)
    seg_hi = b.select(b.cmp_lt(seg_hi, hi), seg_hi, hi)
    return seg_lo, seg_hi


def emit_segment_store(
    b,
    params,
    *,
    token,
    head,
    num_heads,
    seg,
    segments,
    head_size,
    o_acc,
    m,
    denom,
    lane_h,
):
    """Write one segment's state from the transposed 32x32 C[dim, query] layout.

    ``o_acc[dt]`` holds 16 FP32 values per lane: dims ``dt*32 + g*8 + lane_h*4 + kk``
    of query ``token``. Both lane halves hold the same row's ``m`` / ``l``; the
    lower half writes them."""
    segm_out, segm_max, segm_l = params
    ml_idx = b.add(
        b.mul(b.add(b.mul(token, num_heads), head), b.const_i32(segments)), seg
    )
    row_base = b.mul(ml_idx, b.const_i32(head_size))
    d_half = b.mul(lane_h, b.const_i32(4))
    for dt in range(len(o_acc)):
        for g in range(4):
            d0 = b.add(b.const_i32(dt * 32 + g * 8), d_half)
            vals = [b.vec_extract(o_acc[dt], g * 4 + kk) for kk in range(4)]
            b.global_store_vN(
                segm_out, b.add(row_base, d0), b.vec_pack(vals, F32), 4, align=16
            )
    with b.scf_if(b.cmp_eq(lane_h, b.const_i32(0))):
        b.global_store(segm_max, ml_idx, m, align=4)
        b.global_store(segm_l, ml_idx, denom, align=4)


def allocate_segment_workspace(spec, device):
    """FP32 workspace tensors keyed by :data:`SEGMENT_PARAMS`."""
    import torch

    shape = (spec.batch * spec.seqlen_q, spec.num_query_heads, spec.kv_segments)
    return {
        "segm_output_ptr": torch.empty(
            shape + (spec.head_size,), dtype=torch.float32, device=device
        ),
        "segm_max_ptr": torch.empty(shape, dtype=torch.float32, device=device),
        "segm_expsum_ptr": torch.empty(shape, dtype=torch.float32, device=device),
    }


_COMBINE_LAUNCHER_CACHE: dict = {}


def _combine_impl(arch: str):
    if arch == "gfx942":
        from kernels.gfx942.attention_tiled_3d import (
            UnifiedAttentionReduceTiledSpec,
            build_unified_attention_reduce_tiled,
        )
    else:
        from kernels.gfx950.attention_tiled_3d import (
            UnifiedAttentionReduceTiledSpec,
            build_unified_attention_reduce_tiled,
        )
    return UnifiedAttentionReduceTiledSpec, build_unified_attention_reduce_tiled


def combine_spec(spec, arch: str):
    spec_type, _ = _combine_impl(arch)
    return spec_type(
        head_size=spec.head_size,
        num_query_heads=spec.num_query_heads,
        num_kv_heads=spec.num_kv_heads,
        dtype=spec.dtype,
        num_segments=spec.kv_segments,
    )


def launch_segment_combine(spec, arch: str, out, workspace, stream: int) -> None:
    """Merge the segments in ``workspace`` into ``out`` ([B, Sq, Hq, D])."""
    from kernels.common.attention_unified import _reduce_signature
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime import KernelLauncher, LaunchConfig

    red = combine_spec(spec, arch)
    key = (arch, red)
    launcher = _COMBINE_LAUNCHER_CACHE.get(key)
    if launcher is None:
        _, build = _combine_impl(arch)
        art = compile_kernel(
            build(red, arch=arch), arch=arch, backend="python", capture_ir_text=False
        )
        launcher = KernelLauncher(
            hsaco=art.hsaco,
            kernel_name=art.kernel_name,
            signature=_reduce_signature(spec.dtype),
        )
        _COMBINE_LAUNCHER_CACHE[key] = launcher
    import torch

    vals = dict(workspace)
    vals["output_ptr"] = out
    # Unused by the combine body; any int32 buffer satisfies the ABI.
    vals["seq_lens_ptr"] = workspace["segm_max_ptr"].view(torch.int32)
    launcher(
        vals,
        config=LaunchConfig(
            grid=(spec.batch * spec.seqlen_q, spec.num_query_heads, 1),
            block=(64, 1, 1),
            stream=int(stream),
        ),
    )
