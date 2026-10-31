# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Weighted segment sum of the split-KV reduce (its pass 3).

The gfx942 and gfx950 tiled-3D reduce kernels share this pass. One wave64
CTA owns one ``(token, head)`` row. Each segment's normalized weight
``factor[s] = exp2(segm_max[s] - overall_max)`` is already in LDS; this pass
computes ``out[d] = inv_l * sum_s factor[s] * segm_output[s, d]`` and stores
it in the output dtype.

Layout: every lane loads ``vec`` consecutive f32 head dims in one load, so
``head_size / vec`` lanes cover one segment row and the wave splits into
``groups = 64 / (head_size / vec)`` lane groups. ``vec`` is 4 (16-byte loads)
for a long segment list; a short list (a large grid of rows) uses
``vec = head_size / 64`` when that is 2 or 4, so one group covers the row and
there is no cross-group step. Group ``g`` sums a contiguous run of
``ceil(num_segments / groups)`` segments (contiguous, so the per-segment
factor reads merge into wide LDS reads); the loop issues ``unroll`` segment
loads per trip so many loads are in flight at once. Groups ``1..groups-1``
then hand their partial sums to group 0 through LDS, and group 0 normalizes
and writes. A head size whose ``head_size / vec`` does not divide 64 (192)
leaves the remaining lanes idle: they shadow group 0's columns and never
store.

The C++ twin is ``rocke_split_kv_reduce_weighted_segments`` in
``platform/cpp/instances/common/attention_split_kv_reduce.cpp``; keep the
builder-call order identical.
"""

from __future__ import annotations

from rocke.core.ir import F32, Type, Value

# f32 elements per lane per segment for a long segment list (16-byte loads).
_VEC = 4
# Segment loads one lane issues per loop trip: 16 when the wave splits into
# several lane groups, 32 when one group (head size 192 / 256) walks every
# segment alone.
_UNROLL = 16
_UNROLL_ONE_GROUP = 32
# From this many segments on, the reduce asks for one wave per SIMD so the
# backend keeps more segment loads in flight, and splits the wave into lane
# groups. Such a split only comes with a small grid (few rows), so the lost
# occupancy costs nothing there.
_ONE_WAVE_MIN_SEGMENTS = 32


def reduce_waves_per_eu(requested, num_segments: int):
    """Occupancy hint for the reduce kernel: the spec's value, else 1 for a
    long segment list, else None (backend default)."""
    if requested is not None:
        return requested
    return 1 if num_segments >= _ONE_WAVE_MIN_SEGMENTS else None


def reduce_weighted_segments(
    b,
    *,
    head_size: int,
    num_segments: int,
    threads: int,
    dtype: Type,
    tid: Value,
    q_token: Value,
    q_head: Value,
    seg_out: Value,
    out: Value,
    factor_lds: Value,
    seg_acc_desc,
    out_desc,
    inv_l: Value,
    zero_f: Value,
) -> None:
    """Emit the weighted segment sum, normalize, and store one output row."""
    vec = _VEC
    row_vec = head_size // threads
    if (
        num_segments < _ONE_WAVE_MIN_SEGMENTS
        and row_vec in (2, 4)
        and row_vec * threads == head_size
    ):
        vec = row_vec
    lanes_per_row = head_size // vec
    if lanes_per_row * vec != head_size or lanes_per_row > threads:
        raise ValueError(
            f"split-KV reduce: head_size {head_size} needs head_size / {vec} "
            f"<= {threads} lanes"
        )
    groups = threads // lanes_per_row
    seg_per_group = (num_segments + groups - 1) // groups
    # Group g owns the contiguous segments [g * seg_per_group, (g + 1) *
    # seg_per_group). Only the last group can run past num_segments, and only
    # in its slots >= `always_valid`; those slots go to the range-checked tail.
    always_valid = max(
        0, min(seg_per_group, num_segments - (groups - 1) * seg_per_group)
    )
    unroll = _UNROLL_ONE_GROUP if groups == 1 else _UNROLL
    trips = always_valid // unroll
    tail = seg_per_group - trips * unroll

    lpr_v = b.const_i32(lanes_per_row)
    grp = b.div(tid, lpr_v)
    col = b.mul(b.sub(tid, b.mul(grp, lpr_v)), b.const_i32(vec))
    lane_ok = None
    if groups * lanes_per_row != threads:
        lane_ok = b.cmp_lt(grp, b.const_i32(groups))
        grp = b.select(lane_ok, grp, b.const_i32(0))
    seg0 = b.mul(grp, b.const_i32(seg_per_group))

    def _accumulate(accs, seg: Value, in_rng):
        seg_safe = seg if in_rng is None else b.select(in_rng, seg, b.const_i32(0))
        factor = b.vec_extract(b.smem_load_vN_f32(factor_lds, seg_safe, n=1), 0)
        if in_rng is not None:
            factor = b.select(in_rng, factor, zero_f)
        idx, _ = seg_acc_desc.offset(
            b, token=q_token, head=q_head, seg=seg_safe, dim=col
        )
        vals = b.global_load_vN(seg_out, idx, F32, vec, align=4 * vec)
        return [
            b.fadd(accs[k], b.fmul(b.vec_extract(vals, k), factor)) for k in range(vec)
        ]

    accs = [zero_f] * vec
    if trips > 0:
        loop = b.scf_for_iter(
            b.const_i32(0),
            b.const_i32(trips),
            b.const_i32(1),
            [(f"racc{k}", zero_f) for k in range(vec)],
            iv_name="rtrip",
        )
        with loop as (trip, accs_in):
            base = b.add(seg0, b.mul(trip, b.const_i32(unroll)))
            body_accs = list(accs_in)
            for u in range(unroll):
                seg = base if u == 0 else b.add(base, b.const_i32(u))
                body_accs = _accumulate(body_accs, seg, None)
            b.scf_yield(*body_accs)
        accs = list(loop.results)
    for u in range(tail):
        slot = trips * unroll + u
        seg = b.add(seg0, b.const_i32(slot))
        in_rng = None
        if slot >= always_valid:
            in_rng = b.cmp_lt(seg, b.const_i32(num_segments))
        accs = _accumulate(accs, seg, in_rng)

    writer = lane_ok
    if groups > 1:
        part_lds = b.smem_alloc_f32([groups - 1, head_size], name_hint="seg_part")
        with b.scf_if(b.cmp_gt(grp, b.const_i32(0))):
            b.smem_store_vN_f32(
                part_lds,
                [b.sub(grp, b.const_i32(1)), col],
                b.vec_pack(accs, F32),
                vec,
            )
        b.sync()
        writer = b.cmp_eq(grp, b.const_i32(0))

    def _finish() -> None:
        final = list(accs)
        for g in range(1, groups):
            part = b.smem_load_vN_f32(part_lds, b.const_i32(g - 1), col, n=vec)
            final = [b.fadd(final[k], b.vec_extract(part, k)) for k in range(vec)]
        outs = [b.cast_f32_to(b.fmul(final[k], inv_l), dtype) for k in range(vec)]
        out_idx, _ = out_desc.offset(b, token=q_token, head=q_head, dim=col)
        b.global_store_vN(out, out_idx, b.vec_pack(outs, dtype), vec, align=2 * vec)

    if writer is None:
        _finish()
    else:
        with b.scf_if(writer):
            _finish()
