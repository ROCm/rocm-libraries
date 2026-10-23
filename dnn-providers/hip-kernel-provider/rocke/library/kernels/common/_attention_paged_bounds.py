# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Device containment for the versioned one-query paged decode pipeline."""

from rocke.core.ir import I32, PtrType


class PagedBounds:
    """Private per-CTA state; each lane owns its sticky LDS failure slot."""

    def __init__(self, b, sliding_window: int = 0):
        self.b = b
        self.sliding_window = sliding_window
        self.status = b.param("metadata_status_ptr", PtrType(I32, "global"), align=4)
        self.pages = b.param("physical_pages", I32)
        self.capacity = b.param("max_seqlen_k", I32)
        self.cols = b.param("table_cols", I32)
        self.k_bytes = b.param("k_span_bytes", I32)
        self.v_bytes = b.param("v_span_bytes", I32)

    def start(self, seq, head, seg, tid, heads, segments):
        b = self.b
        self.tid = tid
        self.slot = b.add(
            b.mul(b.add(b.mul(seq, b.const_i32(heads)), head), b.const_i32(segments)),
            seg,
        )
        self.failures = b.smem_alloc(I32, [64], name_hint="metadata_failures")
        b.smem_store_vN(self.failures, [tid], b.const_i32(0), 1)

    def mark(self, invalid):
        b = self.b
        old = b.vec_extract(b.smem_load_vN(self.failures, self.tid, dtype=I32, n=1), 0)
        b.smem_store_vN(
            self.failures,
            [self.tid],
            b.select(invalid, b.const_i32(1), old),
            1,
        )

    def finish(self):
        b = self.b
        failed = b.vec_extract(
            b.smem_load_vN(self.failures, self.tid, dtype=I32, n=1), 0
        )
        any_failed = b.cmp_ne(
            b.wave_any(
                b.select(
                    b.cmp_ne(failed, b.const_i32(0)), b.const_i32(1), b.const_i32(0)
                )
            ),
            b.const_i32(0),
        )
        with b.scf_if(b.cmp_eq(self.tid, b.const_i32(0))):
            b.global_store(
                self.status,
                self.slot,
                b.select(any_failed, b.const_i32(1), b.const_i32(0)),
                align=4,
            )

    def physical(self, table, base, page, live):
        b = self.b
        lookup_ok = b.land(
            live,
            b.land(b.cmp_ge(page, b.const_i32(0)), b.cmp_lt(page, self.cols)),
        )
        safe_page = b.select(lookup_ok, page, b.const_i32(0))
        physical = b.masked_global_load(
            table,
            b.add(base, safe_page),
            lookup_ok,
            b.const_i32(0),
            dtype=I32,
            align=4,
        )
        physical_ok = b.land(
            b.cmp_ge(physical, b.const_i32(0)), b.cmp_lt(physical, self.pages)
        )
        good = b.land(lookup_ok, physical_ok)
        false_ = b.cmp_ne(b.const_i32(0), b.const_i32(0))
        self.mark(b.land(live, b.cmp_eq(good, false_)))
        return b.select(good, physical, b.const_i32(0))

    def offset(
        self, b, table, base, head, tile, linear, length, hd, page, tile_size, heads
    ):
        b = self.b
        token = b.add(
            b.mul(tile, b.const_i32(tile_size)), b.div(linear, b.const_i32(hd))
        )
        live = b.land(b.cmp_ge(token, b.const_i32(0)), b.cmp_lt(token, length))
        if self.sliding_window > 0:
            window_start = b.smax(
                b.const_i32(0), b.sub(length, b.const_i32(self.sliding_window))
            )
            live = b.land(live, b.cmp_ge(token, window_start))
        safe_token = b.select(live, token, b.const_i32(0))
        physical = self.physical(
            table, base, b.div(safe_token, b.const_i32(page)), live
        )
        page_off = b.mul(physical, b.const_i32(page * heads * hd * 2))
        row_off = b.mul(
            b.mod(safe_token, b.const_i32(page)), b.const_i32(heads * hd * 2)
        )
        head_off = b.mul(head, b.const_i32(hd * 2))
        dim_off = b.mul(b.mod(linear, b.const_i32(hd)), b.const_i32(2))
        offset = b.add(b.add(page_off, row_off), b.add(head_off, dim_off))
        return b.select(live, offset, b.const_i32(0x7FFFFFFF))


def neutral_segment(
    b, seq, head, seg, tid, out, maximum, expsum, hd, nqk, nqh, segments
):
    """One writer per real element; never redirect inactive query rows to row zero."""
    for qh in range(nqk):
        h = b.add(b.mul(head, b.const_i32(nqk)), b.const_i32(qh))
        ml = b.add(
            b.mul(b.add(b.mul(seq, b.const_i32(nqh)), h), b.const_i32(segments)),
            seg,
        )
        with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
            b.global_store(maximum, ml, b.const_f32(float("-inf")), align=4)
            b.global_store(expsum, ml, b.const_f32(0.0), align=4)
        for chunk in range(hd // 64):
            dim = b.add(b.const_i32(chunk * 64), tid)
            idx = b.add(b.mul(ml, b.const_i32(hd)), dim)
            b.global_store(out, idx, b.const_f32(0.0), align=4)


def bounded_reduce_params(b):
    status = b.param(
        "metadata_status_ptr", PtrType(I32, "global"), readonly=True, align=4
    )
    count = b.param("num_seqs", I32)
    return status, count


def reject_failed_sequence(
    b, status, seq, head, tid, out, dtype, hd, nqh, nkh, segments
):
    false_ = b.cmp_ne(b.const_i32(0), b.const_i32(0))
    failed = false_
    base = b.mul(seq, b.const_i32(nkh * segments))
    for chunk in range((nkh * segments + 63) // 64):
        slot = b.add(b.const_i32(chunk * 64), tid)
        live = b.cmp_lt(slot, b.const_i32(nkh * segments))
        value = b.masked_global_load(
            status, b.add(base, slot), live, b.const_i32(0), dtype=I32, align=4
        )
        failed = b.lor(failed, b.cmp_ne(value, b.const_i32(0)))
    failed = b.cmp_ne(
        b.wave_any(b.select(failed, b.const_i32(1), b.const_i32(0))), b.const_i32(0)
    )
    with b.scf_if(failed):
        base_out = b.mul(b.add(b.mul(seq, b.const_i32(nqh)), head), b.const_i32(hd))
        nan = b.cast_f32_to(b.const_f32(float("nan")), dtype)
        for chunk in range(hd // 64):
            dim = b.add(b.const_i32(chunk * 64), tid)
            b.global_store(out, b.add(base_out, dim), nan, align=2)
        b.ret()
