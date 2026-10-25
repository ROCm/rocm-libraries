// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.instances.common._attention_bwd_band.cpp -- C port of
 * library/kernels/common/_attention_bwd_band.py: q_tile_range,
 * tile_inside_band, is_edge_tile, tile_class, interior_q_range, dkv_store_rule
 * and the RuntimeBand / NoMaskKTail mask policies.
 *
 * Builder calls are made in the order Python makes them, so the emitted IR is
 * byte-identical; nested Python call arguments are bound to locals in Python's
 * left-to-right order. A Python raise is a thrown ckc::Error, caught at each
 * public entry and recorded on the builder.
 */

#include "rocke/helper_rocke.instances.common._attention_bwd_band.h"

#include <stdio.h>

#include "rocke/error_boundary.hpp"

namespace
{

using Value = rocke_value_t;
using Builder = rocke_ir_builder_t;

Value* c32(Builder* b, int64_t v)
{
    return rocke_b_const_i32(b, v);
}

// as_i32(b, x): an int as an i32 constant, an SSA value unchanged.
Value* as_i32(Builder* b, const rocke_int_or_value_t& x)
{
    return x.is_value ? x.v : c32(b, x.i);
}

void require_positive_tiles(int k_m0, int k_n0)
{
    if(k_m0 < 1 || k_n0 < 1)
    {
        ckc::raise_status(ROCKE_ERR_VALUE, "tile sizes must be positive");
    }
}

// floor(a / m) for a signed i32 `a` and a positive constant `m`.
Value* floor_div(Builder* b, Value* a, int m)
{
    Value* cm = c32(b, m);
    Value* zero0 = c32(b, 0);
    Value* m1 = c32(b, (int64_t)m - 1);
    Value* sh = rocke_b_sub(b, m1, a);
    Value* q = rocke_b_div(b, sh, cm);
    Value* neg = rocke_b_sub(b, zero0, q);
    Value* zero1 = c32(b, 0);
    Value* is_neg = rocke_b_cmp_lt(b, a, zero1);
    Value* pos = rocke_b_div(b, a, cm);
    return rocke_b_select(b, is_neg, neg, pos);
}

rocke_status_t q_tile_range_impl(Builder* b,
                                 rocke_band_emitter_t* band,
                                 Value* k0,
                                 int k_n0,
                                 int k_m0,
                                 bool uniform,
                                 Value** out_qt_start,
                                 Value** out_n_qt)
{
    if(k_n0 < 1 || k_m0 < 1)
    {
        ckc::raise_status(ROCKE_ERR_VALUE, "tile sizes must be positive");
    }
    Value* zero = c32(b, 0);
    Value* cm = c32(b, k_m0);
    Value* span = c32(b, (int64_t)k_n0 - 1);
    Value* k_end = rocke_b_add(b, k0, span);
    Value* k_last = rocke_b_smin(b, k_end, band->sk_minus1);
    Value* one = c32(b, 1);
    Value* sq_m1 = rocke_b_sub(b, band->sq, one);
    Value* off = band->off;
    Value* k_first_c = k0;
    if(off != nullptr)
    {
        k_first_c = rocke_b_sub(b, k0, off);
    }
    Value* k_last_c = k_last;
    if(off != nullptr)
    {
        k_last_c = rocke_b_sub(b, k_last, off);
    }
    Value* q_lo = nullptr;
    if(!band->right.has)
    {
        q_lo = zero;
    }
    else
    {
        Value* r = as_i32(b, band->right.bound);
        q_lo = rocke_b_sub(b, k_first_c, r);
        if(band->right.bound.is_value)
        {
            Value* neg = rocke_b_cmp_lt(b, r, zero);
            q_lo = rocke_b_select(b, neg, zero, q_lo);
        }
        q_lo = rocke_b_smax(b, q_lo, zero);
    }
    Value* q_hi = nullptr;
    if(!band->left.has)
    {
        q_hi = sq_m1;
    }
    else
    {
        Value* lft = as_i32(b, band->left.bound);
        q_hi = rocke_b_add(b, k_last_c, lft);
        if(band->left.bound.is_value)
        {
            Value* neg = rocke_b_cmp_lt(b, lft, zero);
            q_hi = rocke_b_select(b, neg, sq_m1, q_hi);
        }
        q_hi = rocke_b_smin(b, q_hi, sq_m1);
    }
    // Empty when the tile has no live key or the band misses every row.
    Value* no_key = rocke_b_cmp_lt(b, k_last, k0);
    Value* no_row = rocke_b_cmp_lt(b, q_hi, q_lo);
    Value* dead = rocke_b_lor(b, no_key, no_row);
    Value* m1 = c32(b, -1);
    q_hi = rocke_b_select(b, dead, m1, q_hi);
    Value* qt_start = rocke_b_div(b, q_lo, cm);
    Value* m1b = c32(b, -1);
    Value* hi_pos = rocke_b_smax(b, q_hi, m1b);
    Value* hi_sh = rocke_b_add(b, hi_pos, cm);
    Value* qt_stop = rocke_b_div(b, hi_sh, cm);
    Value* count = rocke_b_sub(b, qt_stop, qt_start);
    Value* n_qt = rocke_b_smax(b, count, zero);
    if(uniform)
    {
        qt_start = rocke_b_readfirstlane(b, qt_start);
        n_qt = rocke_b_readfirstlane(b, n_qt);
    }
    *out_qt_start = qt_start;
    *out_n_qt = n_qt;
    return ROCKE_OK;
}

Value* tile_inside_band_impl(Builder* b,
                             rocke_band_emitter_t* band,
                             Value* q_first,
                             Value* q_last,
                             Value* k_first,
                             Value* k_last)
{
    Value* zero = c32(b, 0);
    Value* inside = nullptr;
    Value* off = band->off;
    if(band->right.has)
    {
        Value* r = as_i32(b, band->right.bound);
        Value* centre = q_first;
        if(off != nullptr)
        {
            centre = rocke_b_add(b, q_first, off);
        }
        Value* edge = rocke_b_add(b, centre, r);
        Value* ok = rocke_b_cmp_le(b, k_last, edge);
        if(band->right.bound.is_value)
        {
            Value* neg = rocke_b_cmp_lt(b, r, zero);
            ok = rocke_b_lor(b, neg, ok);
        }
        inside = ok;
    }
    if(band->left.has)
    {
        Value* lft = as_i32(b, band->left.bound);
        Value* centre = q_last;
        if(off != nullptr)
        {
            centre = rocke_b_add(b, q_last, off);
        }
        Value* edge = rocke_b_sub(b, centre, lft);
        Value* ok = rocke_b_cmp_ge(b, k_first, edge);
        if(band->left.bound.is_value)
        {
            Value* neg = rocke_b_cmp_lt(b, lft, zero);
            ok = rocke_b_lor(b, neg, ok);
        }
        inside = (inside == nullptr) ? ok : rocke_b_land(b, inside, ok);
    }
    if(inside == nullptr)
    {
        inside = rocke_b_cmp_le(b, zero, zero); // no band limit: always inside
    }
    return inside;
}

Value* is_edge_tile_impl(
    Builder* b, rocke_band_emitter_t* band, Value* qt, Value* k0, int k_m0, int k_n0, bool masked)
{
    Value* cm = c32(b, k_m0);
    Value* q_first = rocke_b_mul(b, qt, cm);
    Value* cm1 = c32(b, (int64_t)k_m0 - 1);
    Value* q_last = rocke_b_add(b, q_first, cm1);
    Value* cn1 = c32(b, (int64_t)k_n0 - 1);
    Value* k_last = rocke_b_add(b, k0, cn1);
    Value* q_in = rocke_b_cmp_lt(b, q_last, band->sq);
    Value* k_in = rocke_b_cmp_lt(b, k_last, band->sk);
    Value* interior = rocke_b_land(b, q_in, k_in);
    if(masked)
    {
        Value* inside = tile_inside_band_impl(b, band, q_first, q_last, k0, k_last);
        interior = rocke_b_land(b, interior, inside);
    }
    return rocke_b_lnot(b, interior);
}

Value* tile_class_impl(Builder* b,
                       rocke_band_emitter_t* band,
                       Value* qt,
                       Value* k0,
                       int k_m0,
                       int k_n0,
                       bool masked,
                       bool uniform)
{
    Value* edge = is_edge_tile_impl(b, band, qt, k0, k_m0, k_n0, masked);
    Value* c_edge = c32(b, ROCKE_ATTN_BWD_TILE_EDGE);
    Value* c_interior = c32(b, ROCKE_ATTN_BWD_TILE_INTERIOR);
    Value* cls = rocke_b_select(b, edge, c_edge, c_interior);
    return uniform ? rocke_b_readfirstlane(b, cls) : cls;
}

rocke_status_t interior_q_range_impl(Builder* b,
                                     rocke_band_emitter_t* band,
                                     Value* k0,
                                     Value* qt_start,
                                     Value* n_qt,
                                     int k_m0,
                                     int k_n0,
                                     bool masked,
                                     bool uniform,
                                     Value** out_i0,
                                     Value** out_i1)
{
    Value* zero = c32(b, 0);
    Value* qt_end = rocke_b_add(b, qt_start, n_qt);
    Value* cn1 = c32(b, (int64_t)k_n0 - 1);
    Value* k_last = rocke_b_add(b, k0, cn1);
    Value* lo = qt_start;
    Value* cm = c32(b, k_m0);
    Value* rows = rocke_b_div(b, band->sq, cm);
    Value* hi = rocke_b_smin(b, qt_end, rows);
    Value* off = band->off;
    bool has_right = masked && band->right.has;
    bool has_left = masked && band->left.has;
    if(has_right)
    {
        // q_first >= k_last - off - right  ->  qt >= ceil((...) / k_m0)
        Value* r = as_i32(b, band->right.bound);
        Value* need = nullptr;
        if(off == nullptr)
        {
            need = rocke_b_sub(b, k_last, r);
        }
        else
        {
            Value* shifted = rocke_b_sub(b, k_last, off);
            need = rocke_b_sub(b, shifted, r);
        }
        Value* neg_need = rocke_b_sub(b, zero, need);
        Value* fl = floor_div(b, neg_need, k_m0);
        Value* first = rocke_b_sub(b, zero, fl);
        if(band->right.bound.is_value)
        {
            Value* neg = rocke_b_cmp_lt(b, r, zero);
            first = rocke_b_select(b, neg, lo, first);
        }
        lo = rocke_b_smax(b, lo, first);
    }
    if(has_left)
    {
        // q_last <= k0 - off + left  ->  qt <= floor((... - k_m0 + 1) / k_m0)
        Value* lft = as_i32(b, band->left.bound);
        Value* top = nullptr;
        if(off == nullptr)
        {
            top = rocke_b_add(b, k0, lft);
        }
        else
        {
            Value* shifted = rocke_b_sub(b, k0, off);
            top = rocke_b_add(b, shifted, lft);
        }
        Value* cm1 = c32(b, (int64_t)k_m0 - 1);
        Value* top_m = rocke_b_sub(b, top, cm1);
        Value* last = floor_div(b, top_m, k_m0);
        Value* one = c32(b, 1);
        Value* stop = rocke_b_add(b, last, one);
        if(band->left.bound.is_value)
        {
            Value* neg = rocke_b_cmp_lt(b, lft, zero);
            stop = rocke_b_select(b, neg, hi, stop);
        }
        hi = rocke_b_smin(b, hi, stop);
    }
    Value* live_kv = rocke_b_cmp_lt(b, k_last, band->sk);
    hi = rocke_b_select(b, live_kv, hi, lo); // tile past len_kv: none
    Value* i0 = rocke_b_smin(b, lo, qt_end);
    Value* hi_c = rocke_b_smin(b, hi, qt_end);
    Value* i1 = rocke_b_smax(b, hi_c, i0);
    if(uniform)
    {
        i0 = rocke_b_readfirstlane(b, i0);
        i1 = rocke_b_readfirstlane(b, i1);
    }
    *out_i0 = i0;
    *out_i1 = i1;
    return ROCKE_OK;
}

// MaskPolicy._row_lo_hi: the unclamped-validity (lo, hi) key limits of row q.
void row_lo_hi(
    rocke_attn_bwd_mask_policy_t* self, Builder* b, Value* q, Value** out_lo, Value** out_hi)
{
    rocke_band_emitter_t* band = &self->band;
    if(self->kind == ROCKE_ATTN_BWD_MASK_RUNTIME_BAND)
    {
        Value* lo = c32(b, 0);
        if(self->lo_off != nullptr)
        {
            Value* sum = rocke_b_add(b, q, self->lo_off);
            lo = rocke_b_smax(b, sum, lo);
        }
        Value* hi = band->sk_minus1;
        if(self->hi_off != nullptr)
        {
            Value* sum = rocke_b_add(b, q, self->hi_off);
            hi = rocke_b_smin(b, sum, hi);
        }
        *out_lo = lo;
        *out_hi = hi;
        return;
    }
    if(band->left.has || band->right.has)
    {
        ckc::raise_status(ROCKE_ERR_NOTIMPL, "a banded policy defines its own row limits");
    }
    *out_lo = c32(b, 0);
    *out_hi = band->sk_minus1;
}

// RuntimeBand._side_offset: off + sign * bound (sign * FAR when unbounded).
Value* side_offset(Builder* b, const rocke_band_limit_t& bound, Value* off, int sign, Value* zero)
{
    if(!bound.has)
    {
        return nullptr;
    }
    Value* far = c32(b, (int64_t)sign * ROCKE_ATTN_I32_FAR);
    if(!bound.bound.is_value)
    {
        Value* v = c32(b, (int64_t)sign * bound.bound.i);
        return off == nullptr ? v : rocke_b_add(b, off, v);
    }
    Value* v = bound.bound.v;
    if(sign <= 0)
    {
        v = rocke_b_sub(b, zero, bound.bound.v);
    }
    if(off != nullptr)
    {
        v = rocke_b_add(b, off, v);
    }
    Value* neg = rocke_b_cmp_lt(b, bound.bound.v, zero);
    return rocke_b_select(b, neg, far, v);
}

// MaskPolicy.__init__ over an already-built BandEmitter.
void policy_init(rocke_attn_bwd_mask_policy_t* self,
                 rocke_attn_bwd_mask_kind_t kind,
                 Builder* b,
                 const rocke_attn_runtime_bounds_t* bounds,
                 int k_m0,
                 int k_n0)
{
    *self = {};
    self->kind = kind;
    rocke_status_t st = rocke_band_emitter_init(&self->band, b, bounds);
    if(st != ROCKE_OK)
    {
        ckc::raise_status(st, rocke_ir_builder_error(b));
    }
    require_positive_tiles(k_m0, k_n0);
    self->k_m0 = k_m0;
    self->k_n0 = k_n0;
    self->masked = kind == ROCKE_ATTN_BWD_MASK_RUNTIME_BAND;
}

} // namespace

rocke_status_t rocke_attn_bwd_band_q_tile_range(Builder* b,
                                                rocke_band_emitter_t* band,
                                                Value* k0,
                                                int k_n0,
                                                int k_m0,
                                                bool uniform,
                                                Value** out_qt_start,
                                                Value** out_n_qt)
{
    return ckc::guard_status(b, [&]() {
        return q_tile_range_impl(b, band, k0, k_n0, k_m0, uniform, out_qt_start, out_n_qt);
    });
}

Value* rocke_attn_bwd_band_tile_inside_band(Builder* b,
                                            rocke_band_emitter_t* band,
                                            Value* q_first,
                                            Value* q_last,
                                            Value* k_first,
                                            Value* k_last)
{
    return ckc::guard_builder(
        b, [&]() { return tile_inside_band_impl(b, band, q_first, q_last, k_first, k_last); });
}

Value* rocke_attn_bwd_band_is_edge_tile(
    Builder* b, rocke_band_emitter_t* band, Value* qt, Value* k0, int k_m0, int k_n0, bool masked)
{
    return ckc::guard_builder(
        b, [&]() { return is_edge_tile_impl(b, band, qt, k0, k_m0, k_n0, masked); });
}

Value* rocke_attn_bwd_band_tile_class(Builder* b,
                                      rocke_band_emitter_t* band,
                                      Value* qt,
                                      Value* k0,
                                      int k_m0,
                                      int k_n0,
                                      bool masked,
                                      bool uniform)
{
    return ckc::guard_builder(
        b, [&]() { return tile_class_impl(b, band, qt, k0, k_m0, k_n0, masked, uniform); });
}

rocke_status_t rocke_attn_bwd_band_interior_q_range(Builder* b,
                                                    rocke_band_emitter_t* band,
                                                    Value* k0,
                                                    Value* qt_start,
                                                    Value* n_qt,
                                                    int k_m0,
                                                    int k_n0,
                                                    bool masked,
                                                    bool uniform,
                                                    Value** out_i0,
                                                    Value** out_i1)
{
    return ckc::guard_status(b, [&]() {
        return interior_q_range_impl(
            b, band, k0, qt_start, n_qt, k_m0, k_n0, masked, uniform, out_i0, out_i1);
    });
}

rocke_status_t rocke_attn_bwd_band_dkv_store_rule(Builder* b,
                                                  Value* k,
                                                  Value* len_kv,
                                                  rocke_attn_bwd_seq_mode_t seq_mode,
                                                  const rocke_int_or_value_t* s_kv_max,
                                                  Value** out_store,
                                                  Value** out_live)
{
    return ckc::guard_status(b, [&]() -> rocke_status_t {
        Value* live = rocke_b_cmp_lt(b, k, len_kv);
        if(seq_mode == ROCKE_ATTN_BWD_SEQ_BATCHED)
        {
            if(s_kv_max == nullptr)
            {
                ckc::raise_status(ROCKE_ERR_VALUE, "batched dK/dV stores need s_kv_max");
            }
            Value* limit = as_i32(b, *s_kv_max);
            *out_store = rocke_b_cmp_lt(b, k, limit);
            *out_live = live;
            return ROCKE_OK;
        }
        if(seq_mode == ROCKE_ATTN_BWD_SEQ_THD)
        {
            *out_store = live;
            *out_live = live;
            return ROCKE_OK;
        }
        char msg[ROCKE_ERR_MSG_CAP];
        snprintf(msg,
                 sizeof(msg),
                 "seq_mode must be 'batched' or 'thd' (got enum value %d)",
                 (int)seq_mode);
        ckc::raise_status(ROCKE_ERR_VALUE, msg);
    });
}

rocke_status_t rocke_attn_bwd_runtime_band_init(rocke_attn_bwd_mask_policy_t* self,
                                                Builder* b,
                                                const rocke_attn_runtime_bounds_t* bounds,
                                                int k_m0,
                                                int k_n0)
{
    return ckc::guard_status(b, [&]() -> rocke_status_t {
        policy_init(self, ROCKE_ATTN_BWD_MASK_RUNTIME_BAND, b, bounds, k_m0, k_n0);
        Value* zero = c32(b, 0);
        self->hi_off = side_offset(b, self->band.right, self->band.off, +1, zero);
        self->lo_off = side_offset(b, self->band.left, self->band.off, -1, zero);
        return ROCKE_OK;
    });
}

rocke_status_t rocke_attn_bwd_no_mask_ktail_init(rocke_attn_bwd_mask_policy_t* self,
                                                 Builder* b,
                                                 rocke_int_or_value_t len_q,
                                                 rocke_int_or_value_t len_kv,
                                                 int k_m0,
                                                 int k_n0)
{
    return ckc::guard_status(b, [&]() -> rocke_status_t {
        rocke_attn_runtime_bounds_t bounds = rocke_attn_runtime_bounds_make(len_q, len_kv);
        policy_init(self, ROCKE_ATTN_BWD_MASK_NO_MASK_KTAIL, b, &bounds, k_m0, k_n0);
        return ROCKE_OK;
    });
}

rocke_status_t rocke_attn_bwd_mask_q_tile_range(rocke_attn_bwd_mask_policy_t* self,
                                                Builder* b,
                                                Value* k0,
                                                Value** out_qt_start,
                                                Value** out_n_qt)
{
    return ckc::guard_status(b, [&]() {
        return q_tile_range_impl(
            b, &self->band, k0, self->k_n0, self->k_m0, true, out_qt_start, out_n_qt);
    });
}

Value* rocke_attn_bwd_mask_tile_class(rocke_attn_bwd_mask_policy_t* self,
                                      Builder* b,
                                      Value* qt,
                                      Value* k0)
{
    return ckc::guard_builder(b, [&]() {
        return tile_class_impl(b, &self->band, qt, k0, self->k_m0, self->k_n0, self->masked, true);
    });
}

rocke_status_t rocke_attn_bwd_mask_interior_q_range(rocke_attn_bwd_mask_policy_t* self,
                                                    Builder* b,
                                                    Value* k0,
                                                    Value* qt_start,
                                                    Value* n_qt,
                                                    Value** out_i0,
                                                    Value** out_i1)
{
    return ckc::guard_status(b, [&]() {
        return interior_q_range_impl(b,
                                     &self->band,
                                     k0,
                                     qt_start,
                                     n_qt,
                                     self->k_m0,
                                     self->k_n0,
                                     self->masked,
                                     true,
                                     out_i0,
                                     out_i1);
    });
}

rocke_status_t rocke_attn_bwd_mask_row(rocke_attn_bwd_mask_policy_t* self,
                                       Builder* b,
                                       Value* q,
                                       rocke_row_band_t* out)
{
    (void)b;
    return rocke_band_row_band(&self->band, q, out);
}

Value* rocke_attn_bwd_mask_keep(rocke_attn_bwd_mask_policy_t* self,
                                Builder* b,
                                const rocke_row_band_t* row,
                                Value* k)
{
    return ckc::guard_builder(b, [&]() -> Value* {
        Value* col = rocke_band_col_in(&self->band, k);
        Value* cell = rocke_band_cell_keep(&self->band, row, k, col);
        if(cell == nullptr)
        {
            return nullptr;
        }
        return rocke_b_land(b, row->row_in, cell);
    });
}

Value* rocke_attn_bwd_mask_cell_keep(rocke_attn_bwd_mask_policy_t* self,
                                     Builder* b,
                                     Value* q,
                                     Value* k)
{
    return ckc::guard_builder(b, [&]() -> Value* {
        rocke_band_emitter_t* band = &self->band;
        if(self->kind == ROCKE_ATTN_BWD_MASK_RUNTIME_BAND)
        {
            // The two-offset form (q >= 0).
            Value* q_in = rocke_b_cmp_lt(b, q, band->sq);
            Value* k_in = rocke_b_cmp_lt(b, k, band->sk);
            Value* keep = rocke_b_land(b, q_in, k_in);
            if(self->hi_off != nullptr)
            {
                Value* top = rocke_b_add(b, q, self->hi_off);
                Value* le = rocke_b_cmp_le(b, k, top);
                keep = rocke_b_land(b, keep, le);
            }
            if(self->lo_off != nullptr)
            {
                Value* bot = rocke_b_add(b, q, self->lo_off);
                Value* ge = rocke_b_cmp_ge(b, k, bot);
                keep = rocke_b_land(b, keep, ge);
            }
            return keep;
        }
        rocke_row_band_t row;
        if(rocke_band_row_band(band, q, &row) != ROCKE_OK)
        {
            return nullptr;
        }
        Value* col = rocke_band_col_in(band, k);
        Value* cell = rocke_band_cell_keep(band, &row, k, col);
        if(cell == nullptr)
        {
            return nullptr;
        }
        return rocke_b_land(b, row.row_in, cell);
    });
}

rocke_status_t rocke_attn_bwd_mask_row_limits(rocke_attn_bwd_mask_policy_t* self,
                                              Builder* b,
                                              Value* q,
                                              Value* ok,
                                              Value** out_lo,
                                              Value** out_hi)
{
    return ckc::guard_status(b, [&]() -> rocke_status_t {
        Value* lo = nullptr;
        Value* hi = nullptr;
        row_lo_hi(self, b, q, &lo, &hi);
        Value* q_in = rocke_b_cmp_lt(b, q, self->band.sq);
        Value* nonempty = rocke_b_cmp_le(b, lo, hi);
        Value* valid = rocke_b_land(b, q_in, nonempty);
        if(ok != nullptr)
        {
            valid = rocke_b_land(b, valid, ok);
        }
        Value* none = c32(b, -1);
        *out_lo = lo;
        *out_hi = rocke_b_select(b, valid, hi, none);
        return ROCKE_OK;
    });
}

Value* rocke_attn_bwd_mask_keep_in(Builder* b, Value* lo, Value* hi, Value* k)
{
    return ckc::guard_builder(b, [&]() -> Value* {
        Value* above = rocke_b_smax(b, k, lo);
        Value* clamped = rocke_b_smin(b, above, hi);
        return rocke_b_cmp_eq(b, clamped, k);
    });
}
