// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.helpers.attention_band.cpp -- C port of
 * rocke.helpers.attention_band: AttnRuntimeBounds, RowBand, BandEmitter.
 *
 * Each emitter reproduces the Python builder-call sequence in the same order so
 * the emitted IR (and its SSA numbering) stays byte-identical. Python evaluates
 * call arguments left to right; C++ does not, so every nested builder call is
 * bound to a local in Python's order.
 *
 * Internally a Python raise is a thrown ckc::Error; each public entry catches it
 * at the boundary and records it on the builder.
 */

#include "rocke/helper_rocke.helpers.attention_band.h"

#include <stdio.h>

#include "rocke/error_boundary.hpp"

rocke_int_or_value_t rocke_iov_int(int64_t i)
{
    rocke_int_or_value_t x = {};
    x.is_value = false;
    x.i = i;
    x.v = nullptr;
    return x;
}

rocke_int_or_value_t rocke_iov_value(rocke_value_t* v)
{
    rocke_int_or_value_t x = {};
    x.is_value = true;
    x.i = 0;
    x.v = v;
    return x;
}

rocke_attn_runtime_bounds_t rocke_attn_runtime_bounds_make(rocke_int_or_value_t seqlen_q,
                                                           rocke_int_or_value_t seqlen_k)
{
    rocke_attn_runtime_bounds_t bounds = {};
    bounds.seqlen_q = seqlen_q;
    bounds.seqlen_k = seqlen_k;
    bounds.diag_offset = rocke_iov_int(0);
    bounds.left_bound = rocke_iov_int(-1);
    bounds.right_bound = rocke_iov_int(-1);
    bounds.skip_tiles = true;
    return bounds;
}

/* __post_init__: writes the ValueError text of the first bad field into `msg`
 * (when non-null) and returns false. */
static bool bounds_valid(const rocke_attn_runtime_bounds_t& bounds, char* msg, size_t cap)
{
    const struct
    {
        const char* name;
        const rocke_int_or_value_t* v;
    } fields[2] = {{"left_bound", &bounds.left_bound}, {"right_bound", &bounds.right_bound}};
    for(const auto& f : fields)
    {
        if(!f.v->is_value && f.v->i < -1)
        {
            if(msg != nullptr && cap != 0)
            {
                snprintf(msg, cap, "%s must be >= -1 (got %lld)", f.name, (long long)f.v->i);
            }
            return false;
        }
    }
    if(msg != nullptr && cap != 0)
    {
        msg[0] = '\0';
    }
    return true;
}

int rocke_attn_runtime_bounds_validate(const rocke_attn_runtime_bounds_t* bounds,
                                       char* reason,
                                       size_t reason_cap)
{
    if(bounds == nullptr)
    {
        if(reason != nullptr && reason_cap != 0)
        {
            snprintf(reason, reason_cap, "AttnRuntimeBounds is NULL");
        }
        return 0;
    }
    return bounds_valid(*bounds, reason, reason_cap) ? 1 : 0;
}

static rocke_value_t* i32_of(rocke_ir_builder_t* b, const rocke_int_or_value_t& x)
{
    return x.is_value ? x.v : rocke_b_const_i32(b, x.i);
}

rocke_value_t* rocke_band_i32(rocke_ir_builder_t* b, rocke_int_or_value_t x)
{
    return ckc::guard_builder(b, [&]() -> rocke_value_t* { return i32_of(b, x); });
}

/* _is_static_unbounded: an int below zero. */
static bool static_unbounded(const rocke_int_or_value_t& x)
{
    return !x.is_value && x.i < 0;
}

static rocke_value_t* far_const(rocke_band_emitter_t* self)
{
    if(self->far_value == nullptr)
    {
        self->far_value = rocke_b_const_i32(self->b, ROCKE_ATTN_I32_FAR);
    }
    return self->far_value;
}

rocke_status_t rocke_band_emitter_init(rocke_band_emitter_t* self,
                                       rocke_ir_builder_t* b,
                                       const rocke_attn_runtime_bounds_t* bounds)
{
    return ckc::guard_status(b, [&]() -> rocke_status_t {
        char msg[ROCKE_ERR_MSG_CAP];
        if(!bounds_valid(*bounds, msg, sizeof(msg)))
        {
            ckc::raise_status(ROCKE_ERR_VALUE, msg);
        }
        *self = {};
        self->b = b;
        self->bounds = *bounds;
        self->sq = i32_of(b, bounds->seqlen_q);
        self->sk = i32_of(b, bounds->seqlen_k);
        rocke_value_t* one = rocke_b_const_i32(b, 1);
        self->zero = rocke_b_const_i32(b, 0);
        self->sk_minus1 = rocke_b_sub(b, self->sk, one);
        self->off = nullptr;
        if(!(!bounds->diag_offset.is_value && bounds->diag_offset.i == 0))
        {
            self->off = i32_of(b, bounds->diag_offset);
        }
        self->left.has = !static_unbounded(bounds->left_bound);
        self->left.bound = bounds->left_bound;
        self->right.has = !static_unbounded(bounds->right_bound);
        self->right.bound = bounds->right_bound;
        self->far_value = nullptr;
        return ROCKE_OK;
    });
}

rocke_status_t
    rocke_band_row_band(rocke_band_emitter_t* self, rocke_value_t* q_pos, rocke_row_band_t* out)
{
    rocke_ir_builder_t* b = self->b;
    return ckc::guard_status(b, [&]() -> rocke_status_t {
        rocke_value_t* row_in = rocke_b_cmp_lt(b, q_pos, self->sq);
        rocke_value_t* centre = q_pos;
        if(self->off != nullptr)
        {
            centre = rocke_b_add(b, q_pos, self->off);
        }
        rocke_value_t* lo = nullptr;
        rocke_value_t* hi = nullptr;
        if(self->right.has)
        {
            rocke_value_t* r = i32_of(b, self->right.bound);
            hi = rocke_b_add(b, centre, r);
            if(self->right.bound.is_value)
            {
                rocke_value_t* neg = rocke_b_cmp_lt(b, r, self->zero);
                rocke_value_t* far = far_const(self);
                hi = rocke_b_select(b, neg, far, hi);
            }
        }
        if(self->left.has)
        {
            rocke_value_t* lft = i32_of(b, self->left.bound);
            lo = rocke_b_sub(b, centre, lft);
            if(self->left.bound.is_value)
            {
                rocke_value_t* neg = rocke_b_cmp_lt(b, lft, self->zero);
                rocke_value_t* far = far_const(self);
                rocke_value_t* neg_far = rocke_b_sub(b, self->zero, far);
                lo = rocke_b_select(b, neg, neg_far, lo);
            }
        }
        rocke_value_t* lo_eff = self->zero;
        if(lo != nullptr)
        {
            lo_eff = rocke_b_smax(b, lo, self->zero);
        }
        rocke_value_t* hi_eff = self->sk_minus1;
        if(hi != nullptr)
        {
            hi_eff = rocke_b_smin(b, hi, self->sk_minus1);
        }
        rocke_value_t* nonempty = rocke_b_cmp_le(b, lo_eff, hi_eff);
        rocke_value_t* valid = rocke_b_land(b, row_in, nonempty);
        out->q_pos = q_pos;
        out->row_in = row_in;
        out->valid = valid;
        out->lo = lo;
        out->hi = hi;
        return ROCKE_OK;
    });
}

rocke_value_t* rocke_band_col_in(rocke_band_emitter_t* self, rocke_value_t* k_pos)
{
    return ckc::guard_builder(
        self->b, [&]() -> rocke_value_t* { return rocke_b_cmp_lt(self->b, k_pos, self->sk); });
}

rocke_value_t* rocke_band_cell_keep(rocke_band_emitter_t* self,
                                    const rocke_row_band_t* rb,
                                    rocke_value_t* k_pos,
                                    rocke_value_t* col_in)
{
    rocke_ir_builder_t* b = self->b;
    return ckc::guard_builder(b, [&]() -> rocke_value_t* {
        rocke_value_t* keep = col_in;
        if(rb->hi != nullptr)
        {
            rocke_value_t* le = rocke_b_cmp_le(b, k_pos, rb->hi);
            keep = rocke_b_land(b, keep, le);
        }
        if(rb->lo != nullptr)
        {
            rocke_value_t* ge = rocke_b_cmp_ge(b, k_pos, rb->lo);
            keep = rocke_b_land(b, keep, ge);
        }
        return keep;
    });
}

rocke_status_t rocke_band_k_tile_range(rocke_band_emitter_t* self,
                                       rocke_value_t* q_pos_first,
                                       int block_m,
                                       int block_k,
                                       rocke_value_t** out_start,
                                       rocke_value_t** out_stop)
{
    rocke_ir_builder_t* b = self->b;
    return ckc::guard_status(b, [&]() -> rocke_status_t {
        rocke_value_t* ck = rocke_b_const_i32(b, block_k);
        rocke_value_t* centre0 = q_pos_first;
        if(self->off != nullptr)
        {
            centre0 = rocke_b_add(b, q_pos_first, self->off);
        }
        rocke_value_t* start = nullptr;
        if(!self->left.has)
        {
            start = self->zero;
        }
        else
        {
            rocke_value_t* lft = i32_of(b, self->left.bound);
            rocke_value_t* lo0 = rocke_b_sub(b, centre0, lft);
            if(self->left.bound.is_value)
            {
                rocke_value_t* neg = rocke_b_cmp_lt(b, lft, self->zero);
                lo0 = rocke_b_select(b, neg, self->zero, lo0);
            }
            rocke_value_t* lo_pos = rocke_b_smax(b, lo0, self->zero);
            start = rocke_b_div(b, lo_pos, ck);
        }
        rocke_value_t* hi_last = nullptr;
        if(!self->right.has)
        {
            hi_last = self->sk_minus1;
        }
        else
        {
            rocke_value_t* r = i32_of(b, self->right.bound);
            rocke_value_t* tail = rocke_b_const_i32(b, (int64_t)block_m - 1);
            rocke_value_t* top = rocke_b_add(b, centre0, tail);
            rocke_value_t* hi_raw = rocke_b_add(b, top, r);
            if(self->right.bound.is_value)
            {
                rocke_value_t* neg = rocke_b_cmp_lt(b, r, self->zero);
                rocke_value_t* far = far_const(self);
                hi_raw = rocke_b_select(b, neg, far, hi_raw);
            }
            hi_last = rocke_b_smin(b, hi_raw, self->sk_minus1);
        }
        // (hi + block_k) // block_k == hi // block_k + 1 for hi >= 0 and 0 for an
        // empty range (hi == -1); clamp so very negative values stay empty.
        rocke_value_t* m1 = rocke_b_const_i32(b, -1);
        rocke_value_t* clamped = rocke_b_smax(b, hi_last, m1);
        rocke_value_t* shifted = rocke_b_add(b, clamped, ck);
        rocke_value_t* stop = rocke_b_div(b, shifted, ck);
        *out_start = start;
        *out_stop = stop;
        return ROCKE_OK;
    });
}

rocke_value_t* rocke_band_k_tile_count_ceil(rocke_band_emitter_t* self, int block_k)
{
    rocke_ir_builder_t* b = self->b;
    return ckc::guard_builder(b, [&]() -> rocke_value_t* {
        rocke_value_t* ck = rocke_b_const_i32(b, block_k);
        rocke_value_t* tail = rocke_b_const_i32(b, (int64_t)block_k - 1);
        rocke_value_t* sum = rocke_b_add(b, self->sk, tail);
        return rocke_b_div(b, sum, ck);
    });
}
