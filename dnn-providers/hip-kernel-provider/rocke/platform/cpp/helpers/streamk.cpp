// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.helpers.streamk.c -- C99 port of rocke.helpers.streamk.
 *
 * Ports the four partitioner symbols:
 *   StreamKReductionStrategy, StreamKPartition, compute_streamk_grid_size,
 *   emit_streamk_decode.
 *
 * compute_streamk_grid_size is pure-int; emit_streamk_decode's builder-call
 * sequence is byte-identical to the Python so the emitted IR op stream matches
 * exactly.
 */
#include "rocke/helper_rocke.helpers.streamk.h"

#include <stddef.h> /* NULL */

#include "rocke/ir_internal.h" /* rocke_i_set_err for Python-ValueError parity */

/* ------------------------------------------------------------------------
 * StreamKReductionStrategy enum value strings (CK Tile naming).
 * ------------------------------------------------------------------------ */
const char* rocke_streamk_reduction_strategy_value(rocke_streamk_reduction_strategy_t s)
{
    switch(s)
    {
    case ROCKE_STREAMK_REDUCTION_ATOMIC:
        return "atomic";
    case ROCKE_STREAMK_REDUCTION_REDUCTION:
        return "reduction";
    default:
        return NULL;
    }
}

/* ------------------------------------------------------------------------
 * StreamKPartition properties.
 *
 *   @property num_macro_tiles: m_tiles * n_tiles * k_iters
 *   @property k_iters_per_output_tile: k_iters
 * ------------------------------------------------------------------------ */
int rocke_streamk_partition_num_macro_tiles(const rocke_streamk_partition_t* spec)
{
    return spec->m_tiles * spec->n_tiles * spec->k_iters;
}

int rocke_streamk_partition_k_iters_per_output_tile(const rocke_streamk_partition_t* spec)
{
    return spec->k_iters;
}

/* Module-level streamk_num_macro_tiles(spec): plain Python view. */
int rocke_streamk_num_macro_tiles(const rocke_streamk_partition_t* spec)
{
    return rocke_streamk_partition_num_macro_tiles(spec);
}

/* ------------------------------------------------------------------------
 * compute_streamk_grid_size
 *
 *   if spec.num_macro_tiles <= 0:
 *       raise ValueError("spec has zero macro tiles")
 *   return min(spec.num_macro_tiles, num_cus * blocks_per_cu)
 * ------------------------------------------------------------------------ */
int rocke_compute_streamk_grid_size(const rocke_streamk_partition_t* spec,
                                    int num_cus,
                                    int blocks_per_cu,
                                    rocke_status_t* out_status)
{
    int num_macro_tiles;
    int cap;

    num_macro_tiles = rocke_streamk_partition_num_macro_tiles(spec);
    if(num_macro_tiles <= 0)
    {
        if(out_status != NULL)
        {
            *out_status = ROCKE_ERR_VALUE;
        }
        return -1; /* Python: raise ValueError("spec has zero macro tiles") */
    }

    cap = num_cus * blocks_per_cu;
    if(out_status != NULL)
    {
        *out_status = ROCKE_OK;
    }
    return (num_macro_tiles < cap) ? num_macro_tiles : cap;
}

/* ------------------------------------------------------------------------
 * emit_streamk_decode
 *
 *   c_k_iters = b.const_i32(spec.k_iters)
 *   c_n_tiles = b.const_i32(spec.n_tiles)
 *   k_iter    = b.mod(linear_id, c_k_iters)
 *   nn        = b.div(linear_id, c_k_iters)
 *   n_tile    = b.mod(nn, c_n_tiles)
 *   m_tile    = b.div(nn, c_n_tiles)
 *   is_first  = b.cmp_eq(k_iter, b.const_i32(0))
 *   is_last   = b.cmp_eq(k_iter, b.const_i32(spec.k_iters - 1))
 *   return (m_tile, n_tile, k_iter, is_first, is_last)
 *
 * The Python evaluates b.const_i32(0) and b.const_i32(spec.k_iters - 1) as
 * arguments inside the cmp_eq calls; C's argument evaluation order is
 * unspecified, so pin the const-then-cmp order with explicit temporaries.
 * ------------------------------------------------------------------------ */
rocke_streamk_decoded_tile_t rocke_emit_streamk_decode(rocke_ir_builder_t* b,
                                                       rocke_value_t* linear_id,
                                                       const rocke_streamk_partition_t* spec)
{
    rocke_streamk_decoded_tile_t res;
    rocke_value_t* c_k_iters;
    rocke_value_t* c_n_tiles;
    rocke_value_t* nn;
    rocke_value_t* c_zero;
    rocke_value_t* c_last;

    res.m_tile = NULL;
    res.n_tile = NULL;
    res.k_iter = NULL;
    res.is_first = NULL;
    res.is_last = NULL;

    c_k_iters = rocke_b_const_i32(b, (int64_t)spec->k_iters);
    c_n_tiles = rocke_b_const_i32(b, (int64_t)spec->n_tiles);

    res.k_iter = rocke_b_mod(b, linear_id, c_k_iters);
    nn = rocke_b_div(b, linear_id, c_k_iters);
    res.n_tile = rocke_b_mod(b, nn, c_n_tiles);
    res.m_tile = rocke_b_div(b, nn, c_n_tiles);

    c_zero = rocke_b_const_i32(b, 0);
    res.is_first = rocke_b_cmp_eq(b, res.k_iter, c_zero);

    c_last = rocke_b_const_i32(b, (int64_t)(spec->k_iters - 1));
    res.is_last = rocke_b_cmp_eq(b, res.k_iter, c_last);

    return res;
}

/* ------------------------------------------------------------------------
 * Iteration-balanced stream-K (Python StreamKIterPartition and friends).
 * ------------------------------------------------------------------------ */
static int sk_ceil_div(int a, int c)
{
    return (a + c - 1) / c;
}

rocke_streamk_iter_partition_t rocke_streamk_iter_partition_make(int m_tiles,
                                                                 int n_tiles,
                                                                 int iters_per_tile,
                                                                 int max_active_wgs,
                                                                 bool persistent,
                                                                 rocke_status_t* out_status)
{
    rocke_streamk_iter_partition_t p;
    p.m_tiles = 0;
    p.n_tiles = 0;
    p.iters_per_tile = 0;
    p.max_active_wgs = 0;
    p.persistent = false;
    if(m_tiles <= 0 || n_tiles <= 0 || iters_per_tile <= 0 || max_active_wgs <= 0)
    {
        if(out_status)
            *out_status = ROCKE_ERR_VALUE;
        return p;
    }
    p.m_tiles = m_tiles;
    p.n_tiles = n_tiles;
    p.iters_per_tile = iters_per_tile;
    p.max_active_wgs = max_active_wgs;
    p.persistent = persistent;
    if(out_status)
        *out_status = ROCKE_OK;
    return p;
}

rocke_streamk_iter_plan_t rocke_streamk_iter_plan(const rocke_streamk_iter_partition_t* p)
{
    rocke_streamk_iter_plan_t r;
    int rem;
    int contributors;
    int rest;
    r.num_tiles = p->m_tiles * p->n_tiles;
    /* sk_tiles: CK "DP + 2-tile SK", reverting to all-DP when the remainder
     * cannot give every stream-K CTA one iteration. */
    rem = p->max_active_wgs > 0 ? r.num_tiles % p->max_active_wgs : 0;
    if(rem == 0)
        r.sk_tiles = 0;
    else
    {
        r.sk_tiles = r.num_tiles > p->max_active_wgs ? p->max_active_wgs + rem : r.num_tiles;
        if(r.sk_tiles * p->iters_per_tile < p->max_active_wgs)
            r.sk_tiles = 0;
    }
    r.sk_ctas = r.sk_tiles ? p->max_active_wgs : 0;
    r.total_sk_iters = r.sk_tiles * p->iters_per_tile;
    /* Clamped to 1 for an all-DP plan: kernels divide by it unconditionally. */
    r.iters_per_sk_cta = r.sk_ctas ? r.total_sk_iters / r.sk_ctas : 1;
    r.extra_iters = r.sk_ctas ? r.total_sk_iters % r.sk_ctas : 0;
    r.dp_tiles = r.num_tiles - r.sk_tiles;
    r.total_dp_iters = r.dp_tiles * p->iters_per_tile;
    r.grid_size = p->persistent ? p->max_active_wgs : r.dp_tiles + r.sk_ctas;
    if(!r.sk_ctas)
        r.max_linear_partners = 0;
    else
        r.max_linear_partners = sk_ceil_div(p->iters_per_tile - 1, r.iters_per_sk_cta);
    /* tree_rounds = (contributors - 1).bit_length() */
    contributors = r.max_linear_partners + 1;
    r.tree_rounds = 0;
    for(rest = contributors - 1; rest > 0; rest >>= 1)
        ++r.tree_rounds;
    return r;
}

/* Python StreamKIterPartition.flags_bytes(n_waves): one i32 flag per
 * (stream-K CTA, wave), padded to 256 bytes. */
size_t rocke_streamk_flags_bytes(const rocke_streamk_iter_plan_t* r,
                                 int n_waves,
                                 rocke_status_t* out_status)
{
    if(r == NULL || n_waves <= 0)
    {
        if(out_status)
            *out_status = ROCKE_ERR_VALUE;
        return 0;
    }
    if(out_status)
        *out_status = ROCKE_OK;
    const size_t flags = (size_t)4 * (size_t)r->sk_ctas * (size_t)n_waves;
    return (flags + 255) / 256 * 256;
}

int rocke_streamk_start_iter(const rocke_streamk_iter_partition_t* p, int sk_cta)
{
    const rocke_streamk_iter_plan_t r = rocke_streamk_iter_plan(p);
    return r.total_dp_iters + sk_cta * r.iters_per_sk_cta
           + (sk_cta < r.extra_iters ? sk_cta : r.extra_iters);
}

int rocke_streamk_end_iter(const rocke_streamk_iter_partition_t* p, int sk_cta)
{
    const rocke_streamk_iter_plan_t r = rocke_streamk_iter_plan(p);
    return rocke_streamk_start_iter(p, sk_cta) + r.iters_per_sk_cta
           + (sk_cta < r.extra_iters ? 1 : 0);
}

/* Python emit_streamk_sk_start_iter:
 *   c_dp_iters = b.const_i32(part.total_dp_iters)
 *   c_per_cta  = b.const_i32(part.iters_per_sk_cta)
 *   c_extra    = b.const_i32(part.extra_iters)
 *   body = b.mul(sk_cta, c_per_cta)
 *   lead = b.smin(sk_cta, c_extra)
 *   return b.add(c_dp_iters, b.add(body, lead))
 */
rocke_value_t* rocke_emit_streamk_sk_start_iter(rocke_ir_builder_t* b,
                                                rocke_value_t* sk_cta,
                                                const rocke_streamk_iter_args_t* a)
{
    rocke_value_t* body = rocke_b_mul(b, sk_cta, a->iters_per_sk_cta);
    rocke_value_t* lead = rocke_b_smin(b, sk_cta, a->extra_iters);
    rocke_value_t* inner = rocke_b_add(b, body, lead);
    return rocke_b_add(b, a->total_dp_iters, inner);
}

/* Python emit_streamk_iter_range; every nested builder call is pinned to a
 * temporary in the Python evaluation order. */
rocke_streamk_iter_range_t rocke_emit_streamk_iter_range(rocke_ir_builder_t* b,
                                                         rocke_value_t* cta,
                                                         const rocke_streamk_iter_args_t* a,
                                                         bool dp)
{
    rocke_streamk_iter_range_t res;
    rocke_value_t* sk_cta;
    rocke_value_t* sk_start;
    rocke_value_t* c_one;
    rocke_value_t* c_zero;
    rocke_value_t* has_extra;
    rocke_value_t* extra;
    rocke_value_t* sk_len;
    rocke_value_t* start;
    rocke_value_t* length;
    if(dp)
        sk_cta = rocke_b_sub(b, cta, a->dp_tiles);
    else
        sk_cta = cta;
    sk_start = rocke_emit_streamk_sk_start_iter(b, sk_cta, a);
    c_one = rocke_b_const_i32(b, 1);
    c_zero = rocke_b_const_i32(b, 0);
    has_extra = rocke_b_cmp_lt(b, sk_cta, a->extra_iters);
    extra = rocke_b_select(b, has_extra, c_one, c_zero);
    sk_len = rocke_b_add(b, a->iters_per_sk_cta, extra);
    if(dp)
    {
        rocke_value_t* is_dp = rocke_b_cmp_lt(b, cta, a->dp_tiles);
        rocke_value_t* dp_start = rocke_b_mul(b, cta, a->iters_per_tile);
        start = rocke_b_select(b, is_dp, dp_start, sk_start);
        length = rocke_b_select(b, is_dp, a->iters_per_tile, sk_len);
    }
    else
    {
        start = sk_start;
        length = sk_len;
    }
    res.start = rocke_b_to_sgpr_u32(b, start);
    res.end = rocke_b_to_sgpr_u32(b, rocke_b_add(b, res.start, length));
    res.sk_cta = sk_cta;
    return res;
}
