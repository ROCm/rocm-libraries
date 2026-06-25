// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_ck_dsl.helpers.streamk.c -- C99 port of ck_dsl.helpers.streamk.
 *
 * Ports the four partitioner symbols:
 *   StreamKReductionStrategy, StreamKPartition, compute_streamk_grid_size,
 *   emit_streamk_decode.
 *
 * compute_streamk_grid_size is pure-int; emit_streamk_decode's builder-call
 * sequence is byte-identical to the Python so the emitted IR op stream matches
 * exactly.
 */
#include "ckc/helper_ck_dsl.helpers.streamk.h"

#include <stddef.h> /* NULL */

#include "ckc/ir_internal.h" /* ckc_i_set_err for Python-ValueError parity */

/* ------------------------------------------------------------------------
 * StreamKReductionStrategy enum value strings (CK Tile naming).
 * ------------------------------------------------------------------------ */
const char* ckc_streamk_reduction_strategy_value(ckc_streamk_reduction_strategy_t s)
{
    switch(s)
    {
    case CKC_STREAMK_REDUCTION_ATOMIC:
        return "atomic";
    case CKC_STREAMK_REDUCTION_REDUCTION:
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
int ckc_streamk_partition_num_macro_tiles(const ckc_streamk_partition_t* spec)
{
    return spec->m_tiles * spec->n_tiles * spec->k_iters;
}

int ckc_streamk_partition_k_iters_per_output_tile(const ckc_streamk_partition_t* spec)
{
    return spec->k_iters;
}

/* Module-level streamk_num_macro_tiles(spec): plain Python view. */
int ckc_streamk_num_macro_tiles(const ckc_streamk_partition_t* spec)
{
    return ckc_streamk_partition_num_macro_tiles(spec);
}

/* ------------------------------------------------------------------------
 * compute_streamk_grid_size
 *
 *   if spec.num_macro_tiles <= 0:
 *       raise ValueError("spec has zero macro tiles")
 *   return min(spec.num_macro_tiles, num_cus * blocks_per_cu)
 * ------------------------------------------------------------------------ */
int ckc_compute_streamk_grid_size(const ckc_streamk_partition_t* spec,
                                  int num_cus,
                                  int blocks_per_cu,
                                  ckc_status_t* out_status)
{
    int num_macro_tiles;
    int cap;

    num_macro_tiles = ckc_streamk_partition_num_macro_tiles(spec);
    if(num_macro_tiles <= 0)
    {
        if(out_status != NULL)
        {
            *out_status = CKC_ERR_VALUE;
        }
        return -1; /* Python: raise ValueError("spec has zero macro tiles") */
    }

    cap = num_cus * blocks_per_cu;
    if(out_status != NULL)
    {
        *out_status = CKC_OK;
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
ckc_streamk_decoded_tile_t ckc_emit_streamk_decode(ckc_ir_builder_t* b,
                                                   ckc_value_t* linear_id,
                                                   const ckc_streamk_partition_t* spec)
{
    ckc_streamk_decoded_tile_t res;
    ckc_value_t* c_k_iters;
    ckc_value_t* c_n_tiles;
    ckc_value_t* nn;
    ckc_value_t* c_zero;
    ckc_value_t* c_last;

    res.m_tile = NULL;
    res.n_tile = NULL;
    res.k_iter = NULL;
    res.is_first = NULL;
    res.is_last = NULL;

    c_k_iters = ckc_b_const_i32(b, (int64_t)spec->k_iters);
    c_n_tiles = ckc_b_const_i32(b, (int64_t)spec->n_tiles);

    res.k_iter = ckc_b_mod(b, linear_id, c_k_iters);
    nn = ckc_b_div(b, linear_id, c_k_iters);
    res.n_tile = ckc_b_mod(b, nn, c_n_tiles);
    res.m_tile = ckc_b_div(b, nn, c_n_tiles);

    c_zero = ckc_b_const_i32(b, 0);
    res.is_first = ckc_b_cmp_eq(b, res.k_iter, c_zero);

    c_last = ckc_b_const_i32(b, (int64_t)(spec->k_iters - 1));
    res.is_last = ckc_b_cmp_eq(b, res.k_iter, c_last);

    return res;
}
