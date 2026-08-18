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
