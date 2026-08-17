// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of rocke/helpers/preshuffle.py: PreshuffleBSpec,
 * emit_preshuffleb_offset, host_preshuffle_layout.
 *
 * See the header for the original Python and the contract. The single
 * builder-emitting symbol (emit_preshuffleb_offset) reproduces the const_i32 /
 * mul / add call sequence byte-faithfully so the downstream IR is identical to
 * the Python.
 */
#include "rocke/helper_rocke.helpers.preshuffle.h"

#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* ------------------------------------------------------------------ *
 * PreshuffleBSpec.tile_bytes
 * ------------------------------------------------------------------ */

int rocke_preshuffleb_spec_tile_bytes(const rocke_preshuffleb_spec_t* spec)
{
    /* Python: block_n * block_k * elem_bytes.
     * The fields are small Python ints whose product fits in 32 bits for every
     * real tile shape, so a plain int multiply reproduces the value exactly. */
    if(spec == NULL)
    {
        return 0;
    }
    return spec->block_n * spec->block_k * spec->elem_bytes;
}

/* ------------------------------------------------------------------ *
 * emit_preshuffleb_offset
 * ------------------------------------------------------------------ */

rocke_value_t* rocke_emit_preshuffleb_offset(rocke_ir_builder_t* b,
                                             const rocke_preshuffleb_spec_t* spec,
                                             rocke_value_t* n_tile,
                                             rocke_value_t* k_tile,
                                             rocke_value_t* n_in_tile,
                                             rocke_value_t* k_in_tile,
                                             rocke_value_t* n_tile_count)
{
    rocke_value_t* c_tile_bytes;
    rocke_value_t* c_block_k;
    rocke_value_t* c_elem_bytes;
    rocke_value_t* tile_id;
    rocke_value_t* tile_base;
    rocke_value_t* inner;
    rocke_value_t* inner_bytes;

    if(b != NULL && b->status != ROCKE_OK)
    {
        return NULL; /* already in error: no-op */
    }
    if(spec == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "emit_preshuffleb_offset: NULL spec");
        return NULL;
    }

    /* Python, in this exact order:
     *   c_tile_bytes = b.const_i32(spec.tile_bytes)
     *   c_block_k    = b.const_i32(spec.block_k)
     *   c_elem_bytes = b.const_i32(spec.elem_bytes)
     *   tile_id     = b.add(b.mul(k_tile, n_tile_count), n_tile)
     *   tile_base   = b.mul(tile_id, c_tile_bytes)
     *   inner       = b.add(b.mul(n_in_tile, c_block_k), k_in_tile)
     *   inner_bytes = b.mul(inner, c_elem_bytes)
     *   return b.add(tile_base, inner_bytes)
     */
    c_tile_bytes = rocke_b_const_i32(b, rocke_preshuffleb_spec_tile_bytes(spec));
    c_block_k = rocke_b_const_i32(b, spec->block_k);
    c_elem_bytes = rocke_b_const_i32(b, spec->elem_bytes);

    tile_id = rocke_b_add(b, rocke_b_mul(b, k_tile, n_tile_count), n_tile);
    tile_base = rocke_b_mul(b, tile_id, c_tile_bytes);
    inner = rocke_b_add(b, rocke_b_mul(b, n_in_tile, c_block_k), k_in_tile);
    inner_bytes = rocke_b_mul(b, inner, c_elem_bytes);
    return rocke_b_add(b, tile_base, inner_bytes);
}

/* ------------------------------------------------------------------ *
 * host_preshuffle_layout
 * ------------------------------------------------------------------ */

rocke_status_t rocke_host_preshuffle_layout(
    const rocke_preshuffleb_spec_t* spec, int n, int k, int out_shape[4], int out_strides[4])
{
    int n_tiles;
    int k_tiles;

    if(spec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    /* Python: floor-division on positive ints == ceil((n)/block) given the
     * +block-1 numerator. The real specs always have block_n / block_k > 0. */
    n_tiles = (n + spec->block_n - 1) / spec->block_n;
    k_tiles = (k + spec->block_k - 1) / spec->block_k;

    /* Python:
     *   if n_tiles*block_n != n or k_tiles*block_k != k: raise ValueError(...)
     */
    if(n_tiles * spec->block_n != n || k_tiles * spec->block_k != k)
    {
        return ROCKE_ERR_VALUE;
    }

    /* shape   = (k_tiles, n_tiles, block_n, block_k) */
    if(out_shape != NULL)
    {
        out_shape[0] = k_tiles;
        out_shape[1] = n_tiles;
        out_shape[2] = spec->block_n;
        out_shape[3] = spec->block_k;
    }
    /* strides = (n_tiles*block_n*block_k, block_n*block_k, block_k, 1) */
    if(out_strides != NULL)
    {
        out_strides[0] = n_tiles * spec->block_n * spec->block_k;
        out_strides[1] = spec->block_n * spec->block_k;
        out_strides[2] = spec->block_k;
        out_strides[3] = 1;
    }
    return ROCKE_OK;
}

rocke_status_t rocke_b_host_preshuffle_layout(rocke_ir_builder_t* b,
                                              const rocke_preshuffleb_spec_t* spec,
                                              int n,
                                              int k,
                                              int out_shape[4],
                                              int out_strides[4])
{
    rocke_status_t st;

    if(b != NULL && b->status != ROCKE_OK)
    {
        return b->status; /* already in error: no-op */
    }
    if(spec == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "host_preshuffle_layout: NULL spec");
        return ROCKE_ERR_VALUE;
    }

    st = rocke_host_preshuffle_layout(spec, n, k, out_shape, out_strides);
    if(st != ROCKE_OK)
    {
        /* Python:
         *   raise ValueError(
         *       "preshuffle requires N / K to divide block_n / block_k "
         *       "(got N=..., block_n=..., K=..., block_k=...)") */
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "preshuffle requires N / K to divide block_n / block_k "
                        "(got N=%d, block_n=%d, K=%d, block_k=%d)",
                        n,
                        spec->block_n,
                        k,
                        spec->block_k);
        return st;
    }
    return ROCKE_OK;
}
