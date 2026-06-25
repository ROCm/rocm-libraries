// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.helpers.grid.c -- C99 port of rocke.helpers.grid.
 *
 * Ports chiplet_aware_super_tile_dynamic and the two dynamic helpers it
 * composes. The builder-call sequence is byte-identical to the Python so the
 * emitted IR op stream matches exactly.
 */
#include "rocke/helper_rocke.helpers.grid.h"

#include "rocke/ir_internal.h" /* rocke_i_set_err for Python-ValueError parity */

/* ------------------------------------------------------------------------
 * chiplet_transform_chunked_dynamic (file-local; Python helper)
 *
 *   block = num_xcds * chunk_size
 *   c_num_xcds   = b.const_i32(int(num_xcds))
 *   c_chunk_size = b.const_i32(int(chunk_size))
 *   c_block      = b.const_i32(int(block))
 *   limit        = b.mul(b.div(num_wgs, c_block), c_block)
 *   xcd          = b.mod(wgid, c_num_xcds)
 *   local_pid    = b.div(wgid, c_num_xcds)
 *   chunk_idx    = b.div(local_pid, c_chunk_size)
 *   pos_in_chunk = b.mod(local_pid, c_chunk_size)
 *   new_wgid     = b.add(b.add(b.mul(chunk_idx, c_block),
 *                              b.mul(xcd, c_chunk_size)),
 *                        pos_in_chunk)
 *   in_full_block = b.cmp_lt(wgid, limit)
 *   return b.select(in_full_block, new_wgid, wgid)
 * ------------------------------------------------------------------------ */
static rocke_value_t* rocke_i_chiplet_transform_chunked_dynamic(rocke_ir_builder_t* b,
                                                                rocke_value_t* wgid,
                                                                rocke_value_t* num_wgs,
                                                                int num_xcds,
                                                                int chunk_size)
{
    int block;
    rocke_value_t* c_num_xcds;
    rocke_value_t* c_chunk_size;
    rocke_value_t* c_block;
    rocke_value_t* limit;
    rocke_value_t* xcd;
    rocke_value_t* local_pid;
    rocke_value_t* chunk_idx;
    rocke_value_t* pos_in_chunk;
    rocke_value_t* new_wgid;
    rocke_value_t* in_full_block;

    if(num_xcds <= 0 || chunk_size <= 0)
    {
        return (rocke_value_t*)rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "chiplet_transform_chunked_dynamic: invalid args num_xcds=%d chunk_size=%d",
            num_xcds,
            chunk_size);
    }

    block = num_xcds * chunk_size;

    c_num_xcds = rocke_b_const_i32(b, (int64_t)num_xcds);
    c_chunk_size = rocke_b_const_i32(b, (int64_t)chunk_size);
    c_block = rocke_b_const_i32(b, (int64_t)block);

    /* limit = (num_wgs / block) * block  (largest full-block boundary) */
    limit = rocke_b_mul(b, rocke_b_div(b, num_wgs, c_block), c_block);

    xcd = rocke_b_mod(b, wgid, c_num_xcds);
    local_pid = rocke_b_div(b, wgid, c_num_xcds);
    chunk_idx = rocke_b_div(b, local_pid, c_chunk_size);
    pos_in_chunk = rocke_b_mod(b, local_pid, c_chunk_size);

    /* Python: b.add(b.add(b.mul(chunk_idx, c_block), b.mul(xcd, c_chunk_size)),
     * pos_in_chunk). Python evaluates the inner add's first arg (mul chunk_idx)
     * before its second (mul xcd), so the SSA order is mul(chunk_idx,block),
     * mul(xcd,chunk_size), add, add. C's argument evaluation order is
     * unspecified -- pin it with explicit temporaries. */
    {
        rocke_value_t* mul_chunk = rocke_b_mul(b, chunk_idx, c_block);
        rocke_value_t* mul_xcd = rocke_b_mul(b, xcd, c_chunk_size);
        new_wgid = rocke_b_add(b, rocke_b_add(b, mul_chunk, mul_xcd), pos_in_chunk);
    }

    in_full_block = rocke_b_cmp_lt(b, wgid, limit);
    return rocke_b_select(b, in_full_block, new_wgid, wgid);
}

/* ------------------------------------------------------------------------
 * super_tile_swizzle_dynamic (file-local; Python helper)
 *
 *   if wgm <= 0: raise ValueError
 *   c_wgm = b.const_i32(int(wgm))
 *   num_wgid_in_group = b.mul(c_wgm, num_pid_n)
 *   group_id    = b.div(wgid, num_wgid_in_group)
 *   first_pid_m = b.mul(group_id, c_wgm)
 *   rem         = b.sub(num_pid_m, first_pid_m)
 *   use_wgm     = b.cmp_lt(c_wgm, rem)
 *   group_size_m = b.select(use_wgm, c_wgm, rem)
 *   local_id    = b.mod(wgid, num_wgid_in_group)
 *   pid_m       = b.add(first_pid_m, b.mod(local_id, group_size_m))
 *   pid_n       = b.div(local_id, group_size_m)
 *   return SuperTileSwizzleResult(row=pid_m, col=pid_n)
 * ------------------------------------------------------------------------ */
static rocke_super_tile_swizzle_result_t
    rocke_i_super_tile_swizzle_dynamic(rocke_ir_builder_t* b,
                                       rocke_value_t* wgid,
                                       rocke_value_t* num_pid_m,
                                       rocke_value_t* num_pid_n,
                                       int wgm)
{
    rocke_super_tile_swizzle_result_t res;
    rocke_value_t* c_wgm;
    rocke_value_t* num_wgid_in_group;
    rocke_value_t* group_id;
    rocke_value_t* first_pid_m;
    rocke_value_t* rem;
    rocke_value_t* use_wgm;
    rocke_value_t* group_size_m;
    rocke_value_t* local_id;

    res.row = NULL;
    res.col = NULL;

    if(wgm <= 0)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "super_tile_swizzle_dynamic: wgm=%d must be > 0", wgm);
        return res;
    }

    c_wgm = rocke_b_const_i32(b, (int64_t)wgm);
    num_wgid_in_group = rocke_b_mul(b, c_wgm, num_pid_n);

    group_id = rocke_b_div(b, wgid, num_wgid_in_group);
    first_pid_m = rocke_b_mul(b, group_id, c_wgm);
    rem = rocke_b_sub(b, num_pid_m, first_pid_m);
    use_wgm = rocke_b_cmp_lt(b, c_wgm, rem);
    group_size_m = rocke_b_select(b, use_wgm, c_wgm, rem);

    local_id = rocke_b_mod(b, wgid, num_wgid_in_group);
    res.row = rocke_b_add(b, first_pid_m, rocke_b_mod(b, local_id, group_size_m));
    res.col = rocke_b_div(b, local_id, group_size_m);

    return res;
}

/* ------------------------------------------------------------------------
 * chiplet_aware_super_tile_dynamic (public)
 *
 *   num_wgs  = b.mul(num_pid_m, num_pid_n)
 *   remapped = chiplet_transform_chunked_dynamic(
 *                  b, wgid, num_wgs=num_wgs,
 *                  num_xcds=num_xcds, chunk_size=chunk_size)
 *   return super_tile_swizzle_dynamic(
 *              b, remapped, num_pid_m=num_pid_m,
 *              num_pid_n=num_pid_n, wgm=wgm)
 * ------------------------------------------------------------------------ */
rocke_super_tile_swizzle_result_t rocke_chiplet_aware_super_tile_dynamic(rocke_ir_builder_t* b,
                                                                         rocke_value_t* wgid,
                                                                         rocke_value_t* num_pid_m,
                                                                         rocke_value_t* num_pid_n,
                                                                         int wgm,
                                                                         int num_xcds,
                                                                         int chunk_size)
{
    rocke_value_t* num_wgs;
    rocke_value_t* remapped;

    num_wgs = rocke_b_mul(b, num_pid_m, num_pid_n);
    remapped = rocke_i_chiplet_transform_chunked_dynamic(b, wgid, num_wgs, num_xcds, chunk_size);
    return rocke_i_super_tile_swizzle_dynamic(b, remapped, num_pid_m, num_pid_n, wgm);
}

/* ------------------------------------------------------------------------
 * chiplet_transform_chunked (file-local; Python COMPILE-TIME helper)
 *
 * Differs from the *_dynamic peer only in that num_wgs (hence `limit`) is a
 * compile-time int, so `c_limit` is a folded const and `in_full_block` compares
 * against it directly (no div/mul IR for the limit). Python (grid.py 84-110):
 *   block = num_xcds * chunk_size
 *   limit = (num_wgs // block) * block
 *   c_num_xcds   = b.const_i32(num_xcds)
 *   c_chunk_size = b.const_i32(chunk_size)
 *   c_block      = b.const_i32(block)
 *   c_limit      = b.const_i32(limit)
 *   xcd          = b.mod(wgid, c_num_xcds)
 *   local_pid    = b.div(wgid, c_num_xcds)
 *   chunk_idx    = b.div(local_pid, c_chunk_size)
 *   pos_in_chunk = b.mod(local_pid, c_chunk_size)
 *   new_wgid     = b.add(b.add(b.mul(chunk_idx, c_block),
 *                              b.mul(xcd, c_chunk_size)),
 *                        pos_in_chunk)
 *   in_full_block = b.cmp_lt(wgid, c_limit)
 *   return b.select(in_full_block, new_wgid, wgid)
 * ------------------------------------------------------------------------ */
static rocke_value_t* rocke_i_chiplet_transform_chunked(
    rocke_ir_builder_t* b, rocke_value_t* wgid, int num_wgs, int num_xcds, int chunk_size)
{
    int block;
    int limit;
    rocke_value_t* c_num_xcds;
    rocke_value_t* c_chunk_size;
    rocke_value_t* c_block;
    rocke_value_t* c_limit;
    rocke_value_t* xcd;
    rocke_value_t* local_pid;
    rocke_value_t* chunk_idx;
    rocke_value_t* pos_in_chunk;
    rocke_value_t* new_wgid;
    rocke_value_t* in_full_block;

    if(num_xcds <= 0 || chunk_size <= 0 || num_wgs <= 0)
    {
        return (rocke_value_t*)rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "chiplet_transform_chunked: invalid args num_wgs=%d num_xcds=%d chunk_size=%d",
            num_wgs,
            num_xcds,
            chunk_size);
    }

    block = num_xcds * chunk_size;
    limit = (num_wgs / block) * block;

    c_num_xcds = rocke_b_const_i32(b, (int64_t)num_xcds);
    c_chunk_size = rocke_b_const_i32(b, (int64_t)chunk_size);
    c_block = rocke_b_const_i32(b, (int64_t)block);
    c_limit = rocke_b_const_i32(b, (int64_t)limit);

    xcd = rocke_b_mod(b, wgid, c_num_xcds);
    local_pid = rocke_b_div(b, wgid, c_num_xcds);
    chunk_idx = rocke_b_div(b, local_pid, c_chunk_size);
    pos_in_chunk = rocke_b_mod(b, local_pid, c_chunk_size);

    {
        rocke_value_t* mul_chunk = rocke_b_mul(b, chunk_idx, c_block);
        rocke_value_t* mul_xcd = rocke_b_mul(b, xcd, c_chunk_size);
        new_wgid = rocke_b_add(b, rocke_b_add(b, mul_chunk, mul_xcd), pos_in_chunk);
    }

    in_full_block = rocke_b_cmp_lt(b, wgid, c_limit);
    return rocke_b_select(b, in_full_block, new_wgid, wgid);
}

/* ------------------------------------------------------------------------
 * super_tile_swizzle (file-local; Python COMPILE-TIME helper)
 *
 * Differs from the *_dynamic peer in that num_pid_m / num_pid_n are
 * compile-time ints: num_wgid_in_group (= wgm * num_pid_n) and c_num_pid_m are
 * folded consts, not IR muls. Python (grid.py 159-181):
 *   num_wgid_in_group = wgm * num_pid_n
 *   c_wgm               = b.const_i32(wgm)
 *   c_num_pid_m         = b.const_i32(num_pid_m)
 *   c_num_wgid_in_group = b.const_i32(num_wgid_in_group)
 *   group_id     = b.div(wgid, c_num_wgid_in_group)
 *   first_pid_m  = b.mul(group_id, c_wgm)
 *   rem          = b.sub(c_num_pid_m, first_pid_m)
 *   use_wgm      = b.cmp_lt(c_wgm, rem)
 *   group_size_m = b.select(use_wgm, c_wgm, rem)
 *   local_id     = b.mod(wgid, c_num_wgid_in_group)
 *   pid_m        = b.add(first_pid_m, b.mod(local_id, group_size_m))
 *   pid_n        = b.div(local_id, group_size_m)
 * ------------------------------------------------------------------------ */
static rocke_super_tile_swizzle_result_t rocke_i_super_tile_swizzle(
    rocke_ir_builder_t* b, rocke_value_t* wgid, int num_pid_m, int num_pid_n, int wgm)
{
    rocke_super_tile_swizzle_result_t res;
    int num_wgid_in_group;
    rocke_value_t* c_wgm;
    rocke_value_t* c_num_pid_m;
    rocke_value_t* c_num_wgid_in_group;
    rocke_value_t* group_id;
    rocke_value_t* first_pid_m;
    rocke_value_t* rem;
    rocke_value_t* use_wgm;
    rocke_value_t* group_size_m;
    rocke_value_t* local_id;

    res.row = NULL;
    res.col = NULL;

    if(num_pid_m <= 0 || num_pid_n <= 0 || wgm <= 0)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "super_tile_swizzle: invalid args num_pid_m=%d num_pid_n=%d wgm=%d",
                        num_pid_m,
                        num_pid_n,
                        wgm);
        return res;
    }

    num_wgid_in_group = wgm * num_pid_n;

    c_wgm = rocke_b_const_i32(b, (int64_t)wgm);
    c_num_pid_m = rocke_b_const_i32(b, (int64_t)num_pid_m);
    c_num_wgid_in_group = rocke_b_const_i32(b, (int64_t)num_wgid_in_group);

    group_id = rocke_b_div(b, wgid, c_num_wgid_in_group);
    first_pid_m = rocke_b_mul(b, group_id, c_wgm);
    rem = rocke_b_sub(b, c_num_pid_m, first_pid_m);
    use_wgm = rocke_b_cmp_lt(b, c_wgm, rem);
    group_size_m = rocke_b_select(b, use_wgm, c_wgm, rem);

    local_id = rocke_b_mod(b, wgid, c_num_wgid_in_group);
    res.row = rocke_b_add(b, first_pid_m, rocke_b_mod(b, local_id, group_size_m));
    res.col = rocke_b_div(b, local_id, group_size_m);

    return res;
}

/* ------------------------------------------------------------------------
 * chiplet_aware_super_tile (public; Python COMPILE-TIME composition)
 *
 *   num_wgs  = int(num_pid_m) * int(num_pid_n)
 *   remapped = chiplet_transform_chunked(b, wgid, num_wgs=num_wgs,
 *                                        num_xcds=num_xcds, chunk_size=chunk_size)
 *   return super_tile_swizzle(b, remapped, num_pid_m=num_pid_m,
 *                             num_pid_n=num_pid_n, wgm=wgm)
 * ------------------------------------------------------------------------ */
rocke_super_tile_swizzle_result_t rocke_chiplet_aware_super_tile(rocke_ir_builder_t* b,
                                                                 rocke_value_t* wgid,
                                                                 int num_pid_m,
                                                                 int num_pid_n,
                                                                 int wgm,
                                                                 int num_xcds,
                                                                 int chunk_size)
{
    int num_wgs = num_pid_m * num_pid_n;
    rocke_value_t* remapped
        = rocke_i_chiplet_transform_chunked(b, wgid, num_wgs, num_xcds, chunk_size);
    return rocke_i_super_tile_swizzle(b, remapped, num_pid_m, num_pid_n, wgm);
}
