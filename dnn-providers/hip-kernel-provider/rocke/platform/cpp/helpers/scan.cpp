// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.helpers.scan.c -- C99 port of selected symbols from
 * rocke/helpers/scan.py.
 *
 * Ported symbols: lds_zero_i32, block_exclusive_scan_i32.
 *
 * Each helper reproduces its Python counterpart's rocke_b_* builder-call sequence
 * byte-faithfully (same ops, same order, same operands). The host-side control
 * structure (the chunk loop, the Hillis-Steele while-loop, the scf_if scoping)
 * is reproduced exactly so the emitted op stream is identical to the Python.
 *
 * scf_if scoping: Python's ``with b.scf_if(cond):`` maps to
 *   rocke_if_t iff = rocke_b_scf_if(b, cond);
 *   rocke_b_region_enter(b, iff.then_region);
 *   ... body ...
 *   rocke_b_region_leave(b);
 *
 * Lifetime: every node is arena-owned (rocke_ir_builder_t.arena). Nothing is freed
 * individually; the arena bulk-frees the whole graph.
 */

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>

#include "rocke/error.hpp"
#include "rocke/helper_rocke.helpers.scan.h"
#include "rocke/ir.h"

/* ----------------------------------------------------------------- helpers */

/* Raise the failure as a ckc::Error (mirroring the Python `raise`); the public
 * entry boundary catches it and records status + message on the builder, so the
 * C ABI is unchanged. [[noreturn]] keeps the existing `rocke_scan_set_err(...);
 * return;` call sites valid -- the trailing return is simply never reached. */
[[noreturn]] static void
    rocke_scan_set_err(rocke_ir_builder_t* b, rocke_status_t st, const char* fmt, ...)
{
    (void)b;
    char msg[ROCKE_ERR_MSG_CAP];
    va_list ap;
    va_start(ap, fmt);
    (void)vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    msg[sizeof(msg) - 1] = '\0';
    ckc::raise_status(st, msg);
}

/* ------------------------------------------------------------- lds_zero_i32 */

void rocke_lds_zero_i32(
    rocke_ir_builder_t* b, rocke_value_t* lds_buf, rocke_value_t* tid, int block_size, int length)
{
    int chunks;
    int c;
    rocke_value_t* c_block;
    rocke_value_t* c_length;
    rocke_value_t* c_zero;

    if(b == NULL)
    {
        return;
    }
    if(length <= 0)
    {
        rocke_scan_set_err(b, ROCKE_ERR_VALUE, "length must be > 0 (got %d)", length);
        return;
    }

    chunks = (length + block_size - 1) / block_size;
    c_block = rocke_b_const_i32(b, block_size);
    c_length = rocke_b_const_i32(b, length);
    c_zero = rocke_b_const_i32(b, 0);

    for(c = 0; c < chunks; ++c)
    {
        /* local = b.add(tid, b.mul(b.const_i32(c), c_block)) */
        rocke_value_t* local
            = rocke_b_add(b, tid, rocke_b_mul(b, rocke_b_const_i32(b, c), c_block));
        /* in_bounds = b.cmp_lt(local, c_length) */
        rocke_value_t* in_bounds = rocke_b_cmp_lt(b, local, c_length);
        /* with b.scf_if(in_bounds): b.smem_store_vN(lds_buf, [local], c_zero, 1) */
        rocke_if_t iff = rocke_b_scf_if(b, in_bounds);
        rocke_b_region_enter(b, iff.then_region);
        {
            rocke_value_t* idx[1];
            idx[0] = local;
            rocke_b_smem_store_vN(b, lds_buf, idx, 1, c_zero, 1);
        }
        rocke_b_region_leave(b);
    }
    rocke_b_sync(b);
}

/* ------------------------------------------------- block_exclusive_scan_i32 */

void rocke_block_exclusive_scan_i32(
    rocke_ir_builder_t* b, rocke_value_t* lds_buf, rocke_value_t* tid, int block_size, int length)
{
    const rocke_type_t* I32;
    rocke_value_t* c_length;
    rocke_value_t* in_bounds;
    int stride;
    rocke_value_t* in_range_left;
    rocke_value_t* left_idx;
    rocke_value_t* left_vec;
    rocke_value_t* left_val;
    rocke_value_t* shifted;
    rocke_if_t iff_final;

    if(b == NULL)
    {
        return;
    }
    if(length <= 0)
    {
        rocke_scan_set_err(b, ROCKE_ERR_VALUE, "length must be > 0 (got %d)", length);
        return;
    }
    if(length > block_size)
    {
        rocke_scan_set_err(b,
                           ROCKE_ERR_VALUE,
                           "length %d > block_size %d; multi-pass scans not implemented yet",
                           length,
                           block_size);
        return;
    }

    I32 = rocke_i32();

    /* c_length = b.const_i32(length); in_bounds = b.cmp_lt(tid, c_length) */
    c_length = rocke_b_const_i32(b, length);
    in_bounds = rocke_b_cmp_lt(b, tid, c_length);

    /* Inclusive Hillis-Steele scan. */
    stride = 1;
    while(stride < length)
    {
        /* c_stride = b.const_i32(stride) */
        rocke_value_t* c_stride = rocke_b_const_i32(b, stride);
        /* do_add = b.land(in_bounds, b.cmp_ge(tid, c_stride)) */
        rocke_value_t* do_add = rocke_b_land(b, in_bounds, rocke_b_cmp_ge(b, tid, c_stride));
        /* self_idx = b.select(in_bounds, tid, b.const_i32(0)) */
        rocke_value_t* self_idx = rocke_b_select(b, in_bounds, tid, rocke_b_const_i32(b, 0));
        /* left_idx = b.select(do_add, b.sub(tid, c_stride), b.const_i32(0)) */
        rocke_value_t* l_idx
            = rocke_b_select(b, do_add, rocke_b_sub(b, tid, c_stride), rocke_b_const_i32(b, 0));
        /* self_vec = b.smem_load_vN(lds_buf, self_idx, dtype=I32, n=1) */
        rocke_value_t* self_vec;
        rocke_value_t* l_vec;
        rocke_value_t* self_val;
        rocke_value_t* l_val;
        rocke_value_t* new_val;
        rocke_if_t iff;
        rocke_value_t* sidx[1];
        rocke_value_t* lidx[1];
        rocke_value_t* widx[1];

        sidx[0] = self_idx;
        self_vec = rocke_b_smem_load_vN(b, lds_buf, sidx, 1, I32, 1);
        /* left_vec = b.smem_load_vN(lds_buf, left_idx, dtype=I32, n=1) */
        lidx[0] = l_idx;
        l_vec = rocke_b_smem_load_vN(b, lds_buf, lidx, 1, I32, 1);
        /* self_val = b.vec_extract(self_vec, 0); left_val = b.vec_extract(left_vec, 0) */
        self_val = rocke_b_vec_extract(b, self_vec, 0);
        l_val = rocke_b_vec_extract(b, l_vec, 0);
        /* new_val = b.add(self_val, left_val) */
        new_val = rocke_b_add(b, self_val, l_val);
        /* b.sync() */
        rocke_b_sync(b);
        /* with b.scf_if(do_add): b.smem_store_vN(lds_buf, [tid], new_val, 1) */
        iff = rocke_b_scf_if(b, do_add);
        rocke_b_region_enter(b, iff.then_region);
        {
            widx[0] = tid;
            rocke_b_smem_store_vN(b, lds_buf, widx, 1, new_val, 1);
        }
        rocke_b_region_leave(b);
        /* b.sync() */
        rocke_b_sync(b);

        stride *= 2;
    }

    /* Convert inclusive -> exclusive via a one-position right-shift. */
    /* in_range_left = b.land(in_bounds, b.cmp_gt(tid, b.const_i32(0))) */
    in_range_left = rocke_b_land(b, in_bounds, rocke_b_cmp_gt(b, tid, rocke_b_const_i32(b, 0)));
    /* left_idx = b.select(in_range_left, b.sub(tid, b.const_i32(1)), b.const_i32(0)) */
    left_idx = rocke_b_select(
        b, in_range_left, rocke_b_sub(b, tid, rocke_b_const_i32(b, 1)), rocke_b_const_i32(b, 0));
    /* left_vec = b.smem_load_vN(lds_buf, left_idx, dtype=I32, n=1) */
    {
        rocke_value_t* lidx2[1];
        lidx2[0] = left_idx;
        left_vec = rocke_b_smem_load_vN(b, lds_buf, lidx2, 1, I32, 1);
    }
    /* left_val = b.vec_extract(left_vec, 0) */
    left_val = rocke_b_vec_extract(b, left_vec, 0);
    /* shifted = b.select(in_range_left, left_val, b.const_i32(0)) */
    shifted = rocke_b_select(b, in_range_left, left_val, rocke_b_const_i32(b, 0));
    /* b.sync() */
    rocke_b_sync(b);
    /* with b.scf_if(in_bounds): b.smem_store_vN(lds_buf, [tid], shifted, 1) */
    iff_final = rocke_b_scf_if(b, in_bounds);
    rocke_b_region_enter(b, iff_final.then_region);
    {
        rocke_value_t* widx2[1];
        widx2[0] = tid;
        rocke_b_smem_store_vN(b, lds_buf, widx2, 1, shifted, 1);
    }
    rocke_b_region_leave(b);
    /* b.sync() */
    rocke_b_sync(b);
}
