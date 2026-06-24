// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_ck_dsl.helpers.scan.c -- C99 port of selected symbols from
 * ck_dsl/helpers/scan.py.
 *
 * Ported symbols: lds_zero_i32, block_exclusive_scan_i32.
 *
 * Each helper reproduces its Python counterpart's ckc_b_* builder-call sequence
 * byte-faithfully (same ops, same order, same operands). The host-side control
 * structure (the chunk loop, the Hillis-Steele while-loop, the scf_if scoping)
 * is reproduced exactly so the emitted op stream is identical to the Python.
 *
 * scf_if scoping: Python's ``with b.scf_if(cond):`` maps to
 *   ckc_if_t iff = ckc_b_scf_if(b, cond);
 *   ckc_b_region_enter(b, iff.then_region);
 *   ... body ...
 *   ckc_b_region_leave(b);
 *
 * Lifetime: every node is arena-owned (ckc_ir_builder_t.arena). Nothing is freed
 * individually; the arena bulk-frees the whole graph.
 */

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>

#include "ckc/error.hpp"
#include "ckc/helper_ck_dsl.helpers.scan.h"
#include "ckc/ir.h"

/* ----------------------------------------------------------------- helpers */

/* Raise the failure as a ckc::Error (mirroring the Python `raise`); the public
 * entry boundary catches it and records status + message on the builder, so the
 * C ABI is unchanged. [[noreturn]] keeps the existing `ckc_scan_set_err(...);
 * return;` call sites valid -- the trailing return is simply never reached. */
[[noreturn]] static void
ckc_scan_set_err(ckc_ir_builder_t* b, ckc_status_t st, const char* fmt, ...)
{
    (void)b;
    char msg[CKC_ERR_MSG_CAP];
    va_list ap;
    va_start(ap, fmt);
    (void)vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    msg[sizeof(msg) - 1] = '\0';
    ckc::raise_status(st, msg);
}

/* ------------------------------------------------------------- lds_zero_i32 */

void ckc_lds_zero_i32(
    ckc_ir_builder_t* b, ckc_value_t* lds_buf, ckc_value_t* tid, int block_size, int length)
{
    int chunks;
    int c;
    ckc_value_t* c_block;
    ckc_value_t* c_length;
    ckc_value_t* c_zero;

    if(b == NULL)
    {
        return;
    }
    if(length <= 0)
    {
        ckc_scan_set_err(b, CKC_ERR_VALUE, "length must be > 0 (got %d)", length);
        return;
    }

    chunks   = (length + block_size - 1) / block_size;
    c_block  = ckc_b_const_i32(b, block_size);
    c_length = ckc_b_const_i32(b, length);
    c_zero   = ckc_b_const_i32(b, 0);

    for(c = 0; c < chunks; ++c)
    {
        /* local = b.add(tid, b.mul(b.const_i32(c), c_block)) */
        ckc_value_t* local = ckc_b_add(b, tid, ckc_b_mul(b, ckc_b_const_i32(b, c), c_block));
        /* in_bounds = b.cmp_lt(local, c_length) */
        ckc_value_t* in_bounds = ckc_b_cmp_lt(b, local, c_length);
        /* with b.scf_if(in_bounds): b.smem_store_vN(lds_buf, [local], c_zero, 1) */
        ckc_if_t iff = ckc_b_scf_if(b, in_bounds);
        ckc_b_region_enter(b, iff.then_region);
        {
            ckc_value_t* idx[1];
            idx[0] = local;
            ckc_b_smem_store_vN(b, lds_buf, idx, 1, c_zero, 1);
        }
        ckc_b_region_leave(b);
    }
    ckc_b_sync(b);
}

/* ------------------------------------------------- block_exclusive_scan_i32 */

void ckc_block_exclusive_scan_i32(
    ckc_ir_builder_t* b, ckc_value_t* lds_buf, ckc_value_t* tid, int block_size, int length)
{
    const ckc_type_t* I32;
    ckc_value_t* c_length;
    ckc_value_t* in_bounds;
    int stride;
    ckc_value_t* in_range_left;
    ckc_value_t* left_idx;
    ckc_value_t* left_vec;
    ckc_value_t* left_val;
    ckc_value_t* shifted;
    ckc_if_t iff_final;

    if(b == NULL)
    {
        return;
    }
    if(length <= 0)
    {
        ckc_scan_set_err(b, CKC_ERR_VALUE, "length must be > 0 (got %d)", length);
        return;
    }
    if(length > block_size)
    {
        ckc_scan_set_err(b,
                         CKC_ERR_VALUE,
                         "length %d > block_size %d; multi-pass scans not implemented yet",
                         length,
                         block_size);
        return;
    }

    I32 = ckc_i32();

    /* c_length = b.const_i32(length); in_bounds = b.cmp_lt(tid, c_length) */
    c_length  = ckc_b_const_i32(b, length);
    in_bounds = ckc_b_cmp_lt(b, tid, c_length);

    /* Inclusive Hillis-Steele scan. */
    stride = 1;
    while(stride < length)
    {
        /* c_stride = b.const_i32(stride) */
        ckc_value_t* c_stride = ckc_b_const_i32(b, stride);
        /* do_add = b.land(in_bounds, b.cmp_ge(tid, c_stride)) */
        ckc_value_t* do_add = ckc_b_land(b, in_bounds, ckc_b_cmp_ge(b, tid, c_stride));
        /* self_idx = b.select(in_bounds, tid, b.const_i32(0)) */
        ckc_value_t* self_idx = ckc_b_select(b, in_bounds, tid, ckc_b_const_i32(b, 0));
        /* left_idx = b.select(do_add, b.sub(tid, c_stride), b.const_i32(0)) */
        ckc_value_t* l_idx =
            ckc_b_select(b, do_add, ckc_b_sub(b, tid, c_stride), ckc_b_const_i32(b, 0));
        /* self_vec = b.smem_load_vN(lds_buf, self_idx, dtype=I32, n=1) */
        ckc_value_t* self_vec;
        ckc_value_t* l_vec;
        ckc_value_t* self_val;
        ckc_value_t* l_val;
        ckc_value_t* new_val;
        ckc_if_t iff;
        ckc_value_t* sidx[1];
        ckc_value_t* lidx[1];
        ckc_value_t* widx[1];

        sidx[0]  = self_idx;
        self_vec = ckc_b_smem_load_vN(b, lds_buf, sidx, 1, I32, 1);
        /* left_vec = b.smem_load_vN(lds_buf, left_idx, dtype=I32, n=1) */
        lidx[0] = l_idx;
        l_vec   = ckc_b_smem_load_vN(b, lds_buf, lidx, 1, I32, 1);
        /* self_val = b.vec_extract(self_vec, 0); left_val = b.vec_extract(left_vec, 0) */
        self_val = ckc_b_vec_extract(b, self_vec, 0);
        l_val    = ckc_b_vec_extract(b, l_vec, 0);
        /* new_val = b.add(self_val, left_val) */
        new_val = ckc_b_add(b, self_val, l_val);
        /* b.sync() */
        ckc_b_sync(b);
        /* with b.scf_if(do_add): b.smem_store_vN(lds_buf, [tid], new_val, 1) */
        iff = ckc_b_scf_if(b, do_add);
        ckc_b_region_enter(b, iff.then_region);
        {
            widx[0] = tid;
            ckc_b_smem_store_vN(b, lds_buf, widx, 1, new_val, 1);
        }
        ckc_b_region_leave(b);
        /* b.sync() */
        ckc_b_sync(b);

        stride *= 2;
    }

    /* Convert inclusive -> exclusive via a one-position right-shift. */
    /* in_range_left = b.land(in_bounds, b.cmp_gt(tid, b.const_i32(0))) */
    in_range_left = ckc_b_land(b, in_bounds, ckc_b_cmp_gt(b, tid, ckc_b_const_i32(b, 0)));
    /* left_idx = b.select(in_range_left, b.sub(tid, b.const_i32(1)), b.const_i32(0)) */
    left_idx = ckc_b_select(
        b, in_range_left, ckc_b_sub(b, tid, ckc_b_const_i32(b, 1)), ckc_b_const_i32(b, 0));
    /* left_vec = b.smem_load_vN(lds_buf, left_idx, dtype=I32, n=1) */
    {
        ckc_value_t* lidx2[1];
        lidx2[0] = left_idx;
        left_vec = ckc_b_smem_load_vN(b, lds_buf, lidx2, 1, I32, 1);
    }
    /* left_val = b.vec_extract(left_vec, 0) */
    left_val = ckc_b_vec_extract(b, left_vec, 0);
    /* shifted = b.select(in_range_left, left_val, b.const_i32(0)) */
    shifted = ckc_b_select(b, in_range_left, left_val, ckc_b_const_i32(b, 0));
    /* b.sync() */
    ckc_b_sync(b);
    /* with b.scf_if(in_bounds): b.smem_store_vN(lds_buf, [tid], shifted, 1) */
    iff_final = ckc_b_scf_if(b, in_bounds);
    ckc_b_region_enter(b, iff_final.then_region);
    {
        ckc_value_t* widx2[1];
        widx2[0] = tid;
        ckc_b_smem_store_vN(b, lds_buf, widx2, 1, shifted, 1);
    }
    ckc_b_region_leave(b);
    /* b.sync() */
    ckc_b_sync(b);
}
