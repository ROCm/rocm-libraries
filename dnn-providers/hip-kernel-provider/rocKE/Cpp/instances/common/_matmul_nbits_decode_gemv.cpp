// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_ck_dsl.instances.common._matmul_nbits_decode_gemv.c
 *   -- C99 port of ck_dsl.instances.common._matmul_nbits_decode_gemv
 *      .build_decode_gemv_matmul_nbits.
 *
 * Byte-faithful translation of the Python build. The op sequence, operands, and
 * attrs are reproduced exactly so the lowered IR is identical:
 *
 *     b.kernel.attrs["max_workgroup_size"] = bs
 *     A  = b.param("A", PtrType(F16,"global"), noalias,readonly,align=16)
 *     B  = b.param("B", PtrType(I8,"global"),  noalias,readonly,align=16)
 *     S  = b.param("Scales", PtrType(scale_t,"global"), noalias,readonly,align=8)
 *     C  = b.param("C", PtrType(F16,"global"), noalias,writeonly,align=16)
 *     M  = b.param("M", I32)
 *     ... constants ...
 *     tid = b.thread_id_x()
 *     n   = b.add(b.mul(b.block_id_x(), b.const_i32(bs)), tid)
 *     with b.scf_if(b.cmp_lt(n, cN)):
 *         b_byte_base  = b.mul(n, c_half_k)
 *         b_scale_base = b.mul(n, b.const_i32(k_group_stride))
 *         with b.scf_for(c0, M, c1, iv_name="m") as m:
 *             a_row_base = b.mul(m, cK)
 *             with b.scf_for_iter(c0, c_half_k, c1, [("acc", const_f32(0.0))],
 *                                 iv_name="j") as (j, accs):
 *                 acc  = accs[0]
 *                 byte = b.global_load(B, b.add(b_byte_base, j), I8)
 *                 lo, hi = unpack_i4_byte_to_pair_f32(b, byte)
 *                 kgrp  = b.div(j, c_half_group)
 *                 scale = b.global_load(S, b.add(b_scale_base, kgrp), scale_t)
 *                 scale_f32 = b.cast_to_f32(scale)
 *                 k_even = b.mul(j, c2)
 *                 a_lo = b.cast_to_f32(b.global_load(A, b.add(a_row_base, k_even), F16))
 *                 a_hi = b.cast_to_f32(
 *                     b.global_load(A, b.add(a_row_base, b.add(k_even, c1)), F16))
 *                 prod = b.fadd(
 *                     b.fmul(a_lo, b.fmul(lo, scale_f32)),
 *                     b.fmul(a_hi, b.fmul(hi, scale_f32)))
 *                 b.scf_yield(b.fadd(acc, prod))
 *             out_h = b.trunc_f32_to_f16(kloop.results[0])
 *             b.global_store(C, b.add(b.mul(m, cN), n), out_h)
 *     return b.kernel
 *
 * The Python `global_load(ptr, idx, dtype)` defaults align=1; we pass align<=0
 * (=> 1) to the C entry point to match. `global_store(ptr, idx, value)` likewise
 * defaults align=1.
 */

#include "ckc/helper_ck_dsl.instances.common._matmul_nbits_decode_gemv.h"

#include <string.h>

#include "ckc/ir_internal.h" /* ckc_i_live, ckc_i_set_err */

ckc_kernel_def_t* ckc_build_decode_gemv_matmul_nbits(
    ckc_ir_builder_t* b, const ckc_matmul_nbits_decode_gemv_spec_t* spec, const char* arch)
{
    int N, K, group, bs;
    int k_packed_stride; /* K // 2  -- packed-byte row stride for B */
    int k_group_stride; /* K // group -- scale row stride          */
    const ckc_type_t* scale_t;

    /* param Values */
    ckc_value_t* A;
    ckc_value_t* Bp;
    ckc_value_t* Sp;
    ckc_value_t* C;
    ckc_value_t* M;

    /* constants */
    ckc_value_t* c0;
    ckc_value_t* c1;
    ckc_value_t* c2;
    ckc_value_t* cN;
    ckc_value_t* cK;
    ckc_value_t* c_half_k;
    ckc_value_t* c_half_group;

    ckc_value_t* tid;
    ckc_value_t* n;

    ckc_if_t nguard;

    (void)arch; /* arch-agnostic body; accepted for signature parity. */

    if(!ckc_i_live(b))
    {
        return NULL;
    }
    if(spec == NULL)
    {
        return (ckc_kernel_def_t*)ckc_i_set_err(
            b, CKC_ERR_VALUE, "build_decode_gemv_matmul_nbits: NULL spec");
    }

    /* N, K, group = spec.N, spec.K, spec.group_size
       bs = spec.block_size */
    N = spec->N;
    K = spec->K;
    group = spec->group_size;
    bs = spec->block_size;

    /* k_packed_stride = K // 2 ; k_group_stride = K // group */
    k_packed_stride = K / 2;
    k_group_stride = K / group;

    /* scale_t = F16 if _scale_wire_dtype(spec.scale_dtype) == "f16" else F32 */
    scale_t = (spec->scale_wire == CKC_NBITS_SCALE_F16) ? ckc_f16() : ckc_f32();

    /* b.kernel.attrs["max_workgroup_size"] = bs */
    ckc_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", (int64_t)bs);

    /* ---- params ---- */
    {
        ckc_param_opts_t opts;

        /* A = b.param("A", PtrType(F16,"global"), noalias=True, readonly=True, align=16) */
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        A = ckc_b_param(b, "A", ckc_ptr_type(b, ckc_f16(), "global"), &opts);

        /* B = b.param("B", PtrType(I8,"global"), noalias=True, readonly=True, align=16) */
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        Bp = ckc_b_param(b, "B", ckc_ptr_type(b, ckc_i8(), "global"), &opts);

        /* Scales = b.param("Scales", PtrType(scale_t,"global"),
                            noalias=True, readonly=True, align=8) */
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 8;
        opts.align_set = true;
        Sp = ckc_b_param(b, "Scales", ckc_ptr_type(b, scale_t, "global"), &opts);

        /* C = b.param("C", PtrType(F16,"global"), noalias=True, writeonly=True, align=16) */
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.writeonly = true;
        opts.writeonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        C = ckc_b_param(b, "C", ckc_ptr_type(b, ckc_f16(), "global"), &opts);

        /* M = b.param("M", I32) */
        M = ckc_b_param(b, "M", ckc_i32(), NULL);
    }

    /* ---- constants ---- */
    c0 = ckc_b_const_i32(b, 0);
    c1 = ckc_b_const_i32(b, 1);
    c2 = ckc_b_const_i32(b, 2);
    cN = ckc_b_const_i32(b, (int64_t)N);
    cK = ckc_b_const_i32(b, (int64_t)K);
    c_half_k = ckc_b_const_i32(b, (int64_t)k_packed_stride);
    c_half_group = ckc_b_const_i32(b, (int64_t)(group / 2));

    /* tid = b.thread_id_x()
       n   = b.add(b.mul(b.block_id_x(), b.const_i32(bs)), tid) */
    tid = ckc_b_thread_id_x(b);
    /* Python evaluates block_id_x() before const_i32(bs); C arg evaluation is
     * right-to-left, so bind the block-id first to preserve SSA-id order. */
    {
        ckc_value_t* bid_x = ckc_b_block_id_x(b);
        n = ckc_b_add(b, ckc_b_mul(b, bid_x, ckc_b_const_i32(b, (int64_t)bs)), tid);
    }

    /* with b.scf_if(b.cmp_lt(n, cN)): */
    nguard = ckc_b_scf_if(b, ckc_b_cmp_lt(b, n, cN));
    if(!ckc_i_live(b))
    {
        return NULL;
    }
    ckc_b_region_enter(b, nguard.then_region);
    {
        ckc_value_t* b_byte_base;
        ckc_value_t* b_scale_base;
        ckc_for_t mloop;

        /* b_byte_base  = b.mul(n, c_half_k) */
        b_byte_base = ckc_b_mul(b, n, c_half_k);
        /* b_scale_base = b.mul(n, b.const_i32(k_group_stride)) */
        b_scale_base = ckc_b_mul(b, n, ckc_b_const_i32(b, (int64_t)k_group_stride));

        /* mloop = b.scf_for(c0, M, c1, iv_name="m") ; with mloop as m: */
        mloop = ckc_b_scf_for(b, c0, M, c1, "m");
        if(!ckc_i_live(b))
        {
            return NULL;
        }
        ckc_b_region_enter(b, mloop.body);
        {
            ckc_value_t* m = mloop.iv;
            ckc_value_t* a_row_base;
            ckc_iter_arg_t iter_args[1];
            ckc_for_t kloop;

            /* a_row_base = b.mul(m, cK) */
            a_row_base = ckc_b_mul(b, m, cK);

            /* kloop = b.scf_for_iter(c0, c_half_k, c1,
                                      [("acc", b.const_f32(0.0))], iv_name="j") */
            iter_args[0].name = "acc";
            iter_args[0].init = ckc_b_const_f32(b, 0.0);
            kloop = ckc_b_scf_for_iter(b,
                                       c0,
                                       c_half_k,
                                       c1,
                                       iter_args,
                                       1,
                                       "j",
                                       /*unroll=*/false,
                                       /*elide_trailing_barrier=*/true);
            if(!ckc_i_live(b))
            {
                return NULL;
            }
            ckc_b_region_enter(b, kloop.body);
            {
                /* with kloop as (j, accs): acc = accs[0] */
                ckc_value_t* j = kloop.iv;
                ckc_value_t* acc = kloop.iter_vars[0];

                ckc_value_t* byte;
                ckc_value_t* lo;
                ckc_value_t* hi;
                ckc_value_t* kgrp;
                ckc_value_t* scale;
                ckc_value_t* scale_f32;
                ckc_value_t* k_even;
                ckc_value_t* a_lo;
                ckc_value_t* a_hi;
                ckc_value_t* prod;
                ckc_value_t* yielded;

                /* byte = b.global_load(Bp, b.add(b_byte_base, j), I8) */
                byte
                    = ckc_b_global_load(b, Bp, ckc_b_add(b, b_byte_base, j), ckc_i8(), /*align*/ 0);

                /* lo, hi = unpack_i4_byte_to_pair_f32(b, byte) */
                ckc_unpack_i4_byte_to_pair_f32(b, byte, &lo, &hi);

                /* kgrp = b.div(j, c_half_group) */
                kgrp = ckc_b_div(b, j, c_half_group);

                /* scale = b.global_load(Sp, b.add(b_scale_base, kgrp), scale_t) */
                scale = ckc_b_global_load(
                    b, Sp, ckc_b_add(b, b_scale_base, kgrp), scale_t, /*align*/ 0);

                /* scale_f32 = b.cast_to_f32(scale) */
                scale_f32 = ckc_b_cast_to_f32(b, scale);

                /* k_even = b.mul(j, c2) */
                k_even = ckc_b_mul(b, j, c2);

                /* a_lo = b.cast_to_f32(b.global_load(A, b.add(a_row_base, k_even), F16)) */
                a_lo = ckc_b_cast_to_f32(
                    b,
                    ckc_b_global_load(
                        b, A, ckc_b_add(b, a_row_base, k_even), ckc_f16(), /*align*/ 0));

                /* a_hi = b.cast_to_f32(
                       b.global_load(A, b.add(a_row_base, b.add(k_even, c1)), F16)) */
                a_hi = ckc_b_cast_to_f32(
                    b,
                    ckc_b_global_load(b,
                                      A,
                                      ckc_b_add(b, a_row_base, ckc_b_add(b, k_even, c1)),
                                      ckc_f16(),
                                      /*align*/ 0));

                /* prod = b.fadd(
                       b.fmul(a_lo, b.fmul(lo, scale_f32)),
                       b.fmul(a_hi, b.fmul(hi, scale_f32)))
                   Python evaluates the fadd's first arg (the a_lo term) before
                   the second; C arg evaluation is right-to-left, so bind each
                   term to a temp in Python order to preserve SSA-id order. */
                {
                    ckc_value_t* prod_lo = ckc_b_fmul(b, a_lo, ckc_b_fmul(b, lo, scale_f32));
                    ckc_value_t* prod_hi = ckc_b_fmul(b, a_hi, ckc_b_fmul(b, hi, scale_f32));
                    prod = ckc_b_fadd(b, prod_lo, prod_hi);
                }

                /* b.scf_yield(b.fadd(acc, prod)) */
                yielded = ckc_b_fadd(b, acc, prod);
                ckc_b_scf_yield(b, &yielded, 1);
            }
            ckc_b_region_leave(b); /* leave kloop body */

            /* out_h = b.trunc_f32_to_f16(kloop.results[0]) */
            {
                ckc_value_t* kloop_result0 = NULL;
                ckc_value_t* out_h;

                if(kloop.op != NULL && kloop.op->num_results > 0)
                {
                    kloop_result0 = kloop.op->results[0];
                }
                out_h = ckc_b_trunc_f32_to_f16(b, kloop_result0);

                /* b.global_store(C, b.add(b.mul(m, cN), n), out_h) */
                ckc_b_global_store(b, C, ckc_b_add(b, ckc_b_mul(b, m, cN), n), out_h, /*align*/ 0);
            }
        }
        ckc_b_region_leave(b); /* leave mloop body */
    }
    ckc_b_region_leave(b); /* leave scf_if then-region */

    if(!ckc_i_live(b))
    {
        return NULL;
    }

    /* return b.kernel */
    return b->kernel;
}
