// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.instances.common._matmul_nbits_decode_gemv.c
 *   -- C99 port of rocke.instances.common._matmul_nbits_decode_gemv
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

#include "rocke/helper_rocke.instances.common._matmul_nbits_decode_gemv.h"

#include <string.h>

#include "rocke/ir_internal.h" /* rocke_i_live, rocke_i_set_err */

rocke_kernel_def_t* rocke_build_decode_gemv_matmul_nbits(
    rocke_ir_builder_t* b, const rocke_matmul_nbits_decode_gemv_spec_t* spec, const char* arch)
{
    int N, K, group, bs;
    int k_packed_stride; /* K // 2  -- packed-byte row stride for B */
    int k_group_stride; /* K // group -- scale row stride          */
    const rocke_type_t* scale_t;

    /* param Values */
    rocke_value_t* A;
    rocke_value_t* Bp;
    rocke_value_t* Sp;
    rocke_value_t* C;
    rocke_value_t* M;

    /* constants */
    rocke_value_t* c0;
    rocke_value_t* c1;
    rocke_value_t* c2;
    rocke_value_t* cN;
    rocke_value_t* cK;
    rocke_value_t* c_half_k;
    rocke_value_t* c_half_group;

    rocke_value_t* tid;
    rocke_value_t* n;

    rocke_if_t nguard;

    (void)arch; /* arch-agnostic body; accepted for signature parity. */

    if(!rocke_i_live(b))
    {
        return NULL;
    }
    if(spec == NULL)
    {
        return (rocke_kernel_def_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "build_decode_gemv_matmul_nbits: NULL spec");
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
    scale_t = (spec->scale_wire == ROCKE_NBITS_SCALE_F16) ? rocke_f16() : rocke_f32();

    /* b.kernel.attrs["max_workgroup_size"] = bs */
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", (int64_t)bs);

    /* ---- params ---- */
    {
        rocke_param_opts_t opts;

        /* A = b.param("A", PtrType(F16,"global"), noalias=True, readonly=True, align=16) */
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        A = rocke_b_param(b, "A", rocke_ptr_type(b, rocke_f16(), "global"), &opts);

        /* B = b.param("B", PtrType(I8,"global"), noalias=True, readonly=True, align=16) */
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        Bp = rocke_b_param(b, "B", rocke_ptr_type(b, rocke_i8(), "global"), &opts);

        /* Scales = b.param("Scales", PtrType(scale_t,"global"),
                            noalias=True, readonly=True, align=8) */
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 8;
        opts.align_set = true;
        Sp = rocke_b_param(b, "Scales", rocke_ptr_type(b, scale_t, "global"), &opts);

        /* C = b.param("C", PtrType(F16,"global"), noalias=True, writeonly=True, align=16) */
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.writeonly = true;
        opts.writeonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        C = rocke_b_param(b, "C", rocke_ptr_type(b, rocke_f16(), "global"), &opts);

        /* M = b.param("M", I32) */
        M = rocke_b_param(b, "M", rocke_i32(), NULL);
    }

    /* ---- constants ---- */
    c0 = rocke_b_const_i32(b, 0);
    c1 = rocke_b_const_i32(b, 1);
    c2 = rocke_b_const_i32(b, 2);
    cN = rocke_b_const_i32(b, (int64_t)N);
    cK = rocke_b_const_i32(b, (int64_t)K);
    c_half_k = rocke_b_const_i32(b, (int64_t)k_packed_stride);
    c_half_group = rocke_b_const_i32(b, (int64_t)(group / 2));

    /* tid = b.thread_id_x()
       n   = b.add(b.mul(b.block_id_x(), b.const_i32(bs)), tid) */
    tid = rocke_b_thread_id_x(b);
    /* Python evaluates block_id_x() before const_i32(bs); C arg evaluation is
     * right-to-left, so bind the block-id first to preserve SSA-id order. */
    {
        rocke_value_t* bid_x = rocke_b_block_id_x(b);
        n = rocke_b_add(b, rocke_b_mul(b, bid_x, rocke_b_const_i32(b, (int64_t)bs)), tid);
    }

    /* with b.scf_if(b.cmp_lt(n, cN)): */
    nguard = rocke_b_scf_if(b, rocke_b_cmp_lt(b, n, cN));
    if(!rocke_i_live(b))
    {
        return NULL;
    }
    rocke_b_region_enter(b, nguard.then_region);
    {
        rocke_value_t* b_byte_base;
        rocke_value_t* b_scale_base;
        rocke_for_t mloop;

        /* b_byte_base  = b.mul(n, c_half_k) */
        b_byte_base = rocke_b_mul(b, n, c_half_k);
        /* b_scale_base = b.mul(n, b.const_i32(k_group_stride)) */
        b_scale_base = rocke_b_mul(b, n, rocke_b_const_i32(b, (int64_t)k_group_stride));

        /* mloop = b.scf_for(c0, M, c1, iv_name="m") ; with mloop as m: */
        mloop = rocke_b_scf_for(b, c0, M, c1, "m");
        if(!rocke_i_live(b))
        {
            return NULL;
        }
        rocke_b_region_enter(b, mloop.body);
        {
            rocke_value_t* m = mloop.iv;
            rocke_value_t* a_row_base;
            rocke_iter_arg_t iter_args[1];
            rocke_for_t kloop;

            /* a_row_base = b.mul(m, cK) */
            a_row_base = rocke_b_mul(b, m, cK);

            /* kloop = b.scf_for_iter(c0, c_half_k, c1,
                                      [("acc", b.const_f32(0.0))], iv_name="j") */
            iter_args[0].name = "acc";
            iter_args[0].init = rocke_b_const_f32(b, 0.0);
            kloop = rocke_b_scf_for_iter(b,
                                         c0,
                                         c_half_k,
                                         c1,
                                         iter_args,
                                         1,
                                         "j",
                                         /*unroll=*/false,
                                         /*elide_trailing_barrier=*/true);
            if(!rocke_i_live(b))
            {
                return NULL;
            }
            rocke_b_region_enter(b, kloop.body);
            {
                /* with kloop as (j, accs): acc = accs[0] */
                rocke_value_t* j = kloop.iv;
                rocke_value_t* acc = kloop.iter_vars[0];

                rocke_value_t* byte;
                rocke_value_t* lo;
                rocke_value_t* hi;
                rocke_value_t* kgrp;
                rocke_value_t* scale;
                rocke_value_t* scale_f32;
                rocke_value_t* k_even;
                rocke_value_t* a_lo;
                rocke_value_t* a_hi;
                rocke_value_t* prod;
                rocke_value_t* yielded;

                /* byte = b.global_load(Bp, b.add(b_byte_base, j), I8) */
                byte = rocke_b_global_load(
                    b, Bp, rocke_b_add(b, b_byte_base, j), rocke_i8(), /*align*/ 0);

                /* lo, hi = unpack_i4_byte_to_pair_f32(b, byte) */
                rocke_unpack_i4_byte_to_pair_f32(b, byte, &lo, &hi);

                /* kgrp = b.div(j, c_half_group) */
                kgrp = rocke_b_div(b, j, c_half_group);

                /* scale = b.global_load(Sp, b.add(b_scale_base, kgrp), scale_t) */
                scale = rocke_b_global_load(
                    b, Sp, rocke_b_add(b, b_scale_base, kgrp), scale_t, /*align*/ 0);

                /* scale_f32 = b.cast_to_f32(scale) */
                scale_f32 = rocke_b_cast_to_f32(b, scale);

                /* k_even = b.mul(j, c2) */
                k_even = rocke_b_mul(b, j, c2);

                /* a_lo = b.cast_to_f32(b.global_load(A, b.add(a_row_base, k_even), F16)) */
                a_lo = rocke_b_cast_to_f32(
                    b,
                    rocke_b_global_load(
                        b, A, rocke_b_add(b, a_row_base, k_even), rocke_f16(), /*align*/ 0));

                /* a_hi = b.cast_to_f32(
                       b.global_load(A, b.add(a_row_base, b.add(k_even, c1)), F16)) */
                a_hi = rocke_b_cast_to_f32(
                    b,
                    rocke_b_global_load(b,
                                        A,
                                        rocke_b_add(b, a_row_base, rocke_b_add(b, k_even, c1)),
                                        rocke_f16(),
                                        /*align*/ 0));

                /* prod = b.fadd(
                       b.fmul(a_lo, b.fmul(lo, scale_f32)),
                       b.fmul(a_hi, b.fmul(hi, scale_f32)))
                   Python evaluates the fadd's first arg (the a_lo term) before
                   the second; C arg evaluation is right-to-left, so bind each
                   term to a temp in Python order to preserve SSA-id order. */
                {
                    rocke_value_t* prod_lo = rocke_b_fmul(b, a_lo, rocke_b_fmul(b, lo, scale_f32));
                    rocke_value_t* prod_hi = rocke_b_fmul(b, a_hi, rocke_b_fmul(b, hi, scale_f32));
                    prod = rocke_b_fadd(b, prod_lo, prod_hi);
                }

                /* b.scf_yield(b.fadd(acc, prod)) */
                yielded = rocke_b_fadd(b, acc, prod);
                rocke_b_scf_yield(b, &yielded, 1);
            }
            rocke_b_region_leave(b); /* leave kloop body */

            /* out_h = b.trunc_f32_to_f16(kloop.results[0]) */
            {
                rocke_value_t* kloop_result0 = NULL;
                rocke_value_t* out_h;

                if(kloop.op != NULL && kloop.op->num_results > 0)
                {
                    kloop_result0 = kloop.op->results[0];
                }
                out_h = rocke_b_trunc_f32_to_f16(b, kloop_result0);

                /* b.global_store(C, b.add(b.mul(m, cN), n), out_h) */
                rocke_b_global_store(
                    b, C, rocke_b_add(b, rocke_b_mul(b, m, cN), n), out_h, /*align*/ 0);
            }
        }
        rocke_b_region_leave(b); /* leave mloop body */
    }
    rocke_b_region_leave(b); /* leave scf_if then-region */

    if(!rocke_i_live(b))
    {
        return NULL;
    }

    /* return b.kernel */
    return b->kernel;
}
