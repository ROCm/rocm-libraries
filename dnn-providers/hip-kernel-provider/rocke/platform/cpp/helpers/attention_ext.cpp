// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_helper_rocke.helpers.attention.c -- C99 port of a second selection of
 * symbols from rocke/helpers/attention.py (companion to
 * helper_rocke.helpers.attention.c).
 *
 * Each helper reproduces its Python counterpart's rocke_b_* builder-call sequence
 * byte-faithfully (same ops, same order, same operands). Host-side control
 * structure (the fixed-count XOR butterfly, the dtype-name dispatch, the
 * binary-search scf.for loop) is reproduced exactly so the emitted op stream is
 * identical to the Python.
 *
 * Lifetime: every node is arena-owned (rocke_ir_builder_t.arena). Nothing is freed
 * individually; the arena bulk-frees the whole graph.
 */

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "rocke/error.hpp"
#include "rocke/helper_helper_rocke.helpers.attention.h"
#include "rocke/ir.h"

/* ----------------------------------------------------------------- helpers */

/* Raise the failure as a ckc::Error (mirroring the Python `raise`); the public
 * entry boundary catches it and records status + message on the builder, so the
 * C ABI is unchanged. [[noreturn]] keeps the existing
 * `return (T*)rocke_attn2_set_err(...)` call sites valid -- the cast/return is
 * simply never reached. */
[[noreturn]] static void*
    rocke_attn2_set_err(rocke_ir_builder_t* b, rocke_status_t st, const char* fmt, ...)
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

/* ------------------------------------------------------- softcap (log2-domain) */

rocke_value_t* rocke_apply_softcap_log2(rocke_ir_builder_t* b,
                                        rocke_value_t* score_log2,
                                        rocke_value_t* softcap)
{
    /* sdiv = b.fdiv(score_log2, softcap)
     * p1 = b.exp2(sdiv)
     * p2 = b.exp2(b.fneg(sdiv))
     * return b.fmul(softcap, b.fmul(b.fsub(p1, p2), b.rcp(b.fadd(p1, p2)))) */
    rocke_value_t* sdiv = rocke_b_fdiv(b, score_log2, softcap);
    rocke_value_t* p1 = rocke_b_exp2(b, sdiv);
    rocke_value_t* p2 = rocke_b_exp2(b, rocke_b_fneg(b, sdiv));
    /* Sequence the inner sub/rcp so the C argument-evaluation order matches the
     * Python: b.fsub(p1, p2) is emitted before b.rcp(b.fadd(p1, p2)). */
    rocke_value_t* diff = rocke_b_fsub(b, p1, p2);
    rocke_value_t* den = rocke_b_rcp(b, rocke_b_fadd(b, p1, p2));
    return rocke_b_fmul(b, softcap, rocke_b_fmul(b, diff, den));
}

/* ------------------------------------------------------- MFMA dtype dispatch */

rocke_value_t* rocke_mfma_16x16x16_for_dtype(rocke_ir_builder_t* b,
                                             const rocke_type_t* dtype,
                                             rocke_value_t* a,
                                             rocke_value_t* bv,
                                             rocke_value_t* c)
{
    /* if dtype.name == "f16": return b.mfma_f32_16x16x16_f16(a, bv, c)
     * if dtype.name == "bf16": return b.mfma_f32_16x16x16_bf16(a, bv, c)
     * raise ValueError(f"unsupported MFMA 16x16x16 dtype {dtype.name}") */
    if(dtype == NULL || dtype->name == NULL)
    {
        return (rocke_value_t*)rocke_attn2_set_err(
            b, ROCKE_ERR_VALUE, "unsupported MFMA 16x16x16 dtype (null)");
    }
    if(strcmp(dtype->name, "f16") == 0)
    {
        return rocke_b_mfma_f32_16x16x16_f16(b, a, bv, c);
    }
    if(strcmp(dtype->name, "bf16") == 0)
    {
        return rocke_b_mfma_f32_16x16x16_bf16(b, a, bv, c);
    }
    return (rocke_value_t*)rocke_attn2_set_err(
        b, ROCKE_ERR_VALUE, "unsupported MFMA 16x16x16 dtype %s", dtype->name);
}

/* ------------------------------------------------- wave64 cross-lane reduction */

rocke_value_t* rocke_wave64_reduce_max(rocke_ir_builder_t* b, rocke_value_t* v)
{
    /* cur = v
     * for k in range(6):
     *     remote = b.warp_shuffle_xor(cur, 1 << k)
     *     cur = b.fmax(cur, remote)
     * return cur */
    rocke_value_t* cur = v;
    int k;
    for(k = 0; k < 6; ++k)
    {
        rocke_value_t* remote = rocke_b_warp_shuffle_xor(b, cur, 1 << k);
        cur = rocke_b_fmax(b, cur, remote);
    }
    return cur;
}

rocke_value_t* rocke_wave64_reduce_sum(rocke_ir_builder_t* b, rocke_value_t* v)
{
    /* cur = v
     * for k in range(6):
     *     remote = b.warp_shuffle_xor(cur, 1 << k)
     *     cur = b.fadd(cur, remote)
     * return cur */
    rocke_value_t* cur = v;
    int k;
    for(k = 0; k < 6; ++k)
    {
        rocke_value_t* remote = rocke_b_warp_shuffle_xor(b, cur, 1 << k);
        cur = rocke_b_fadd(b, cur, remote);
    }
    return cur;
}

/* ----------------------------------------------- binary search on cu_q */

rocke_value_t* rocke_binary_search_seq_idx(rocke_ir_builder_t* b,
                                           rocke_value_t* cu_q,
                                           rocke_value_t* q_block_global_idx,
                                           rocke_value_t* num_seqs,
                                           int block_q,
                                           int iterations,
                                           bool per_token)
{
    rocke_value_t* bq;
    rocke_iter_arg_t iter_args[2];
    rocke_for_t loop;
    rocke_value_t* left;
    rocke_value_t* right;
    rocke_value_t* done;
    rocke_value_t* mid;
    rocke_value_t* val;
    rocke_value_t* mid_val;
    rocke_value_t* le;
    rocke_value_t* nl;
    rocke_value_t* nr;
    rocke_value_t* yields[2];
    rocke_value_t* res0;

    /* bq = b.const_i32(block_q) */
    bq = rocke_b_const_i32(b, (int64_t)block_q);

    /* loop = b.scf_for_iter(0, iterations, 1,
     *     [("left", 0), ("right", num_seqs)], iv_name="bs_i")
     * Python evaluates args left-to-right: lb, ub, step consts are emitted
     * before the iter_arg init consts. C arg-eval order is unspecified, so
     * hoist lb/ub/step first, then build the iter inits, to pin IR order. */
    {
        rocke_value_t* lb = rocke_b_const_i32(b, 0);
        rocke_value_t* ub = rocke_b_const_i32(b, (int64_t)iterations);
        rocke_value_t* step = rocke_b_const_i32(b, 1);
        iter_args[0].name = "left";
        iter_args[0].init = rocke_b_const_i32(b, 0);
        iter_args[1].name = "right";
        iter_args[1].init = num_seqs;
        loop = rocke_b_scf_for_iter(b, lb, ub, step, iter_args, 2, "bs_i", false, true);
    }

    /* with loop as (_iv, (left, right)): */
    rocke_b_region_enter(b, loop.body);
    left = loop.iter_vars[0];
    right = loop.iter_vars[1];

    /* done = b.cmp_ge(left, right)
     * mid = b.div(b.add(left, right), b.const_i32(2)) */
    done = rocke_b_cmp_ge(b, left, right);
    {
        /* Sequence the add before the const so the SSA id order matches Python. */
        rocke_value_t* lr_sum = rocke_b_add(b, left, right);
        mid = rocke_b_div(b, lr_sum, rocke_b_const_i32(b, 2));
    }

    /* val = b.global_load_i32(cu_q, mid) */
    val = rocke_b_global_load_i32(b, cu_q, mid, 4);

    /* if per_token: mid_val = val
     * else: mid_val = b.add(b.div(val, bq), mid) */
    if(per_token)
    {
        mid_val = val;
    }
    else
    {
        mid_val = rocke_b_add(b, rocke_b_div(b, val, bq), mid);
    }

    /* le = b.cmp_le(mid_val, q_block_global_idx)
     * nl = b.select(le, b.add(mid, 1), left)
     * nr = b.select(le, right, mid) */
    le = rocke_b_cmp_le(b, mid_val, q_block_global_idx);
    nl = rocke_b_select(b, le, rocke_b_add(b, mid, rocke_b_const_i32(b, 1)), left);
    nr = rocke_b_select(b, le, right, mid);

    /* b.scf_yield(b.select(done, left, nl), b.select(done, right, nr)) */
    yields[0] = rocke_b_select(b, done, left, nl);
    yields[1] = rocke_b_select(b, done, right, nr);
    rocke_b_scf_yield(b, yields, 2);
    rocke_b_region_leave(b);

    /* return b.sub(loop.results[0], b.const_i32(1)) */
    res0 = (loop.op != NULL) ? loop.op->results[0] : NULL;
    return rocke_b_sub(b, res0, rocke_b_const_i32(b, 1));
}
