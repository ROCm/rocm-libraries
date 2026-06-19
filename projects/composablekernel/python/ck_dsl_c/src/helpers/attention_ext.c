/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_helper_ck_dsl.helpers.attention.c -- C99 port of a second selection of
 * symbols from ck_dsl/helpers/attention.py (companion to
 * helper_ck_dsl.helpers.attention.c).
 *
 * Each helper reproduces its Python counterpart's ckc_b_* builder-call sequence
 * byte-faithfully (same ops, same order, same operands). Host-side control
 * structure (the fixed-count XOR butterfly, the dtype-name dispatch, the
 * binary-search scf.for loop) is reproduced exactly so the emitted op stream is
 * identical to the Python.
 *
 * Lifetime: every node is arena-owned (ckc_ir_builder_t.arena). Nothing is freed
 * individually; the arena bulk-frees the whole graph.
 */

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "ckc/helper_helper_ck_dsl.helpers.attention.h"
#include "ckc/error.hpp"
#include "ckc/ir.h"

/* ----------------------------------------------------------------- helpers */

/* Raise the failure as a ckc::Error (mirroring the Python `raise`); the public
 * entry boundary catches it and records status + message on the builder, so the
 * C ABI is unchanged. [[noreturn]] keeps the existing
 * `return (T*)ckc_attn2_set_err(...)` call sites valid -- the cast/return is
 * simply never reached. */
[[noreturn]] static void* ckc_attn2_set_err(ckc_ir_builder_t* b, ckc_status_t st, const char* fmt, ...)
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

/* ------------------------------------------------------- softcap (log2-domain) */

ckc_value_t* ckc_apply_softcap_log2(ckc_ir_builder_t* b,
                                    ckc_value_t* score_log2,
                                    ckc_value_t* softcap)
{
    /* sdiv = b.fdiv(score_log2, softcap)
     * p1 = b.exp2(sdiv)
     * p2 = b.exp2(b.fneg(sdiv))
     * return b.fmul(softcap, b.fmul(b.fsub(p1, p2), b.rcp(b.fadd(p1, p2)))) */
    ckc_value_t* sdiv = ckc_b_fdiv(b, score_log2, softcap);
    ckc_value_t* p1   = ckc_b_exp2(b, sdiv);
    ckc_value_t* p2   = ckc_b_exp2(b, ckc_b_fneg(b, sdiv));
    /* Sequence the inner sub/rcp so the C argument-evaluation order matches the
     * Python: b.fsub(p1, p2) is emitted before b.rcp(b.fadd(p1, p2)). */
    ckc_value_t* diff = ckc_b_fsub(b, p1, p2);
    ckc_value_t* den  = ckc_b_rcp(b, ckc_b_fadd(b, p1, p2));
    return ckc_b_fmul(b, softcap, ckc_b_fmul(b, diff, den));
}

/* ------------------------------------------------------- MFMA dtype dispatch */

ckc_value_t* ckc_mfma_16x16x16_for_dtype(ckc_ir_builder_t* b,
                                         const ckc_type_t* dtype,
                                         ckc_value_t* a,
                                         ckc_value_t* bv,
                                         ckc_value_t* c)
{
    /* if dtype.name == "f16": return b.mfma_f32_16x16x16_f16(a, bv, c)
     * if dtype.name == "bf16": return b.mfma_f32_16x16x16_bf16(a, bv, c)
     * raise ValueError(f"unsupported MFMA 16x16x16 dtype {dtype.name}") */
    if(dtype == NULL || dtype->name == NULL)
    {
        return (ckc_value_t*)ckc_attn2_set_err(
            b, CKC_ERR_VALUE, "unsupported MFMA 16x16x16 dtype (null)");
    }
    if(strcmp(dtype->name, "f16") == 0)
    {
        return ckc_b_mfma_f32_16x16x16_f16(b, a, bv, c);
    }
    if(strcmp(dtype->name, "bf16") == 0)
    {
        return ckc_b_mfma_f32_16x16x16_bf16(b, a, bv, c);
    }
    return (ckc_value_t*)ckc_attn2_set_err(
        b, CKC_ERR_VALUE, "unsupported MFMA 16x16x16 dtype %s", dtype->name);
}

/* ------------------------------------------------- wave64 cross-lane reduction */

ckc_value_t* ckc_wave64_reduce_max(ckc_ir_builder_t* b, ckc_value_t* v)
{
    /* cur = v
     * for k in range(6):
     *     remote = b.warp_shuffle_xor(cur, 1 << k)
     *     cur = b.fmax(cur, remote)
     * return cur */
    ckc_value_t* cur = v;
    int k;
    for(k = 0; k < 6; ++k)
    {
        ckc_value_t* remote = ckc_b_warp_shuffle_xor(b, cur, 1 << k);
        cur                 = ckc_b_fmax(b, cur, remote);
    }
    return cur;
}

ckc_value_t* ckc_wave64_reduce_sum(ckc_ir_builder_t* b, ckc_value_t* v)
{
    /* cur = v
     * for k in range(6):
     *     remote = b.warp_shuffle_xor(cur, 1 << k)
     *     cur = b.fadd(cur, remote)
     * return cur */
    ckc_value_t* cur = v;
    int k;
    for(k = 0; k < 6; ++k)
    {
        ckc_value_t* remote = ckc_b_warp_shuffle_xor(b, cur, 1 << k);
        cur                 = ckc_b_fadd(b, cur, remote);
    }
    return cur;
}

/* ----------------------------------------------- binary search on cu_q */

ckc_value_t* ckc_binary_search_seq_idx(ckc_ir_builder_t* b,
                                       ckc_value_t* cu_q,
                                       ckc_value_t* q_block_global_idx,
                                       ckc_value_t* num_seqs,
                                       int block_q,
                                       int iterations,
                                       bool per_token)
{
    ckc_value_t* bq;
    ckc_iter_arg_t iter_args[2];
    ckc_for_t loop;
    ckc_value_t* left;
    ckc_value_t* right;
    ckc_value_t* done;
    ckc_value_t* mid;
    ckc_value_t* val;
    ckc_value_t* mid_val;
    ckc_value_t* le;
    ckc_value_t* nl;
    ckc_value_t* nr;
    ckc_value_t* yields[2];
    ckc_value_t* res0;

    /* bq = b.const_i32(block_q) */
    bq = ckc_b_const_i32(b, (int64_t)block_q);

    /* loop = b.scf_for_iter(0, iterations, 1,
     *     [("left", 0), ("right", num_seqs)], iv_name="bs_i")
     * Python evaluates args left-to-right: lb, ub, step consts are emitted
     * before the iter_arg init consts. C arg-eval order is unspecified, so
     * hoist lb/ub/step first, then build the iter inits, to pin IR order. */
    {
        ckc_value_t* lb   = ckc_b_const_i32(b, 0);
        ckc_value_t* ub   = ckc_b_const_i32(b, (int64_t)iterations);
        ckc_value_t* step = ckc_b_const_i32(b, 1);
        iter_args[0].name = "left";
        iter_args[0].init = ckc_b_const_i32(b, 0);
        iter_args[1].name = "right";
        iter_args[1].init = num_seqs;
        loop              = ckc_b_scf_for_iter(b,
                                  lb,
                                  ub,
                                  step,
                                  iter_args,
                                  2,
                                  "bs_i",
                                  false,
                                  true);
    }

    /* with loop as (_iv, (left, right)): */
    ckc_b_region_enter(b, loop.body);
    left  = loop.iter_vars[0];
    right = loop.iter_vars[1];

    /* done = b.cmp_ge(left, right)
     * mid = b.div(b.add(left, right), b.const_i32(2)) */
    done = ckc_b_cmp_ge(b, left, right);
    {
        /* Sequence the add before the const so the SSA id order matches Python. */
        ckc_value_t* lr_sum = ckc_b_add(b, left, right);
        mid                 = ckc_b_div(b, lr_sum, ckc_b_const_i32(b, 2));
    }

    /* val = b.global_load_i32(cu_q, mid) */
    val = ckc_b_global_load_i32(b, cu_q, mid, 4);

    /* if per_token: mid_val = val
     * else: mid_val = b.add(b.div(val, bq), mid) */
    if(per_token)
    {
        mid_val = val;
    }
    else
    {
        mid_val = ckc_b_add(b, ckc_b_div(b, val, bq), mid);
    }

    /* le = b.cmp_le(mid_val, q_block_global_idx)
     * nl = b.select(le, b.add(mid, 1), left)
     * nr = b.select(le, right, mid) */
    le = ckc_b_cmp_le(b, mid_val, q_block_global_idx);
    nl = ckc_b_select(b, le, ckc_b_add(b, mid, ckc_b_const_i32(b, 1)), left);
    nr = ckc_b_select(b, le, right, mid);

    /* b.scf_yield(b.select(done, left, nl), b.select(done, right, nr)) */
    yields[0] = ckc_b_select(b, done, left, nl);
    yields[1] = ckc_b_select(b, done, right, nr);
    ckc_b_scf_yield(b, yields, 2);
    ckc_b_region_leave(b);

    /* return b.sub(loop.results[0], b.const_i32(1)) */
    res0 = (loop.op != NULL) ? loop.op->results[0] : NULL;
    return ckc_b_sub(b, res0, ckc_b_const_i32(b, 1));
}
