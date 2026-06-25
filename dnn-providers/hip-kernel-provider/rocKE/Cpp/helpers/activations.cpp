// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_ck_dsl.helpers.activations.c -- C99 port of
 * ck_dsl.helpers.activations._sigmoid_via_exp2 / _tanh_via_exp2.
 *
 * Faithful translation of:
 *
 *     def _sigmoid_via_exp2(b: IRBuilder, x: Value) -> Value:
 *         c_neg_log2e = b.const_f32(-1.4426950408889634)
 *         one = b.const_f32(1.0)
 *         return b.rcp(b.fadd(one, b.exp2(b.fmul(c_neg_log2e, x))))
 *
 *     def _tanh_via_exp2(b: IRBuilder, x: Value) -> Value:
 *         c_2log2e = b.const_f32(2.0 * 1.4426950408889634)
 *         one = b.const_f32(1.0)
 *         e2x = b.exp2(b.fmul(c_2log2e, x))
 *         return b.fmul(b.fsub(e2x, one), b.rcp(b.fadd(e2x, one)))
 *
 * The builder-call sequence is reproduced in the exact same order as the Python
 * so the emitted IR (and SSA value numbering) stays byte-identical. Where a
 * Python expression nests several builder calls, C argument-evaluation order is
 * unspecified, so the sub-emissions are sequenced explicitly to match Python's
 * strict left-to-right evaluation of each call's arguments.
 */

#include "ckc/helper_ck_dsl.helpers.activations.h"

#include "ckc/ir_internal.h" /* ckc_i_live */

ckc_value_t* ckc_sigmoid_via_exp2(ckc_ir_builder_t* b, ckc_value_t* x)
{
    ckc_value_t* c_neg_log2e;
    ckc_value_t* one;

    /* Sticky-error model: a failed builder makes every call a NULL no-op. */
    if(!ckc_i_live(b))
    {
        return NULL;
    }

    /* c_neg_log2e = b.const_f32(-1.4426950408889634) */
    c_neg_log2e = ckc_b_const_f32(b, -1.4426950408889634);
    /* one = b.const_f32(1.0) */
    one = ckc_b_const_f32(b, 1.0);
    /* return b.rcp(b.fadd(one, b.exp2(b.fmul(c_neg_log2e, x))))
     *
     * Innermost-out: the fmul, exp2 and fadd are each emitted before rcp.
     * `one` was emitted above; fadd's first argument (`one`) precedes its
     * second (the exp2 chain) in Python's left-to-right argument evaluation,
     * but since `one` is an already-bound Value the only new emissions inside
     * fadd come from the exp2 subexpression -- sequenced here for clarity. */
    {
        ckc_value_t* prod = ckc_b_fmul(b, c_neg_log2e, x);
        ckc_value_t* ex = ckc_b_exp2(b, prod);
        ckc_value_t* sum = ckc_b_fadd(b, one, ex);
        return ckc_b_rcp(b, sum);
    }
}

ckc_value_t* ckc_tanh_via_exp2(ckc_ir_builder_t* b, ckc_value_t* x)
{
    ckc_value_t* c_2log2e;
    ckc_value_t* one;
    ckc_value_t* e2x;

    /* Sticky-error model: a failed builder makes every call a NULL no-op. */
    if(!ckc_i_live(b))
    {
        return NULL;
    }

    /* c_2log2e = b.const_f32(2.0 * 1.4426950408889634) */
    c_2log2e = ckc_b_const_f32(b, 2.0 * 1.4426950408889634);
    /* one = b.const_f32(1.0) */
    one = ckc_b_const_f32(b, 1.0);
    /* e2x = b.exp2(b.fmul(c_2log2e, x)) */
    e2x = ckc_b_exp2(b, ckc_b_fmul(b, c_2log2e, x));
    /* return b.fmul(b.fsub(e2x, one), b.rcp(b.fadd(e2x, one)))
     *
     * Python evaluates fmul's arguments left-to-right: the fsub is emitted
     * before the rcp(fadd) chain, and each consumes SSA value-counter ticks.
     * Sequenced explicitly to keep value numbering byte-identical. */
    {
        ckc_value_t* num = ckc_b_fsub(b, e2x, one);
        ckc_value_t* den = ckc_b_rcp(b, ckc_b_fadd(b, e2x, one));
        return ckc_b_fmul(b, num, den);
    }
}
